# Joint hand+object render-and-compare optimizer (the redesign's final stage).
#
# Runs in the sam3d-objects env (smplx + PyTorch3D + numpy<2 so MANO loads natively).
# Optimizes, jointly and differentiably:
#   * the MANO hand  -- global orient + wrist translation + 15-joint ARTICULATION
#     + shape betas (so the fingers actually curl to grasp, not rigid motion)
#   * the object 6D pose
# under:
#   L_kp2d         robust (Geman-McClure) 2D keypoint reprojection: projected MANO
#                  joints vs HaMeR's 21 keypoints — the hand's PRIMARY image-space
#                  evidence (the HMP / Dyn-HaMR / RHO recipe; before this loss the
#                  hand had nothing pulling it back to the pixels: median 44px
#                  centroid error vs the object's 7px)
#   L_hand_sil     hand-silhouette precision vs the SAM2 hand mask: rendered MANO
#                  pixels on neither the hand mask nor the object mask (fingers may
#                  be occluded by the held object = don't-care) are penalized; the
#                  mask is NOT required to be covered (SAM2 includes the forearm,
#                  which MANO does not model)
#   L_hand_anchor  MANO verts stay near the (depth-anchored) HaMeR reconstruction
#   L_obj_sil/photo object don't-care IoU + photometric (texture -> spin)
#   L_contact      hand contact verts pulled onto the object surface (grasp frames)
#   L_pen          one-sided non-penetration (hand outside object)
#   L_temporal     velocity + ACCELERATION smoothness on hand+object motion
#   L_prior        MANO pose stays near HaMeR; betas regularized; object stays on
#                  its image-grounded track
# Inputs are the stage bundles (exported to npz by the caller).
import os
import sys
import glob
import argparse

import numpy as np
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", required=True, help="npz: mano_global,mano_pose,mano_betas,verts,joints,contact_idx,hand_faces,hand_side")
    ap.add_argument("--obj", required=True, help="npz: verts,faces,vertex_colors,poses")
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--masks_dir", required=True)
    ap.add_argument("--K", required=True)
    ap.add_argument("--mano_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--occluder_dir", default=None,
                    help="per-frame occluder masks (SAM2 hand, %%05d.npy): object render "
                         "spilling onto them is don't-care in the silhouette term")
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--scale", type=float, default=0.3)
    ap.add_argument("--chunk", type=int, default=24)
    ap.add_argument("--w_prior_obj", type=float, default=50.0,
                    help="object translation prior to the stage-6 (image-grounded) track")
    ap.add_argument("--w_kp2d", type=float, default=3.0,
                    help="robust 2D keypoint reprojection (hand image registration)")
    ap.add_argument("--kp_sigma", type=float, default=60.0,
                    help="Geman-McClure scale for the keypoint residual, full-res px "
                         "(wide: the hand's median kp2d residual floor is ~30px due to "
                         "the metric-depth re-anchoring, so a tight sigma would suppress "
                         "the very corrections we want — robustify only true outliers)")
    ap.add_argument("--w_hsil", type=float, default=1.0,
                    help="hand-silhouette precision vs the SAM2 hand mask")
    ap.add_argument("--w_accel", type=float, default=1.0,
                    help="acceleration (2nd-difference) smoothness on hand+object motion")
    a = ap.parse_args()

    import torch
    import smplx
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (TexturesVertex, RasterizationSettings, MeshRenderer,
                                    MeshRasterizer, SoftSilhouetteShader, SoftPhongShader,
                                    PointLights, BlendParams)
    from pytorch3d.utils import cameras_from_opencv_projection
    from pytorch3d.transforms import (matrix_to_rotation_6d, rotation_6d_to_matrix)
    dev = "cuda"

    H = np.load(a.hand); O = np.load(a.obj)
    mano_global = torch.tensor(H["mano_global"], dtype=torch.float32, device=dev)   # (T,3,3)
    mano_pose = torch.tensor(H["mano_pose"], dtype=torch.float32, device=dev)       # (T,15,3,3)
    betas0 = torch.tensor(H["mano_betas"].mean(0), dtype=torch.float32, device=dev) # (10,)
    hamer_v = torch.tensor(H["verts"], dtype=torch.float32, device=dev)             # (T,778,3)
    cidx = torch.tensor(H["contact_idx"].astype(np.int64), device=dev)
    T = hamer_v.shape[0]
    # Per-frame handedness (1=right, 0=left). HaMeR always yields RIGHT-hand MANO
    # params (left hands are estimated on a mirrored crop), and stage 2 mirrors the
    # resulting verts back (x -> -x). Reproduce that here: run the right-hand MANO
    # layer and mirror its output on left-hand frames, so the optimized hand has
    # the correct chirality and matches the stage-2 evidence (vertex ids, and thus
    # contact_idx, are preserved under the mirror).
    side = H["hand_side"].astype(np.float32) if "hand_side" in H.files else np.ones(T, np.float32)
    mir = torch.ones(T, 1, 3, device=dev)
    mir[:, 0, 0] = torch.tensor(np.where(side > 0.5, 1.0, -1.0), dtype=torch.float32, device=dev)
    hfc = torch.tensor(H["hand_faces"].astype(np.int64), device=dev)   # MANO faces (1538,3)

    ov = torch.tensor(O["verts"], dtype=torch.float32, device=dev)
    ofaces = torch.tensor(O["faces"].astype(np.int64), device=dev)
    ocol = torch.tensor(O["vertex_colors"][:, :3] / 255.0, dtype=torch.float32, device=dev)
    Pobj = O["poses"].astype(np.float32)

    K0 = np.load(a.K).astype(np.float64)
    frames = sorted(glob.glob(os.path.join(a.frames_dir, "*.jpg")))
    H0, W0 = cv2.imread(frames[0]).shape[:2]
    Hh, Ww = int(round(H0 * a.scale)), int(round(W0 * a.scale))
    K = K0.copy(); K[:2] *= a.scale
    Kt = torch.tensor(K, dtype=torch.float32, device=dev)

    # HaMeR 2D keypoints (full-image px, OpenPose 21-joint order) -> render res.
    # Frames with all-zero kp2d (none detected) get zero weight.
    if "kp2d" in H.files:
        kp_t = torch.tensor(H["kp2d"] * a.scale, dtype=torch.float32, device=dev)
    else:
        kp_t = torch.zeros(T, 21, 2, device=dev)
    kpvalid = (kp_t.abs().sum((1, 2)) > 0).float()
    kp_sig2 = float(a.kp_sigma * a.scale) ** 2

    rgb = torch.zeros(T, Hh, Ww, 3, device=dev); omask = torch.zeros(T, Hh, Ww, device=dev)
    hmask = torch.zeros(T, Hh, Ww, device=dev)        # occluder (hand, dilated); 0 if absent
    hraw = torch.zeros(T, Hh, Ww, device=dev)         # hand mask (undilated, for L_hand_sil)
    visible = np.zeros(T, bool)
    for t in range(T):
        rgb[t] = torch.tensor(cv2.cvtColor(cv2.resize(cv2.imread(frames[t]), (Ww, Hh)),
                                           cv2.COLOR_BGR2RGB) / 255.0, device=dev)
        mp = os.path.join(a.masks_dir, f"{t:05d}.npy")
        if os.path.exists(mp):
            mr = cv2.resize(np.load(mp).astype(np.float32), (Ww, Hh), interpolation=cv2.INTER_NEAREST)
            omask[t] = torch.tensor(mr, device=dev); visible[t] = mr.sum() > 200
        if a.occluder_dir:
            hp = os.path.join(a.occluder_dir, f"{t:05d}.npy")
            if os.path.exists(hp):
                hh = cv2.resize(np.load(hp).astype(np.float32), (Ww, Hh),
                                interpolation=cv2.INTER_NEAREST)
                hraw[t] = torch.tensor(hh, device=dev)
                hh = cv2.dilate(hh, np.ones((9, 9), np.float32))   # over-cover on purpose
                hmask[t] = torch.tensor(hh, device=dev)
    vis = np.where(visible)[0]
    if a.occluder_dir:
        print(f"[jopt] occluder masks: {a.occluder_dir} "
              f"({int((hmask.sum((1,2))>0).sum())}/{T} frames)")

    # MANO (right hand) as a LAYER -> takes rotation matrices directly (HaMeR uses
    # this; smplx.MANO mishandles rotmat input via pose_mean). Left-hand frames are
    # handled by mirroring the output (see `mir` above), not a left MANO model.
    mano = smplx.MANOLayer(model_path=a.mano_dir, is_rhand=True).to(dev)
    # HaMeR's 21-joint set: 16 smplx joints + 5 fingertip vertices, remapped to
    # OpenPose order (matches hamer/models/mano_wrapper.py, and thus kp2d).
    from smplx.vertex_ids import vertex_ids
    TIPS = torch.tensor(list(vertex_ids["mano"].values()), dtype=torch.long, device=dev)
    M2O = torch.tensor([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],
                       dtype=torch.long, device=dev)

    def mano_fwd(glob6, pose6, betas):
        go = rotation_6d_to_matrix(glob6).view(T, 1, 3, 3)
        hp = rotation_6d_to_matrix(pose6).view(T, 15, 3, 3)
        out = mano(global_orient=go, hand_pose=hp, betas=betas[None].expand(T, -1))
        j = torch.cat([out.joints, out.vertices[:, TIPS]], dim=1)[:, M2O]
        return out.vertices * mir, j * mir       # (T,778,3),(T,21,3) root-rel, side-corrected

    # init params (6D) + per-frame translation so MANO verts ~ HaMeR verts
    g6 = matrix_to_rotation_6d(mano_global).clone()
    p6 = matrix_to_rotation_6d(mano_pose.reshape(T * 15, 3, 3)).reshape(T, 15, 6).clone()
    with torch.no_grad():
        v0, _ = mano_fwd(g6, p6, betas0)
        transl0 = hamer_v.mean(1) - v0.mean(1)                # align centroids
    g6 = g6.requires_grad_(True); p6 = p6.clone().requires_grad_(True)
    transl = transl0.clone().requires_grad_(True)
    betas = betas0.clone().requires_grad_(True)
    o_r6 = matrix_to_rotation_6d(torch.tensor(Pobj[:, :3, :3], device=dev)).requires_grad_(True)
    o_t = torch.tensor(Pobj[:, :3, 3], device=dev).clone().requires_grad_(True)
    o_t0 = torch.tensor(Pobj[:, :3, 3], device=dev)   # image-grounded init (prior)
    opt = torch.optim.Adam([
        {"params": [g6, transl, o_r6, o_t], "lr": 0.006},
        {"params": [p6, betas], "lr": 0.003}], )

    # object renderers
    omesh = Meshes(verts=[ov], faces=[ofaces], textures=TexturesVertex(ocol[None]))
    sil_rs = RasterizationSettings(image_size=(Hh, Ww), blur_radius=2e-4, faces_per_pixel=10)
    pho_rs = RasterizationSettings(image_size=(Hh, Ww), blur_radius=0.0, faces_per_pixel=1)
    bp = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0., 0., 0.))
    lights = PointLights(device=dev, ambient_color=((1.,1.,1.),),
                         diffuse_color=((0.,0.,0.),), specular_color=((0.,0.,0.),))
    silsh = SoftSilhouetteShader(blend_params=bp)
    ofn = torch.tensor(Meshes(verts=[ov], faces=[ofaces]).verts_normals_packed(), device=dev)

    grasp = torch.tensor(visible, device=dev)
    cache = {}

    def obj_world(t_idx):
        R = rotation_6d_to_matrix(o_r6[t_idx]); t = o_t[t_idx]
        return torch.einsum("nij,vj->nvi", R, ov) + t[:, None], R

    for it in range(a.iters):
        opt.zero_grad()
        hv_r, jh_r = mano_fwd(g6, p6, betas)
        hv = hv_r + transl[:, None]                           # (T,778,3) camera frame
        jh = jh_r + transl[:, None]                           # (T,21,3)
        loss = 0.0
        # 2D keypoint reprojection (Geman-McClure robust) — the hand's primary
        # image-space evidence: projected MANO joints vs HaMeR's kp2d.
        z = jh[..., 2].clamp(min=1e-4)
        u = Kt[0, 0] * jh[..., 0] / z + Kt[0, 2]
        v2 = Kt[1, 1] * jh[..., 1] / z + Kt[1, 2]
        r2 = (u - kp_t[..., 0]) ** 2 + (v2 - kp_t[..., 1]) ** 2
        loss = loss + a.w_kp2d * ((r2 / (r2 + kp_sig2)) * kpvalid[:, None]).mean()
        # hand anchor, split by axis: STRONG on z — the metric-depth anchor is the
        # only depth evidence (kp2d is blind in z) — but WEAK on x/y, where the
        # anchor is just the box-centroid heuristic and kp2d is the better
        # evidence (a uniform 6.0 anchor capped the kp2d fit at ~32px residual).
        d_anchor = (hv - hamer_v) ** 2
        loss = loss + 6.0 * d_anchor[..., 2].mean() + 1.0 * d_anchor[..., :2].mean()
        loss = loss + 2.0 * ((p6[1:] - p6[:-1]) ** 2).mean() + 2.0 * ((transl[1:]-transl[:-1])**2).mean()
        loss = loss + 0.5 * ((p6 - matrix_to_rotation_6d(mano_pose.reshape(-1,3,3)).reshape(T,15,6)) ** 2).mean()
        loss = loss + 0.01 * (betas ** 2).mean()
        loss = loss + 2.0 * ((o_t[1:]-o_t[:-1])**2).mean() + 2.0 * ((o_r6[1:]-o_r6[:-1])**2).mean()
        # acceleration (2nd difference) smoothness — kills residual jitter that
        # velocity terms alone leave (HMP-style)
        for q in (transl, p6.reshape(T, -1), o_t, o_r6):
            loss = loss + a.w_accel * ((q[2:] - 2 * q[1:-1] + q[:-2]) ** 2).mean()
        # object translation prior to the image-grounded stage-6 track (anti-
        # inflation guard for the one-sided silhouette coverage term below)
        loss = loss + a.w_prior_obj * ((o_t - o_t0) ** 2).mean()
        loss.backward(retain_graph=False)

        # contact + penetration + object image losses (chunked over visible frames)
        if it % 10 == 0:
            with torch.no_grad():
                hv_d = (mano_fwd(g6, p6, betas)[0] + transl[:, None]).detach()
                for t in vis:
                    ow, _ = obj_world(torch.tensor([t], device=dev))
                    d = torch.cdist(hv_d[t][cidx], ow[0]); cache[t] = d.argmin(1)
        for s in range(0, len(vis), a.chunk):
            ti = torch.tensor(vis[s:s+a.chunk], device=dev)
            hv = mano_fwd(g6, p6, betas)[0] + transl[:, None]
            ow, R = obj_world(ti)
            nw = torch.einsum("nij,vj->nvi", R, ofn)
            # object silhouette + photometric
            cams = cameras_from_opencv_projection(R=R, tvec=o_t[ti],
                camera_matrix=Kt[None].expand(len(ti), -1, -1),
                image_size=torch.tensor([[Hh, Ww]], device=dev).expand(len(ti), -1).float())
            meshes = omesh.extend(len(ti))
            sil = silsh(MeshRasterizer(cameras=cams, raster_settings=sil_rs)(meshes), meshes, cameras=cams)[..., 3]
            # occlusion-robust silhouette: DON'T-CARE IoU — render-over-occluder
            # pixels excluded from the union (plain IoU when no occluder; under
            # occlusion neither pulled into the visible sliver nor rewarded for
            # inflating over the hand region)
            mk = omask[ti]; hk = hmask[ti]
            inter = (sil*mk).sum((1,2))
            union = (sil + mk - sil*mk - sil*(1-mk)*hk).sum((1,2))
            l_sil = (1 - inter / union.clamp(min=1)).mean()
            img = MeshRenderer(rasterizer=MeshRasterizer(cameras=cams, raster_settings=pho_rs),
                               shader=SoftPhongShader(device=dev, cameras=cams, lights=lights, blend_params=bp))(meshes)
            w = (img[...,3].detach()*mk)[...,None]
            l_pho = ((img[...,:3]-rgb[ti]).abs()*w).sum()/w.sum().clamp(min=1)
            # hand-silhouette precision vs the SAM2 hand mask: rendered hand pixels
            # on the background are penalized; pixels on the object mask are
            # don't-care (fingers may be occluded BY the object); the hand mask is
            # not required to be covered (it includes the forearm, MANO does not).
            l_hsil = torch.zeros((), device=dev)
            hk_raw = hraw[ti]
            has_h = (hk_raw.sum((1, 2)) > 200).float()
            if a.w_hsil > 0 and float(has_h.sum()) > 0:
                n = len(ti)
                hmesh = Meshes(verts=[hv[t] for t in ti.tolist()], faces=[hfc] * n)
                cam0 = cameras_from_opencv_projection(
                    R=torch.eye(3, device=dev)[None].expand(n, -1, -1),
                    tvec=torch.zeros(n, 3, device=dev),
                    camera_matrix=Kt[None].expand(n, -1, -1),
                    image_size=torch.tensor([[Hh, Ww]], device=dev).expand(n, -1).float())
                hsil_r = silsh(MeshRasterizer(cameras=cam0, raster_settings=sil_rs)(hmesh),
                               hmesh, cameras=cam0)[..., 3]
                bad = hsil_r * (1 - hk_raw) * (1 - omask[ti])
                l_hsil = ((bad.sum((1, 2)) / hsil_r.sum((1, 2)).clamp(min=1)) * has_h).sum() \
                    / has_h.sum().clamp(min=1)
            # contact + penetration (grasp frames)
            lc = 0.0; lp = 0.0
            for k, t in enumerate(ti.tolist()):
                hc = hv[t][cidx]; anc = ow[k][cache[t]]; nrm = nw[k][cache[t]]
                diff = hc - anc; signed = (diff*nrm).sum(1)
                lc = lc + (diff**2).sum(1).mean()
                # penetration over all hand verts
                dall = torch.cdist(hv[t], ow[k]); idx = dall.argmin(1)
                s2 = ((hv[t]-ow[k][idx])*nw[k][idx]).sum(1)
                lp = lp + torch.relu(-s2).pow(2).mean()
            cl = (3.0*l_sil + 1.0*l_pho + a.w_hsil*l_hsil + 5.0*lc/len(ti) + 30.0*lp/len(ti))
            cl.backward()
        opt.step()
        if it % 30 == 0 or it == a.iters-1:
            with torch.no_grad():
                kp_px = float((r2.clamp(min=0).sqrt() * kpvalid[:, None]).sum()
                              / kpvalid.sum().clamp(min=1) / 21 / a.scale)
            print(f"[jopt] iter {it:3d} anchor+temp {float(loss):.4f} kp2d~{kp_px:.1f}px")

    with torch.no_grad():
        hv_r, jh_r = mano_fwd(g6, p6, betas)
        hv = (hv_r + transl[:, None]).cpu().numpy()
        jh = (jh_r + transl[:, None]).cpu().numpy()
        Rf = rotation_6d_to_matrix(o_r6).cpu().numpy(); tf = o_t.cpu().numpy()
    poses = np.tile(np.eye(4), (T,1,1)).astype(np.float32); poses[:,:3,:3]=Rf; poses[:,:3,3]=tf
    np.savez(a.out, hand_verts=hv, hand_joints=jh, obj_poses=poses, visible=visible)
    print(f"[jopt] wrote {a.out}")


if __name__ == "__main__":
    main()
