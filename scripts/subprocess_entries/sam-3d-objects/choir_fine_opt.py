# CHOIR Stage-3 joint hand+object optimizer (registry-based).
#
# Adapted from joint_opt.py. Runs in the sam3d-objects env (smplx + PyTorch3D).
# Replaces the hand-coded weighted loss with the tested CHOIR energy-term registry
# from hoi_recon.choir_fine (imported via PYTHONPATH set by the driver).
#
# Loss structure (per the CHOIR spec + presets):
#   - Geometric terms: contact, pen, anc_2d, anc_anat, anc_pose_h, anc_pose_o,
#       temp_pose_vel, temp_obj_vel, temp_wrist_anchor, temp_hand_tr_vel,
#       temp_root_R_vel, temp_pose_acc, temp_hand_tr_acc, temp_root_R_acc
#       (all computed by choir_fine.step.compute_geometric_terms)
#   - Render terms: sil (object don't-care IoU), hand_sil (hand silhouette precision)
#       (computed by the existing PyTorch3D render loop, unchanged)
#   - Contact-family stabilizers: template, bridge, gap, patch -- zeroed here;
#       full annealing schedule deferred to a follow-on plan
#
# Contact anchor approach (DOCUMENTED CHOICE):
#   The plan allows using either the faithful barycentric build_correspondences
#   (choir_fine.contact) or the existing torch nearest-anchor cache mechanism.
#   We use the EXISTING TORCH NEAREST-ANCHOR CACHE (mirroring joint_opt.py):
#     - Every 10 iters, for each visible frame t, we build cache[t] = argmin of
#       cdist(contact_verts, object_world_verts) -> (Nc,) index into obj surface.
#     - We then gather anc_pts (T,Nc,K=1,3) and set anc_w uniform (=1.0 for K=1),
#       with conf = phase_mask (manipulation frames only, via choir_phases).
#   Rationale: choir_fine.contact.build_correspondences uses trimesh CPU ops
#   (non-differentiable, numpy) and rebuilds a 10K surface sample + cKDTree every
#   call -- expensive in the inner loop. The torch cdist cache is differentiable
#   (the anchors are gathered from ow which is a function of o_r6/o_t), cheap, and
#   matches the existing optimizer's cadence. The contact_loss function in
#   terms_torch.py supports K=1 anchors with anc_w=1.0 trivially.
#
# Penetration shape handling:
#   pen_hand (T,Nh,3) = hv (all hand verts, camera frame)
#   pen_surf (T,Nh,3) = nearest object surface point per hand vertex (from ow)
#   pen_normal (T,Nh,3) = outward normal at that surface point (from nw)
#   Built from FULL-SEQUENCE tensors every 10 iters (same cadence as contact cache).
#   Between cache updates they use the stale pen_surf/pen_normal from the last
#   rebuild, but the gradient still flows through hv (the hand vertices), and the
#   loss keeps pushing the hand out of the object.
import os
import sys
import glob
import argparse
import json

# hoi_recon.choir_fine is importable via PYTHONPATH (set by the driver). Import the
# tested energy-term library so this optimizer only *assembles* validated pieces.
from hoi_recon.choir_fine import step as choir_step
from hoi_recon.choir_fine import registry as choir_registry
from hoi_recon.choir_fine import contact as choir_contact
from hoi_recon.choir_fine import phases as choir_phases

import numpy as np
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", required=True,
                    help="npz: mano_global,mano_pose,mano_betas,verts,joints,contact_idx,hand_faces,hand_side")
    ap.add_argument("--obj", required=True, help="npz: verts,faces,vertex_colors,poses")
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--masks_dir", required=True)
    ap.add_argument("--K", required=True)
    ap.add_argument("--mano_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--occluder_dir", default=None,
                    help="per-frame occluder masks (SAM2 hand, %%05d.npy): object render "
                         "spilling onto them is don't-care in the silhouette term")
    ap.add_argument("--weights", required=True, help="json file: {term_name: weight} preset")
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--scale", type=float, default=0.3)
    ap.add_argument("--chunk", type=int, default=24)
    ap.add_argument("--lr_object", type=float, default=3e-4)
    ap.add_argument("--lr_finger", type=float, default=5e-4)
    ap.add_argument("--lr_wrist", type=float, default=5e-5)
    a = ap.parse_args()

    import torch
    import smplx
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (TexturesVertex, RasterizationSettings, MeshRenderer,
                                    MeshRasterizer, SoftSilhouetteShader, SoftPhongShader,
                                    PointLights, BlendParams)
    from pytorch3d.utils import cameras_from_opencv_projection
    from pytorch3d.transforms import (matrix_to_rotation_6d, rotation_6d_to_matrix,
                                      matrix_to_axis_angle)
    dev = "cuda"

    # ------------------------------------------------------------------ helpers
    def rot6d_to_aa(p6):
        """Convert (T,15,6) rot6d to (T,15,3) axis-angle for anatomical loss."""
        R = rotation_6d_to_matrix(p6)               # (T,15,3,3)
        shape = R.shape[:-2]
        return matrix_to_axis_angle(R.reshape(-1, 3, 3)).reshape(*shape, 3)

    # ------------------------------------------------------------------ load
    weights = json.load(open(a.weights))

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

    rgb = torch.zeros(T, Hh, Ww, 3, device=dev); omask = torch.zeros(T, Hh, Ww, device=dev)
    hmask = torch.zeros(T, Hh, Ww, device=dev)        # occluder (hand, dilated); 0 if absent
    hraw = torch.zeros(T, Hh, Ww, device=dev)         # hand mask (undilated, for hand_sil)
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
        print(f"[choir_fine_opt] occluder masks: {a.occluder_dir} "
              f"({int((hmask.sum((1,2))>0).sum())}/{T} frames)")

    # Compute phase mask: contact terms active on manipulation-phase frames only.
    # Use 'visible' as a proxy for contact presence (object visible -> hand near it).
    phase_labels = choir_phases.segment_phases(visible)
    manip_idx = choir_phases.PHASES.index("manipulation")
    phase_mask = torch.tensor(
        (phase_labels == manip_idx).astype(np.float32), device=dev)  # (T,)

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
    o_t0 = torch.tensor(Pobj[:, :3, 3], device=dev)   # image-grounded init (transl prior)
    o_r60 = matrix_to_rotation_6d(torch.tensor(Pobj[:, :3, :3], device=dev))  # rot prior

    # Save initial hand wrist-anchor from HaMeR centroid (proxy: hamer_v.mean(1))
    hamer_j0 = hamer_v.mean(1).detach()   # (T,3) wrist init proxy

    # CHOIR per-group Adam: separate LRs for object, fingers, and wrist/transl
    opt = torch.optim.Adam([
        {"params": [o_r6, o_t], "lr": a.lr_object},
        {"params": [p6, betas], "lr": a.lr_finger},
        {"params": [g6, transl], "lr": a.lr_wrist},
    ])

    # object renderers
    omesh = Meshes(verts=[ov], faces=[ofaces], textures=TexturesVertex(ocol[None]))
    sil_rs = RasterizationSettings(image_size=(Hh, Ww), blur_radius=2e-4, faces_per_pixel=10)
    pho_rs = RasterizationSettings(image_size=(Hh, Ww), blur_radius=0.0, faces_per_pixel=1)
    bp = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0., 0., 0.))
    lights = PointLights(device=dev, ambient_color=((1.,1.,1.),),
                         diffuse_color=((0.,0.,0.),), specular_color=((0.,0.,0.),))
    silsh = SoftSilhouetteShader(blend_params=bp)
    ofn = torch.tensor(Meshes(verts=[ov], faces=[ofaces]).verts_normals_packed(), device=dev)

    Nc = len(cidx)
    Nh = hamer_v.shape[1]

    # Caches for contact anchors and penetration (rebuilt every 10 iters)
    # cache[t] = (Nc,) int tensor: index of nearest obj surface vert per contact vert
    cache = {}
    # Full-sequence penetration cache: pen_surf (T,Nh,3), pen_normal (T,Nh,3)
    pen_surf_cache = torch.zeros(T, Nh, 3, device=dev)
    pen_normal_cache = torch.ones(T, Nh, 3, device=dev)

    def obj_world(t_idx):
        R = rotation_6d_to_matrix(o_r6[t_idx]); t = o_t[t_idx]
        return torch.einsum("nij,vj->nvi", R, ov) + t[:, None], R

    for it in range(a.iters):
        opt.zero_grad()
        hv_r, jh_r = mano_fwd(g6, p6, betas)
        hv = hv_r + transl[:, None]                           # (T,778,3) camera frame
        jh = jh_r + transl[:, None]                           # (T,21,3)

        # -----------------------------------------------------------
        # Rebuild contact anchor cache and penetration cache every 10 iters
        # -----------------------------------------------------------
        if it % 10 == 0:
            with torch.no_grad():
                hv_d = hv.detach()
                all_ti = torch.arange(T, device=dev)
                ow_all, R_all = obj_world(all_ti)   # (T,No,3), (T,3,3)
                nw_all = torch.einsum("nij,vj->nvi", R_all, ofn)  # (T,No,3)
                for t in range(T):
                    if visible[t]:
                        # contact anchor: nearest obj vert per contact vert (visible frames)
                        d_c = torch.cdist(hv_d[t][cidx], ow_all[t])  # (Nc,No)
                        cache[t] = d_c.argmin(1)                       # (Nc,)
                    # penetration: nearest obj vert per hand vert (all frames)
                    d_p = torch.cdist(hv_d[t], ow_all[t])             # (Nh,No)
                    pen_idx = d_p.argmin(1)                            # (Nh,)
                    pen_surf_cache[t] = ow_all[t][pen_idx]
                    pen_normal_cache[t] = nw_all[t][pen_idx]

        # -----------------------------------------------------------
        # Build anc_pts (T,Nc,1,3), anc_w (T,Nc,1), conf (T,Nc)
        # from the torch nearest-anchor cache.
        # K=1 so anc_w = 1.0 and the contact_loss reduces to plain L2.
        # conf = phase_mask (manipulation frames only, broadcast to Nc).
        # -----------------------------------------------------------
        all_ti = torch.arange(T, device=dev)
        ow_curr, _ = obj_world(all_ti)   # (T,No,3) -- differentiable via o_r6/o_t

        # Build anc_pts: for frames in cache, gather from differentiable ow_curr;
        # for non-visible frames use zeros (conf=0 => terms masked out). (T,Nc,1,3)
        anc_list = []
        for t in range(T):
            if t in cache:
                anc_list.append(ow_curr[t][cache[t]])  # (Nc,3) differentiable
            else:
                anc_list.append(torch.zeros(Nc, 3, device=dev))
        anc_pts = torch.stack(anc_list, dim=0).unsqueeze(2)  # (T,Nc,1,3)
        anc_w = torch.ones(T, Nc, 1, device=dev)             # uniform, K=1
        # conf: (T,Nc) -- manipulation-phase mask, broadcast over contact verts
        conf = phase_mask.unsqueeze(1).expand(T, Nc)

        # Penetration: use stale-but-detached cache; grad flows through hv
        pen_surf = pen_surf_cache.detach()      # (T,Nh,3)
        pen_normal = pen_normal_cache.detach()  # (T,Nh,3)

        # -----------------------------------------------------------
        # Assemble state dict and compute geometric terms via registry
        # -----------------------------------------------------------
        state = {
            "hand_c": hv[:, cidx],               # (T,Nc,3) contact verts
            "anchors": anc_pts,                   # (T,Nc,1,3) object surface anchors
            "anc_w": anc_w,                       # (T,Nc,1) uniform K=1
            "conf": conf,                         # (T,Nc) manipulation mask
            "pen_hand": hv,                       # (T,Nh,3) all hand verts
            "pen_surf": pen_surf,                 # (T,Nh,3) nearest obj surface pts
            "pen_normal": pen_normal,             # (T,Nh,3) outward normals
            "joints_cam": jh,                     # (T,21,3)
            "kp2d": kp_t,                         # (T,21,2) scaled 2D keypoints
            "K": Kt,                              # (3,3)
            "kp_valid": kpvalid,                  # (T,)
            "mano_pose": rot6d_to_aa(p6),         # (T,15,3) axis-angle for anatomical
            "hand_verts": hv,                     # (T,778,3)
            "hand_verts_init": hamer_v,           # (T,778,3) HaMeR init (detached)
            "o_t": o_t,                           # (T,3)
            "o_t0": o_t0,                         # (T,3) init prior
            "o_r6": o_r6,                         # (T,6)
            "o_r60": o_r60,                       # (T,6) init prior
            "p6": p6,                             # (T,15,6)
            "transl": transl,                     # (T,3)
            "g6": g6,                             # (T,6)
            "wrist": jh[:, 0],                    # (T,3) MANO wrist joint (joint 0)
            "wrist_init": hamer_j0,               # (T,3) init wrist (centroid proxy)
        }
        values = choir_step.compute_geometric_terms(state)

        # -----------------------------------------------------------
        # Render terms: object don't-care IoU sil + hand_sil
        # Computed chunked over visible frames (mirrors joint_opt.py).
        # -----------------------------------------------------------
        l_sil_total = torch.zeros((), device=dev)
        l_hsil_total = torch.zeros((), device=dev)
        n_chunks = 0

        for s in range(0, len(vis), a.chunk):
            ti = torch.tensor(vis[s:s+a.chunk], device=dev)
            hv_chunk = mano_fwd(g6, p6, betas)[0] + transl[:, None]   # (T,778,3)
            ow_c, R_c = obj_world(ti)
            # object silhouette + photometric
            cams = cameras_from_opencv_projection(R=R_c, tvec=o_t[ti],
                camera_matrix=Kt[None].expand(len(ti), -1, -1),
                image_size=torch.tensor([[Hh, Ww]], device=dev).expand(len(ti), -1).float())
            meshes = omesh.extend(len(ti))
            sil = silsh(MeshRasterizer(cameras=cams, raster_settings=sil_rs)(meshes), meshes, cameras=cams)[..., 3]
            # occlusion-robust silhouette: DON'T-CARE IoU
            mk = omask[ti]; hk = hmask[ti]
            inter = (sil*mk).sum((1,2))
            union = (sil + mk - sil*mk - sil*(1-mk)*hk).sum((1,2))
            l_sil_chunk = (1 - inter / union.clamp(min=1)).mean()
            img = MeshRenderer(rasterizer=MeshRasterizer(cameras=cams, raster_settings=pho_rs),
                               shader=SoftPhongShader(device=dev, cameras=cams, lights=lights, blend_params=bp))(meshes)
            w = (img[...,3].detach()*mk)[...,None]
            l_pho = ((img[...,:3]-rgb[ti]).abs()*w).sum()/w.sum().clamp(min=1)
            # hand-silhouette precision vs the SAM2 hand mask: rendered hand pixels
            # on the background are penalized; pixels on the object mask are
            # don't-care (fingers may be occluded BY the object); the hand mask is
            # not required to be covered (it includes the forearm, MANO does not).
            l_hsil_chunk = torch.zeros((), device=dev)
            hk_raw = hraw[ti]
            has_h = (hk_raw.sum((1, 2)) > 200).float()
            if float(has_h.sum()) > 0:
                n = len(ti)
                hmesh = Meshes(verts=[hv_chunk[t] for t in ti.tolist()], faces=[hfc] * n)
                cam0 = cameras_from_opencv_projection(
                    R=torch.eye(3, device=dev)[None].expand(n, -1, -1),
                    tvec=torch.zeros(n, 3, device=dev),
                    camera_matrix=Kt[None].expand(n, -1, -1),
                    image_size=torch.tensor([[Hh, Ww]], device=dev).expand(n, -1).float())
                hsil_r = silsh(MeshRasterizer(cameras=cam0, raster_settings=sil_rs)(hmesh),
                               hmesh, cameras=cam0)[..., 3]
                bad = hsil_r * (1 - hk_raw) * (1 - omask[ti])
                l_hsil_chunk = ((bad.sum((1, 2)) / hsil_r.sum((1, 2)).clamp(min=1)) * has_h).sum() \
                    / has_h.sum().clamp(min=1)

            # Combine sil + photometric (l_pho scaled 1/3 relative to l_sil, matching
            # joint_opt.py's 3.0*l_sil + 1.0*l_pho where sil weight is 3x pho)
            l_sil_total = l_sil_total + l_sil_chunk + l_pho / 3.0
            l_hsil_total = l_hsil_total + l_hsil_chunk
            n_chunks += 1

        n_chunks_safe = max(n_chunks, 1)
        values["sil"] = l_sil_total / n_chunks_safe       # scalar: object sil + photo
        values["hand_sil"] = l_hsil_total / n_chunks_safe  # scalar: hand sil precision
        for k in ("template", "bridge", "gap", "patch"):
            values[k] = torch.zeros((), device=dev)        # annealed (start 0)

        loss = choir_registry.assemble_energy(weights, values)
        opt.zero_grad(); loss.backward(); opt.step()

        if it % 30 == 0 or it == a.iters-1:
            with torch.no_grad():
                z2 = jh[..., 2].clamp(min=1e-4)
                u2 = Kt[0, 0] * jh[..., 0] / z2 + Kt[0, 2]
                v2 = Kt[1, 1] * jh[..., 1] / z2 + Kt[1, 2]
                r2 = (u2 - kp_t[..., 0]) ** 2 + (v2 - kp_t[..., 1]) ** 2
                kp_px = float((r2.clamp(min=0).sqrt() * kpvalid[:, None]).sum()
                              / kpvalid.sum().clamp(min=1) / 21 / a.scale)
            print(f"[choir_fine_opt] iter {it:3d} loss {float(loss):.4f} kp2d~{kp_px:.1f}px")

    with torch.no_grad():
        hv_r, jh_r = mano_fwd(g6, p6, betas)
        hv_out = (hv_r + transl[:, None]).cpu().numpy()
        jh_out = (jh_r + transl[:, None]).cpu().numpy()
        Rf = rotation_6d_to_matrix(o_r6).cpu().numpy(); tf = o_t.cpu().numpy()
    poses = np.tile(np.eye(4), (T,1,1)).astype(np.float32); poses[:,:3,:3]=Rf; poses[:,:3,3]=tf
    np.savez(a.out, hand_verts=hv_out, hand_joints=jh_out, obj_poses=poses, visible=visible)
    print(f"[choir_fine_opt] wrote {a.out}")


if __name__ == "__main__":
    main()
