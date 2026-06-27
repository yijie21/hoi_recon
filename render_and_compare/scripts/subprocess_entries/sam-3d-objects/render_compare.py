# Differentiable render-and-compare object 6D pose optimization (Phase 2 engine).
#
# Runs in the sam3d-objects env (PyTorch3D). Renders the textured SAM-3D mesh into
# each frame with a differentiable renderer and optimizes the per-frame object 6D
# pose against TWO image signals:
#   * OCCLUSION-ROBUST silhouette vs the SAM2 mask: DON'T-CARE IoU — symmetric
#     IoU with render-over-occluder pixels (optional SAM2 hand mask) excluded
#     from the union. Plain IoU pulls the pose into the visible sliver when the
#     hand occludes the object; a one-sided coverage loss instead rewards
#     inflating the render to swallow the mask (tried: render/mask area -> 1.6x).
#     Don't-care IoU does neither, and reduces to plain IoU when unoccluded.
#   * PHOTOMETRIC  vs the RGB frame    (the object's texture/label -> recovers the
#     spin-about-axis DOF that silhouette can't see; already occlusion-robust: it
#     only compares pixels inside the SAM2 object mask)
# plus temporal smoothness and a translation prior to the depth-lift init (the
# anti-inflation guard: with a one-sided coverage loss alone the object could
# drift toward the camera to cover the mask).
#
#   python render_compare.py --mesh m.npz --frames_dir F --masks_dir M --K K.npy \
#       --init_poses p.npz --out out.npz [--occluder_dir H] [--scale 0.3] [--iters 150]
import os
import sys
import glob
import argparse

import numpy as np
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, help="npz with verts,faces,vertex_colors")
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--masks_dir", required=True)
    ap.add_argument("--K", required=True)
    ap.add_argument("--init_poses", required=True, help="npz with poses[T,4,4]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=0.3, help="render downscale")
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--occluder_dir", default=None,
                    help="per-frame occluder masks (e.g. SAM2 hand masks, %%05d.npy); "
                         "render spilling onto them is not penalized")
    ap.add_argument("--w_photo", type=float, default=1.0)
    ap.add_argument("--w_sil", type=float, default=2.0)
    ap.add_argument("--w_temp", type=float, default=5.0,
                    help="velocity (1st-difference) smoothness on pose")
    ap.add_argument("--w_accel", type=float, default=20.0,
                    help="acceleration (2nd-difference) smoothness — kills the per-frame "
                         "shake that a velocity term alone leaves; weighted higher than "
                         "w_temp because jitter is a 2nd-order phenomenon")
    ap.add_argument("--w_prior", type=float, default=50.0,
                    help="translation prior to the (image-grounded) init poses")
    ap.add_argument("--chunk", type=int, default=24)
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        TexturesVertex, RasterizationSettings, MeshRenderer, MeshRasterizer,
        SoftSilhouetteShader, SoftPhongShader, PointLights, BlendParams)
    from pytorch3d.utils import cameras_from_opencv_projection
    from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
    dev = "cuda"

    m = np.load(a.mesh)
    verts = torch.tensor(m["verts"], dtype=torch.float32, device=dev)
    faces = torch.tensor(m["faces"].astype(np.int64), device=dev)
    vcol = torch.tensor(m["vertex_colors"][:, :3] / 255.0, dtype=torch.float32, device=dev)
    K0 = np.load(a.K).astype(np.float64)
    P0 = np.load(a.init_poses)["poses"].astype(np.float32)
    T = P0.shape[0]
    frames = sorted(glob.glob(os.path.join(a.frames_dir, "*.jpg")))

    H0, W0 = cv2.imread(frames[0]).shape[:2]
    H, W = int(round(H0 * a.scale)), int(round(W0 * a.scale))
    K = K0.copy(); K[:2] *= a.scale
    Kt = torch.tensor(K, dtype=torch.float32, device=dev)

    # load frames + object masks (+ optional occluder masks) at render res
    rgb = torch.zeros(T, H, W, 3, device=dev)
    mask = torch.zeros(T, H, W, device=dev)
    hmask = torch.zeros(T, H, W, device=dev)          # occluder (hand); 0 if absent
    visible = np.zeros(T, bool)
    for t in range(T):
        rgb[t] = torch.tensor(cv2.cvtColor(cv2.resize(cv2.imread(frames[t]), (W, H)),
                                           cv2.COLOR_BGR2RGB) / 255.0, device=dev)
        mp = os.path.join(a.masks_dir, f"{t:05d}.npy")
        if os.path.exists(mp):
            mm = np.load(mp).astype(np.float32)
            mr = cv2.resize(mm, (W, H), interpolation=cv2.INTER_NEAREST)
            mask[t] = torch.tensor(mr, device=dev)
            visible[t] = mr.sum() > 200
        if a.occluder_dir:
            hp = os.path.join(a.occluder_dir, f"{t:05d}.npy")
            if os.path.exists(hp):
                hh = cv2.resize(np.load(hp).astype(np.float32), (W, H),
                                interpolation=cv2.INTER_NEAREST)
                # dilate: exclusion tolerates over-coverage, not under-coverage
                hh = cv2.dilate(hh, np.ones((9, 9), np.float32))
                hmask[t] = torch.tensor(hh, device=dev)
    vis_idx = np.where(visible)[0]
    if a.occluder_dir:
        print(f"[rc] occluder masks: {a.occluder_dir} "
              f"({int((hmask.sum((1,2))>0).sum())}/{T} frames)")

    mesh = Meshes(verts=[verts], faces=[faces],
                  textures=TexturesVertex(vcol[None]))

    sil_raster = RasterizationSettings(image_size=(H, W), blur_radius=2e-4, faces_per_pixel=12)
    pho_raster = RasterizationSettings(image_size=(H, W), blur_radius=0.0, faces_per_pixel=1)
    lights = PointLights(device=dev, ambient_color=((1.0, 1.0, 1.0),),
                         diffuse_color=((0.0, 0.0, 0.0),), specular_color=((0.0, 0.0, 0.0),))
    bp = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0., 0., 0.))
    sil_shader = SoftSilhouetteShader(blend_params=bp)

    # params: per-frame rot (6D) + translation, init from P0
    R0 = torch.tensor(P0[:, :3, :3], device=dev)
    rot6d = matrix_to_rotation_6d(R0).clone().requires_grad_(True)
    transl = torch.tensor(P0[:, :3, 3], device=dev).clone().requires_grad_(True)
    opt = torch.optim.Adam([rot6d, transl], lr=0.01)

    def render_chunk(idx, shader_raster, want_rgb):
        R = rotation_6d_to_matrix(rot6d[idx])
        t = transl[idx]
        n = len(idx)
        cams = cameras_from_opencv_projection(
            R=R, tvec=t, camera_matrix=Kt[None].expand(n, -1, -1),
            image_size=torch.tensor([[H, W]], device=dev).expand(n, -1).float())
        meshes = mesh.extend(n)
        if want_rgb:
            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(cameras=cams, raster_settings=pho_raster),
                shader=SoftPhongShader(device=dev, cameras=cams, lights=lights, blend_params=bp))
            img = renderer(meshes)                      # (n,H,W,4)
            return img[..., :3], img[..., 3]
        else:
            rasterizer = MeshRasterizer(cameras=cams, raster_settings=sil_raster)
            frags = rasterizer(meshes)
            sil = sil_shader(frags, meshes, cameras=cams)   # (n,H,W,4)
            return None, sil[..., 3]

    transl0 = transl.detach().clone()                  # image-grounded init (prior)
    for it in range(a.iters):
        opt.zero_grad()
        total = 0.0
        for s in range(0, len(vis_idx), a.chunk):
            idx = torch.tensor(vis_idx[s:s + a.chunk], device=dev)
            # occlusion-robust silhouette: DON'T-CARE IoU — symmetric IoU with
            # render-over-occluder pixels excluded from the union. With no occluder
            # this is exactly plain IoU (unoccluded frames keep baseline quality);
            # under occlusion the visible sliver must still be covered and spill
            # onto background still costs, but render over the hand is free — so
            # the pose is neither pulled into the sliver (plain IoU failure) nor
            # rewarded for inflating to swallow the mask (one-sided-coverage
            # failure: that variant drove render/mask area to 1.6x).
            _, sil = render_chunk(idx, sil_raster, want_rgb=False)
            mk = mask[idx]; hk = hmask[idx]
            inter = (sil * mk).sum((1, 2))
            union = (sil + mk - sil * mk - sil * (1 - mk) * hk).sum((1, 2))
            l_sil = (1 - inter / union.clamp(min=1)).mean()
            # photometric (object region only — already occlusion-robust)
            rgb_r, alpha = render_chunk(idx, pho_raster, want_rgb=True)
            w = (alpha.detach() * mk)[..., None]
            l_pho = ((rgb_r - rgb[idx]).abs() * w).sum() / w.sum().clamp(min=1)
            loss = a.w_sil * l_sil + a.w_photo * l_pho
            loss.backward()
            total += float(loss)
        # temporal smoothness + translation prior (anti-inflation; the coverage
        # term alone would let the object drift toward the camera)
        tl = (a.w_temp * ((transl[1:] - transl[:-1]) ** 2).mean()
              + a.w_temp * ((rot6d[1:] - rot6d[:-1]) ** 2).mean()
              + a.w_accel * ((transl[2:] - 2 * transl[1:-1] + transl[:-2]) ** 2).mean()
              + a.w_accel * ((rot6d[2:] - 2 * rot6d[1:-1] + rot6d[:-2]) ** 2).mean()
              + a.w_prior * ((transl - transl0) ** 2).mean())
        tl.backward()
        opt.step()
        if it % 30 == 0 or it == a.iters - 1:
            print(f"[rc] iter {it:3d} sil+photo~{total/max(len(range(0,len(vis_idx),a.chunk)),1):.4f} temp {float(tl):.5f}")

    with torch.no_grad():
        Rf = rotation_6d_to_matrix(rot6d).cpu().numpy()
        tf = transl.cpu().numpy()
    poses = np.tile(np.eye(4), (T, 1, 1)).astype(np.float32)
    poses[:, :3, :3] = Rf; poses[:, :3, 3] = tf
    np.savez(a.out, poses=poses, visible=visible)
    print(f"[rc] wrote {a.out} ({len(vis_idx)} frames optimized)")


if __name__ == "__main__":
    main()
