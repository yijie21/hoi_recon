# Differentiable render-and-compare object 6D pose optimization (Phase 2 engine).
#
# Runs in the sam3d-objects env (PyTorch3D). Renders the textured SAM-3D mesh into
# each frame with a differentiable renderer and optimizes the per-frame object 6D
# pose against TWO image signals:
#   * silhouette IoU vs the SAM2 mask  (position + visible orientation)
#   * PHOTOMETRIC  vs the RGB frame    (the object's texture/label -> recovers the
#                                       spin-about-axis DOF that silhouette can't see)
# plus a temporal-smoothness term. Init from the silhouette-tracker poses.
#
#   python render_compare.py --mesh m.npz --frames_dir F --masks_dir M --K K.npy \
#       --init_poses p.npz --out out.npz [--scale 0.3] [--iters 150]
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
    ap.add_argument("--w_photo", type=float, default=1.0)
    ap.add_argument("--w_sil", type=float, default=2.0)
    ap.add_argument("--w_temp", type=float, default=5.0)
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

    # load frames + object masks at render res; mark visible frames
    rgb = torch.zeros(T, H, W, 3, device=dev)
    mask = torch.zeros(T, H, W, device=dev)
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
    vis_idx = np.where(visible)[0]

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

    for it in range(a.iters):
        opt.zero_grad()
        total = 0.0
        for s in range(0, len(vis_idx), a.chunk):
            idx = torch.tensor(vis_idx[s:s + a.chunk], device=dev)
            # silhouette
            _, sil = render_chunk(idx, sil_raster, want_rgb=False)
            mk = mask[idx]
            inter = (sil * mk).sum((1, 2)); union = (sil + mk - sil * mk).sum((1, 2))
            l_sil = (1 - inter / union.clamp(min=1)).mean()
            # photometric (object region only)
            rgb_r, alpha = render_chunk(idx, pho_raster, want_rgb=True)
            w = (alpha.detach() * mk)[..., None]
            l_pho = ((rgb_r - rgb[idx]).abs() * w).sum() / w.sum().clamp(min=1)
            loss = a.w_sil * l_sil + a.w_photo * l_pho
            loss.backward()
            total += float(loss)
        # temporal smoothness (whole sequence)
        opt.zero_grad() if False else None
        tl = (a.w_temp * ((transl[1:] - transl[:-1]) ** 2).mean()
              + a.w_temp * ((rot6d[1:] - rot6d[:-1]) ** 2).mean())
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
