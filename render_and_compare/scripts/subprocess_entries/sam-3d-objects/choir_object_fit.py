# CHOIR object isolated fit (Stage 1, Eq 2-3) — faithful reproduction.
#
# Runs in the sam3d-objects env (PyTorch3D). Given the FIXED SAM-3D anchor mesh and
# per-frame init 6D poses (from the guarded follow-tracker), refines per-frame
# rotation+translation against the SAM2 object mask using CHOIR's *silhouette-only*
# objective (no photometric term — that is this repo's addition, not CHOIR's):
#
#   L_o = lambda_rep * L_rep + lambda_attr * L_attr + lambda_temp * L_temp + lambda_stat * L_stat
#
#   L_rep  = mean_Omega [ max(M_render - M_target, 0) ]^2          # render outside the mask
#   L_attr = mean_{p in S} min_{v in V} || p - Pi(v) ||^2          # pull mesh to UNCOVERED mask px
#   L_temp = mean (rot6d[t]-rot6d[t-1])^2 + mean (transl[t]-transl[t-1])^2
#   L_stat = mean || transl - transl_init ||^2 + || rot6d - rot6d_init ||^2   # anchor to init
#
# Paper weights: lambda_rep=1.5, lambda_attr=1.0, lambda_temp=10, lambda_stat=1, Adam 1e-3, 500 it.
import os
import sys
import glob
import argparse

import numpy as np
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, help="npz with verts,faces (SAM-3D anchor mesh)")
    ap.add_argument("--masks_dir", required=True)
    ap.add_argument("--frames_dir", required=True, help="only used for image size")
    ap.add_argument("--K", required=True)
    ap.add_argument("--init_poses", required=True, help="npz with poses[T,4,4] (follow-tracker init)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=0.3)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda_rep", type=float, default=1.5)
    ap.add_argument("--lambda_attr", type=float, default=1.0)
    ap.add_argument("--lambda_temp", type=float, default=10.0)
    ap.add_argument("--lambda_stat", type=float, default=1.0)
    ap.add_argument("--attr_px", type=int, default=200, help="uncovered mask px sampled / frame")
    ap.add_argument("--chunk", type=int, default=24)
    a = ap.parse_args()

    import torch
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (RasterizationSettings, MeshRasterizer,
                                    SoftSilhouetteShader, BlendParams)
    from pytorch3d.utils import cameras_from_opencv_projection
    from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
    dev = "cuda"

    m = np.load(a.mesh)
    verts = torch.tensor(m["verts"], dtype=torch.float32, device=dev)
    faces = torch.tensor(m["faces"].astype(np.int64), device=dev)
    K0 = np.load(a.K).astype(np.float64)
    P0 = np.load(a.init_poses)["poses"].astype(np.float32)
    T = P0.shape[0]
    frames = sorted(glob.glob(os.path.join(a.frames_dir, "*.jpg")))
    H0, W0 = cv2.imread(frames[0]).shape[:2]
    H, W = int(round(H0 * a.scale)), int(round(W0 * a.scale))
    K = K0.copy(); K[:2] *= a.scale
    Kt = torch.tensor(K, dtype=torch.float32, device=dev)

    mask = torch.zeros(T, H, W, device=dev)
    visible = np.zeros(T, bool)
    for t in range(T):
        mp = os.path.join(a.masks_dir, f"{t:05d}.npy")
        if os.path.exists(mp):
            mr = cv2.resize(np.load(mp).astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
            mask[t] = torch.tensor(mr, device=dev); visible[t] = mr.sum() > 200
    vis_idx = np.where(visible)[0]

    mesh = Meshes(verts=[verts], faces=[faces])
    raster = RasterizationSettings(image_size=(H, W), blur_radius=2e-4, faces_per_pixel=12)
    bp = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0., 0., 0.))
    silsh = SoftSilhouetteShader(blend_params=bp)

    rot6d = matrix_to_rotation_6d(torch.tensor(P0[:, :3, :3], device=dev)).clone().requires_grad_(True)
    transl = torch.tensor(P0[:, :3, 3], device=dev).clone().requires_grad_(True)
    rot0 = rot6d.detach().clone(); t0 = transl.detach().clone()        # static anchor (L_stat)
    opt = torch.optim.Adam([rot6d, transl], lr=a.lr)

    # pixel grid (render res) for the attraction term
    ys, xs = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
    grid = torch.stack([xs, ys], -1).float()                          # (H,W,2)

    def render_sil(idx):
        R = rotation_6d_to_matrix(rot6d[idx]); t = transl[idx]; n = len(idx)
        cams = cameras_from_opencv_projection(R=R, tvec=t,
            camera_matrix=Kt[None].expand(n, -1, -1),
            image_size=torch.tensor([[H, W]], device=dev).expand(n, -1).float())
        meshes = mesh.extend(n)
        sil = silsh(MeshRasterizer(cameras=cams, raster_settings=raster)(meshes),
                    meshes, cameras=cams)[..., 3]                     # (n,H,W) soft alpha
        # projected vertices for the attraction term
        vproj = cams.transform_points_screen(verts[None].expand(n, -1, -1),
                                             image_size=((H, W),))[..., :2]  # (n,Nv,2)
        return sil, vproj

    for it in range(a.iters):
        opt.zero_grad()
        total = 0.0
        for s in range(0, len(vis_idx), a.chunk):
            idx = torch.tensor(vis_idx[s:s + a.chunk], device=dev)
            sil, vproj = render_sil(idx)
            mk = mask[idx]
            # repulsion: render on, mask off
            l_rep = torch.relu(sil - mk).pow(2).mean()
            # attraction: uncovered mask pixels -> nearest projected vertex
            l_attr = sil.new_zeros(())
            for k in range(len(idx)):
                unc = (mk[k] > 0.5) & (sil[k].detach() < 0.5)         # mask px not yet covered
                pts = grid[unc]
                if pts.shape[0] == 0:
                    continue
                if pts.shape[0] > a.attr_px:
                    sel = torch.randperm(pts.shape[0], device=dev)[:a.attr_px]
                    pts = pts[sel]
                d = torch.cdist(pts, vproj[k])                        # (P, Nv)
                l_attr = l_attr + d.min(1).values.pow(2).mean()
            l_attr = l_attr / max(len(idx), 1)
            loss = a.lambda_rep * l_rep + a.lambda_attr * l_attr
            loss.backward()
            total += float(loss)
        # temporal smoothness (rotation + translation) + static anchor — full sequence
        tl = (a.lambda_temp * ((transl[1:] - transl[:-1]) ** 2).mean()
              + a.lambda_temp * ((rot6d[1:] - rot6d[:-1]) ** 2).mean()
              + a.lambda_stat * ((transl - t0) ** 2).mean()
              + a.lambda_stat * ((rot6d - rot0) ** 2).mean())
        tl.backward()
        opt.step()
        if it % 50 == 0 or it == a.iters - 1:
            print(f"[choir_obj] iter {it:3d} rep+attr~{total/max(len(range(0,len(vis_idx),a.chunk)),1):.4f} temp {float(tl):.5f}")

    with torch.no_grad():
        Rf = rotation_6d_to_matrix(rot6d).cpu().numpy(); tf = transl.cpu().numpy()
    poses = np.tile(np.eye(4), (T, 1, 1)).astype(np.float32)
    poses[:, :3, :3] = Rf; poses[:, :3, 3] = tf
    np.savez(a.out, poses=poses, visible=visible)
    print(f"[choir_obj] wrote {a.out} ({len(vis_idx)} frames)")


if __name__ == "__main__":
    main()
