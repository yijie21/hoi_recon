"""Image-space diagnostic the depth metrics can't see: projected mesh
footprint vs object mask (IoU, area ratio, centroid offset) per arm.

Motivation: rc_ab3_*.mp4 shows the final arm's mesh visually hanging off the
kettle while its 3D fit is 3.9 mm — the top-down partial dome view leaves
in-plane translation loosely constrained (sliding basin), the masked depth
metrics only score covered mask pixels, and mesh overhang outside the mask
is invisible to them. This script quantifies that overhang.

Usage: silhouette_check.py [run_suffix ...]   (default: base icp icp2 icp4)
"""
import sys

import cv2
import numpy as np
import trimesh

RC = "/workspace/code/hoi_recon/render_and_compare/runs"
B = 8            # pixel bin
N_SRC = 30000


def main():
    names = sys.argv[1:] or ["base", "icp", "icp2", "icp4"]
    K = np.load(f"{RC}/kettle_gt/stage0_preprocess/arrays.npz")["intrinsics"]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    for name in names:
        z = np.load(f"{RC}/kettle_gt_{name}/stage8_eval/pseudo_gt.npz")
        mesh = trimesh.Trimesh(z["obj_verts"], z["obj_faces"], process=False)
        src = np.asarray(trimesh.sample.sample_surface(mesh, N_SRC, seed=0)[0])
        T = len(z["obj_poses"])
        iou, ar, cd = [], [], []
        for t in range(0, T, 3):
            m = np.load(f"{RC}/kettle_gt/stage1_detect_track/masks/{t:05d}.npy")
            gm = cv2.resize(m.astype(np.uint8), (1920 // B, 1080 // B),
                            interpolation=cv2.INTER_AREA) > 0.5
            M = z["obj_poses"][t]
            V = src @ M[:3, :3].T + M[:3, 3]
            u = np.clip((V[:, 0] / V[:, 2] * fx + cx) / B, 0, 1920 // B - 1).astype(int)
            v = np.clip((V[:, 1] / V[:, 2] * fy + cy) / B, 0, 1080 // B - 1).astype(int)
            pm = np.zeros_like(gm)
            pm[v, u] = True
            pm = cv2.dilate(pm.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            iou.append((pm & gm).sum() / max((pm | gm).sum(), 1))
            ar.append(pm.sum() / max(gm.sum(), 1))
            py, px = np.nonzero(pm)
            gy, gx = np.nonzero(gm)
            cd.append(np.hypot(px.mean() - gx.mean(), py.mean() - gy.mean()) * B)
        print(f"{name:6s} IoU med {np.median(iou):.3f} (p10 {np.percentile(iou, 10):.3f})"
              f"  area ratio med {np.median(ar):.2f}"
              f"  centroid offset med {np.median(cd):.0f}px")


if __name__ == "__main__":
    main()
