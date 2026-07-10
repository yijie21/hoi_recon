"""Three-panel 'current best strategy' overlay on the rectified HOT3D clip:
GT eval GLB @ mocap GT pose (green) | icpjgr estimate (orange, rotation-robust
BEST_ARM) | any6dp estimate (cyan, learned placement-optimal core). Splatted over
the pinhole-rectified frames the pipeline saw. Shows the placement-vs-rotation
Pareto trade-off directly: any6dp usually seats the object more tightly, icpjgr
usually orients symmetric objects more correctly.

Usage: make_best_overlay.py <cat> <num>  [out.mp4]     # e.g. mug_white 001970
2D overlay caveat: it shows silhouette agreement, not the full 3D verdict.
"""
import os
import sys

import cv2
import numpy as np
import trimesh

DS = "/workspace/datasets/hot3d"
RC = "/workspace/code/hoi_recon/render_and_compare"
B, N = 2, 200000
FONT = cv2.FONT_HERSHEY_SIMPLEX


def sample(mesh):
    P, fidx = trimesh.sample.sample_surface(mesh, N, seed=0)
    return np.asarray(P), np.asarray(mesh.face_normals)[fidx]


def load_run(run):
    z = np.load(f"{run}/stage8_eval/pseudo_gt.npz")
    P, Nrm = sample(trimesh.Trimesh(z["obj_verts"], z["obj_faces"], process=False))
    return P, Nrm, z["obj_poses"]


def main():
    cat, num = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else \
        f"overlays/best/best3_{cat}_{num}.mp4"
    inp = f"{DS}/rc_input_{num}_{cat}"
    icp_run = f"{RC}/runs/hot3d_{cat}_{num}_icpjgr"
    a6_run = f"{RC}/runs/hot3d_{cat}_{num}_any6dp"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    gt = np.load(f"{inp}/gt_target.npz")
    K, poses_gt, uid = gt["K"], gt["poses"], int(gt["uid"])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    g = trimesh.load(f"{DS}/object_models_eval/obj_{uid:06d}.glb")
    cad = (trimesh.util.concatenate(list(g.geometry.values()))
           if isinstance(g, trimesh.Scene) else g)
    Pc, Ncn = sample(cad)
    Pi, Ni, poses_i = load_run(icp_run)
    Pa, Na, poses_a = load_run(a6_run)

    frames_dir = f"{icp_run}/stage0_preprocess/frames"
    H, W = cv2.imread(f"{frames_dir}/00000.jpg").shape[:2]
    Wd, Hd = W // B, H // B

    def overlay(img, X, Nrm, base):
        z = X[:, 2]
        ok = z > 0.05
        u = np.clip((X[ok, 0] / z[ok] * fx + cx) / B, 0, Wd - 1).astype(int)
        v = np.clip((X[ok, 1] / z[ok] * fy + cy) / B, 0, Hd - 1).astype(int)
        lam = 0.35 + 0.65 * np.clip(-(Nrm[ok] @ np.array([0.3, -0.5, -0.81])), 0, 1)
        col = (np.asarray(base)[None] * lam[:, None]).astype(np.uint8)
        order = np.argsort(z[ok])[::-1]
        lay = np.zeros((Hd, Wd, 3), np.uint8)
        cov = np.zeros((Hd, Wd), bool)
        lay[v[order], u[order]] = col[order]
        cov[v, u] = True
        cov = cv2.morphologyEx(cov.astype(np.uint8), cv2.MORPH_CLOSE,
                               np.ones((3, 3), np.uint8)) > 0
        hole = cov & (lay.sum(2) == 0)
        lay[hole] = cv2.blur(lay, (3, 3))[hole]
        sm = cv2.resize(img, (Wd, Hd))
        o = sm.copy()
        o[cov] = (0.42 * sm[cov] + 0.58 * lay[cov]).astype(np.uint8)
        return o

    T = min(len(poses_gt), len(poses_i), len(poses_a))
    raw = out + ".raw.mp4"
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), 30, (Wd * 3, Hd))
    panels = [(Pc, Ncn, poses_gt, (80, 220, 80), "GT (mocap)"),
              (Pi, Ni, poses_i, (80, 160, 255), "icpjgr (rotation-robust)"),
              (Pa, Na, poses_a, (255, 170, 80), "any6dp (placement-optimal)")]
    for t in range(T):
        img = cv2.imread(f"{frames_dir}/{t:05d}.jpg")
        cells = []
        for P, Nrm, poses, base, txt in panels:
            M = poses[t]
            im = overlay(img, P @ M[:3, :3].T + M[:3, 3], Nrm @ M[:3, :3].T, base)
            cv2.putText(im, txt, (10, 26), FONT, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(im, txt, (10, 26), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(im, f"frame {t}", (10, Hd - 12), FONT, 0.55,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cells.append(im)
        vw.write(np.hstack(cells))
    vw.release()
    os.system(f"ffmpeg -y -loglevel error -i {raw} -c:v libx264 -crf 20 "
              f"-pix_fmt yuv420p {out} && rm {raw}")
    print("wrote", out)


if __name__ == "__main__":
    main()
