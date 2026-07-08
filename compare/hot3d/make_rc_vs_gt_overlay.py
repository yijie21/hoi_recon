"""Side-by-side overlay on the rectified HOT3D clip: GT eval GLB @ mocap GT
pose (green, left) vs the render_and_compare pipeline estimate (orange,
right), splatted over the pinhole-rectified frames the pipeline actually saw.

Same caveat as the HOI4D version: a 2D overlay hides depth/attitude error —
it shows silhouette agreement, not the 3D verdict.

Usage: make_rc_vs_gt_overlay.py <rc_input_dir> <run_dir> [out.mp4]
"""
import json
import os
import sys

import cv2
import numpy as np
import trimesh

DS = "/workspace/datasets/hot3d"
B = 2
N = 200000
FONT = cv2.FONT_HERSHEY_SIMPLEX


def sample(mesh):
    P, fidx = trimesh.sample.sample_surface(mesh, N, seed=0)
    return np.asarray(P), np.asarray(mesh.face_normals)[fidx]


def main():
    inp, run = sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/")
    out = sys.argv[3] if len(sys.argv) > 3 else \
        f"rc_vs_gt_{os.path.basename(run)}.mp4"
    gt = np.load(f"{inp}/gt_target.npz")
    K, poses_gt, uid = gt["K"], gt["poses"], int(gt["uid"])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    g = trimesh.load(f"{DS}/object_models_eval/obj_{uid:06d}.glb")
    cad = (trimesh.util.concatenate(list(g.geometry.values()))
           if isinstance(g, trimesh.Scene) else g)
    Pc, Ncn = sample(cad)

    z8 = np.load(f"{run}/stage8_eval/pseudo_gt.npz")
    est = trimesh.Trimesh(z8["obj_verts"], z8["obj_faces"], process=False)
    Pe, Ne = sample(est)
    poses_est = z8["obj_poses"]

    frames_dir = f"{run}/stage0_preprocess/frames"
    img0 = cv2.imread(f"{frames_dir}/00000.jpg")
    H, W = img0.shape[:2]
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

    T = min(len(poses_gt), len(poses_est))
    raw = out + ".raw.mp4"
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), 30, (Wd * 2, Hd))
    for t in range(T):
        img = cv2.imread(f"{frames_dir}/{t:05d}.jpg")
        Mg, Me = poses_gt[t], poses_est[t]
        a = overlay(img, Pc @ Mg[:3, :3].T + Mg[:3, 3], Ncn @ Mg[:3, :3].T,
                    (80, 220, 80))
        b = overlay(img, Pe @ Me[:3, :3].T + Me[:3, 3], Ne @ Me[:3, :3].T,
                    (80, 160, 255))
        for im, txt in ((a, "GT model + mocap GT pose (HOT3D)"),
                        (b, f"pipeline estimate ({os.path.basename(run)})")):
            cv2.putText(im, txt, (10, 26), FONT, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(im, txt, (10, 26), FONT, 0.6, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(im, f"frame {t}", (10, Hd - 12), FONT, 0.55,
                        (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(np.hstack([a, b]))
    vw.release()
    os.system(f"ffmpeg -y -loglevel error -i {raw} -c:v libx264 -crf 20 "
              f"-pix_fmt yuv420p {out} && rm {raw}")
    print("wrote", out)


if __name__ == "__main__":
    main()
