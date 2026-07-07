"""Semantic-extraction + mesh-registration test on GT depth (kettle_N15).

Question (user's idea, GT-depth regime): with the object fully localized in 3D
(semantic masks x GT depth -> per-frame object point cloud) and a canonical
SAM-3D object mesh, does per-frame registration produce a stable, accurate
object trajectory — i.e. is "align the extracted object to the point cloud"
sufficient for HOI alignment once depth is trustworthy?

Data: the archived runs/kettle_gt RC run (the raw HOI4D clip tree is gone):
  stage0_preprocess/depth/*.npy   GT-injected depth (float16, meters)
  stage0_preprocess/arrays.npz    intrinsics
  stage1_detect_track/masks/*.npy SAM2 object masks (IoU 0.98 vs kill-test ref)
  stage3_object/arrays.npz        SAM-3D canonical mesh (metric, radius 0.10 m)
  stage8_eval/pseudo_gt.npz       full-pipeline final poses (baseline)

Method: per frame, backproject GT depth inside the 5px-eroded object mask
(subsampled to 3000 pts, seed 0); trimmed ICP (keep best 80% target->mesh NN,
Umeyama solve, <=60 iters) of 20k mesh surface samples. Frame 0 initialized
from the pipeline pose; frame t from the t-1 result. Variants:
  rigid  scale locked at 1 (canonical mesh is already metric)
  sim    per-frame free scale — diagnostic of scale observability under
         occlusion (the pivot test's rig-o failure mode)

Metrics (defs identical to eval_rc_matrix.py): per-frame err_t = median z of
posed verts - median GT depth in eroded mask; MAE/bias/wiggle(std). Plus
GT-free smoothness: centroid acceleration RMS, rotation geodesic speed, and
per-frame trimmed ICP residual. Outputs: results.json, trajectories.npz,
curves.png.
"""
import json
import os

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree

RUN = "/workspace/code/hoi_recon/render_and_compare/runs/kettle_gt"
OUT = os.path.dirname(os.path.abspath(__file__))
ERODE = 5
N_TGT = 3000
N_SRC = 20000
TRIM = 0.8
SEED = 0


def umeyama(src, dst, with_scale):
    """Similarity transform (s, R, t) minimizing ||s*R@src + t - dst||."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    A, B = src - mu_s, dst - mu_d
    cov = B.T @ A / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = (D * S.diagonal()).sum() / (A ** 2).sum() * len(src) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def icp(src_pts, src_tree, tgt, s0, R0, t0, with_scale, iters=60):
    """Trimmed ICP: canonical mesh samples (with prebuilt KDTree) -> target
    cloud. Correspondences target->mesh (target is a partial view)."""
    s, R, t = s0, R0.copy(), t0.copy()
    prev = np.inf
    for _ in range(iters):
        # pull targets into canonical frame, match against static tree
        tgt_c = (tgt - t) @ R / s
        d, j = src_tree.query(tgt_c, workers=-1)
        keep = d <= np.quantile(d, TRIM)
        s, R, t = umeyama(src_pts[j[keep]], tgt[keep], with_scale)
        res = float(np.sqrt(np.mean(
            np.sum((s * src_pts[j[keep]] @ R.T + t - tgt[keep]) ** 2, 1))))
        if abs(prev - res) < 1e-6:
            break
        prev = res
    return s, R, t, res


def main():
    K = np.load(f"{RUN}/stage0_preprocess/arrays.npz")["intrinsics"]
    pg = np.load(f"{RUN}/stage8_eval/pseudo_gt.npz")
    verts, faces, poses_rc = pg["obj_verts"], pg["obj_faces"], pg["obj_poses"]
    T = len(poses_rc)

    mesh = trimesh.Trimesh(verts, faces, process=False)
    src_pts, _ = trimesh.sample.sample_surface(mesh, N_SRC, seed=SEED)
    src_pts = np.asarray(src_pts)
    src_tree = cKDTree(src_pts)
    rng = np.random.default_rng(SEED)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ERODE + 1,) * 2)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    tgts, gt_med = [], []
    for tt in range(T):
        g = np.load(f"{RUN}/stage0_preprocess/depth/{tt:05d}.npy").astype(np.float32)
        m = np.load(f"{RUN}/stage1_detect_track/masks/{tt:05d}.npy")
        m = (cv2.erode(m.astype(np.uint8), ker) > 0) & (g > 0.25) & (g < 5.0)
        ys, xs = np.nonzero(m)
        z = g[ys, xs]
        gt_med.append(float(np.median(z)))
        P = np.stack([(xs - cx) / fx * z, (ys - cy) / fy * z, z], 1)
        if len(P) > N_TGT:
            P = P[rng.choice(len(P), N_TGT, replace=False)]
        tgts.append(P)

    results = {}
    trajs = {"rc": poses_rc}
    for name, with_scale in [("rigid", False), ("sim", True)]:
        s, R, t = 1.0, poses_rc[0][:3, :3].copy(), poses_rc[0][:3, 3].copy()
        P, scales, resid = [], [], []
        for tt in range(T):
            s, R, t, r = icp(src_pts, src_tree, tgts[tt], s, R, t, with_scale)
            M = np.eye(4)
            M[:3, :3], M[:3, 3] = s * R, t
            P.append(M)
            scales.append(s)
            resid.append(r)
        trajs[name] = np.stack(P)
        results[name] = {"scale_mean": float(np.mean(scales)),
                         "scale_cv_pct": float(np.std(scales) / np.mean(scales) * 100),
                         "icp_resid_mm_med": float(np.median(resid) * 1000)}
        results[name + "_scales"] = np.round(scales, 4).tolist()

    # score all trajectories with the eval_rc_matrix depth-error convention
    for name, P in trajs.items():
        err, cen, ang = [], [], [0.0]
        for tt in range(T):
            V = verts @ P[tt][:3, :3].T + P[tt][:3, 3]
            err.append(float(np.median(V[:, 2])) - gt_med[tt])
            cen.append(V.mean(0))
        cen = np.stack(cen)
        for tt in range(1, T):
            A = P[tt - 1][:3, :3] / np.cbrt(np.linalg.det(P[tt - 1][:3, :3]))
            B = P[tt][:3, :3] / np.cbrt(np.linalg.det(P[tt][:3, :3]))
            c = np.clip((np.trace(A.T @ B) - 1) / 2, -1, 1)
            ang.append(float(np.degrees(np.arccos(c))))
        err = np.array(err)
        acc = np.diff(cen, 2, axis=0)
        results.setdefault(name, {}).update({
            "obj_depth_MAE_cm": float(np.abs(err).mean() * 100),
            "obj_depth_bias_cm": float(err.mean() * 100),
            "obj_wiggle_cm": float(err.std() * 100),
            "centroid_accel_rms_mm": float(np.sqrt((acc ** 2).sum(1).mean()) * 1000),
            "rot_speed_deg_med": float(np.median(ang[1:]))})
        results[name + "_err_cm"] = np.round(err * 100, 3).tolist()

    np.savez_compressed(os.path.join(OUT, "trajectories.npz"),
                        gt_med=np.array(gt_med), **trajs)
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    for name in trajs:
        r = results[name]
        print(f"{name:6s} MAE {r['obj_depth_MAE_cm']:6.2f}  bias "
              f"{r['obj_depth_bias_cm']:7.2f}  wiggle {r['obj_wiggle_cm']:5.2f} cm  "
              f"accel {r['centroid_accel_rms_mm']:6.1f} mm  "
              f"rot {r['rot_speed_deg_med']:5.2f} deg/f"
              + (f"  scale_cv {r['scale_cv_pct']:.2f}%  resid "
                 f"{r['icp_resid_mm_med']:.1f} mm" if "scale_cv_pct" in r else ""))


if __name__ == "__main__":
    main()
