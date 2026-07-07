"""End-to-end A/B: RC stages 4-8 with vs without the stage4 object-ICP flag,
scored with the honest visibility-aware metrics of fair_metrics.py against the
run's own GT depth. Each run uses ITS OWN canonical mesh (base/icp share the
archived flat mesh; base2/icp2 use the sam3d5090-regenerated kettle).

Usage: eval_pipeline_ab.py [run_suffix ...]   (default: base icp)
"""
import json
import os
import sys

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree

RC = "/workspace/code/hoi_recon/render_and_compare/runs"
GT = f"{RC}/kettle_gt"          # depth + masks source (identical across runs)
OUT = os.path.dirname(os.path.abspath(__file__))
ERODE, SEED, N_SRC, BIN = 5, 0, 20000, 4


def main():
    names = sys.argv[1:] or ["base", "icp"]
    K = np.load(f"{GT}/stage0_preprocess/arrays.npz")["intrinsics"]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    runs, srcs = {}, {}
    for name in names:
        z = np.load(f"{RC}/kettle_gt_{name}/stage8_eval/pseudo_gt.npz")
        runs[name] = z
        mesh = trimesh.Trimesh(z["obj_verts"], z["obj_faces"], process=False)
        srcs[name] = np.asarray(
            trimesh.sample.sample_surface(mesh, N_SRC, seed=SEED)[0])
    T = len(runs[names[0]]["obj_poses"])

    rng = np.random.default_rng(SEED)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ERODE + 1,) * 2)

    out = {n: {"fit_mm": [], "vis_cm": []} for n in runs}
    for tt in range(T):
        g = np.load(f"{GT}/stage0_preprocess/depth/{tt:05d}.npy").astype(np.float32)
        m = (cv2.erode(np.load(f"{GT}/stage1_detect_track/masks/{tt:05d}.npy")
                       .astype(np.uint8), ker) > 0) & (g > 0.25) & (g < 5.0)
        ys, xs = np.nonzero(m)
        z = g[ys, xs]
        P = np.stack([(xs - cx) / fx * z, (ys - cy) / fy * z, z], 1)
        if len(P) > 4000:
            sel = rng.choice(len(P), 4000, replace=False)
            P, ys, xs, z = P[sel], ys[sel], xs[sel], z[sel]
        pix = (ys // BIN) * 10000 + (xs // BIN)
        for n, zz in runs.items():
            M = zz["obj_poses"][tt]
            V = srcs[n] @ M[:3, :3].T + M[:3, 3]
            out[n]["fit_mm"].append(
                float(np.median(cKDTree(V).query(P, workers=-1)[0]) * 1000))
            u = np.clip((V[:, 0] / V[:, 2] * fx + cx).astype(int), 0, 1919)
            v = np.clip((V[:, 1] / V[:, 2] * fy + cy).astype(int), 0, 1079)
            key = (v // BIN) * 10000 + (u // BIN)
            order = np.argsort(V[:, 2])
            ks = key[order]
            first = np.unique(ks, return_index=True)[1]
            front = dict(zip(ks[first], V[:, 2][order][first]))
            dz = [front[p] - d for p, d in zip(pix, z) if p in front]
            out[n]["vis_cm"].append(
                float(np.median(dz) * 100) if len(dz) > 50 else np.nan)

    res = {}
    for n, d in out.items():
        fit, vis = np.array(d["fit_mm"]), np.array(d["vis_cm"])
        ok = ~np.isnan(vis)
        # hand: median-z error vs GT is not recomputable without hand masks in
        # the run dir; report hand jitter (accel) instead
        hv = runs[n]["hand_verts"]
        hacc = float(np.sqrt((np.diff(hv.mean(1), 2, axis=0) ** 2)
                             .sum(1).mean()) * 1000)
        res[n] = {"fit_mm_med": float(np.median(fit)),
                  "fit_mm_p90": float(np.percentile(fit, 90)),
                  "vis_MAE_cm": float(np.abs(vis[ok]).mean()),
                  "vis_bias_cm": float(vis[ok].mean()),
                  "vis_wiggle_cm": float(vis[ok].std()),
                  "hand_centroid_accel_mm": hacc,
                  "fit_mm": np.round(fit, 2).tolist(),
                  "vis_cm": np.round(vis, 2).tolist()}
        print(f"{n:5s} fit {res[n]['fit_mm_med']:5.1f}/{res[n]['fit_mm_p90']:5.1f} mm"
              f"  vis MAE {res[n]['vis_MAE_cm']:5.2f}  bias {res[n]['vis_bias_cm']:6.2f}"
              f"  wiggle {res[n]['vis_wiggle_cm']:5.2f} cm  hand_accel {hacc:5.1f} mm")
    tag = "_".join(names)
    with open(os.path.join(OUT, f"pipeline_ab_{tag}.json" if names != ["base", "icp"]
                           else "pipeline_ab.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
