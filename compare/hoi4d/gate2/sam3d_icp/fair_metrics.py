"""Metric-artifact check for sam3d_icp_test: the eval_rc_matrix depth error
compares median z of ALL mesh verts vs median GT depth of the VISIBLE front
surface — a geometrically perfect fit of a ~20 cm object should read ~+4 cm,
not 0. Score every trajectory with two honest metrics instead:

  fit_mm    per-frame median distance from GT object cloud points to the
            posed mesh surface (approx: NN against 20k surface samples)
  vis_cm    visibility-aware depth error: z-buffer the posed samples into
            pixels, median over mask pixels of (mesh front z - GT z)

Also: correlation of per-frame error with visible mask area (occlusion).
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
SEED = 0
N_SRC = 20000
BIN = 4  # z-buffer pixel bin (px) at 1920x1080


def main():
    K = np.load(f"{RUN}/stage0_preprocess/arrays.npz")["intrinsics"]
    pg = np.load(f"{RUN}/stage8_eval/pseudo_gt.npz")
    verts, faces = pg["obj_verts"], pg["obj_faces"]
    tr = np.load(os.path.join(OUT, "trajectories.npz"))
    trajs = {k: tr[k] for k in ("rc", "rigid", "sim")}
    T = len(trajs["rc"])

    mesh = trimesh.Trimesh(verts, faces, process=False)
    src, _ = trimesh.sample.sample_surface(mesh, N_SRC, seed=SEED)
    src = np.asarray(src)
    rng = np.random.default_rng(SEED)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ERODE + 1,) * 2)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    out = {k: {"fit_mm": [], "vis_cm": []} for k in trajs}
    areas = []
    for tt in range(T):
        g = np.load(f"{RUN}/stage0_preprocess/depth/{tt:05d}.npy").astype(np.float32)
        m = (cv2.erode(np.load(f"{RUN}/stage1_detect_track/masks/{tt:05d}.npy")
                       .astype(np.uint8), ker) > 0) & (g > 0.25) & (g < 5.0)
        areas.append(int(m.sum()))
        ys, xs = np.nonzero(m)
        z = g[ys, xs]
        P = np.stack([(xs - cx) / fx * z, (ys - cy) / fy * z, z], 1)
        if len(P) > 4000:
            sel = rng.choice(len(P), 4000, replace=False)
            P, ys, xs, z = P[sel], ys[sel], xs[sel], z[sel]
        pix = (ys // BIN) * 10000 + (xs // BIN)

        for name, Ptraj in trajs.items():
            M = Ptraj[tt]
            V = src @ M[:3, :3].T + M[:3, 3]
            out[name]["fit_mm"].append(
                float(np.median(cKDTree(V).query(P, workers=-1)[0]) * 1000))
            # z-buffer posed samples into binned pixels -> front-surface z
            u = np.clip((V[:, 0] / V[:, 2] * fx + cx).astype(int), 0, 1919)
            v = np.clip((V[:, 1] / V[:, 2] * fy + cy).astype(int), 0, 1079)
            key = (v // BIN) * 10000 + (u // BIN)
            order = np.argsort(V[:, 2])
            key_s = key[order]
            first = np.unique(key_s, return_index=True)[1]
            front = dict(zip(key_s[first], V[:, 2][order][first]))
            dz = [front[p] - zz for p, zz in zip(pix, z) if p in front]
            out[name]["vis_cm"].append(
                float(np.median(dz) * 100) if len(dz) > 50 else np.nan)

    areas = np.array(areas, float)
    res = {}
    for name, d in out.items():
        fit = np.array(d["fit_mm"])
        vis = np.array(d["vis_cm"])
        ok = ~np.isnan(vis)
        cor = float(np.corrcoef(areas[ok], vis[ok])[0, 1])
        res[name] = {"fit_mm_med": float(np.median(fit)),
                     "fit_mm_p90": float(np.percentile(fit, 90)),
                     "vis_MAE_cm": float(np.abs(vis[ok]).mean()),
                     "vis_bias_cm": float(vis[ok].mean()),
                     "vis_wiggle_cm": float(vis[ok].std()),
                     "corr_err_vs_area": cor,
                     "fit_mm": np.round(fit, 2).tolist(),
                     "vis_cm": np.round(vis, 2).tolist()}
        print(f"{name:6s} fit_med {res[name]['fit_mm_med']:5.1f} mm (p90 "
              f"{res[name]['fit_mm_p90']:5.1f})  vis: MAE "
              f"{res[name]['vis_MAE_cm']:5.2f}  bias {res[name]['vis_bias_cm']:6.2f}  "
              f"wiggle {res[name]['vis_wiggle_cm']:5.2f} cm  corr(area) {cor:+.2f}")

    with open(os.path.join(OUT, "fair_metrics.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
