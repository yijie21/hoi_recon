"""Evaluate render_and_compare runs against HOI4D GT depth — the step-1
substrate scoreboard (mono MoGe vs VGGT-Omega injection vs GT injection).

Per run (reads <run>/stage8_eval/pseudo_gt.npz), per frame t:
  object depth err : median z of posed object verts  −  median GT depth inside
                     the (eroded) object mask                       [cm]
  object size      : 5–95 pct extent norm of posed verts, compared with the
                     same statistic on GT-unprojected object-mask points
                     (median over frames) — matches depth_eval.md's method
  hand depth err   : median z of hand verts − median GT depth in hand mask
  wiggle           : std over frames of the signed depth errors (stability)

Metrics per run: obj_depth_MAE/bias/wiggle, obj_size_err, hand_depth_MAE/
wiggle. Aggregation: median over clips per condition + oracle-gap closure
(moge − vggt) / (moge − gt) per metric, pooled and per clip.

Usage:
  python eval_rc_matrix.py --runs-root ../../render_and_compare/runs/hoi4d_matrix
  python eval_rc_matrix.py --validate  # reproduce depth_eval.md on runs/kettle_gt
"""
import argparse, glob, json, os
import numpy as np
import cv2

CLIPS_ROOT = "/workspace/hoi4d/clips"
RC_ROOT = "/workspace/code/hoi_recon/render_and_compare"
ERODE = 5


def masked_gt_stats(clip, t, region, K):
    g = cv2.imread(os.path.join(clip, "depth", f"{t:06d}.png"), cv2.IMREAD_UNCHANGED)
    if g is None:
        return None
    g = g.astype(np.float32) / 1000.0
    m = cv2.imread(os.path.join(clip, "masks", f"frame_{t:06d}_masks", f"{region}.png"),
                   cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ERODE + 1,) * 2)
    m = (cv2.erode((m > 127).astype(np.uint8), ke) > 0) & (g > 0.25) & (g < 5.0)
    if m.sum() < 200:
        return None
    ys, xs = np.nonzero(m)
    z = g[ys, xs]
    depth_med = float(np.median(z))
    size = None
    if region == "object":
        P = np.stack([(xs - K[0, 2]) / K[0, 0] * z, (ys - K[1, 2]) / K[1, 1] * z, z], 1)
        lo, hi = np.percentile(P, 5, axis=0), np.percentile(P, 95, axis=0)
        size = float(np.linalg.norm(hi - lo))
    return depth_med, size


def extent_size(V):
    lo, hi = np.percentile(V, 5, axis=0), np.percentile(V, 95, axis=0)
    return float(np.linalg.norm(hi - lo))


def eval_run(run_dir, clip):
    pg = os.path.join(run_dir, "stage8_eval", "pseudo_gt.npz")
    if not os.path.exists(pg):
        return None
    z = np.load(pg, allow_pickle=True)
    hand_v, obj_v = z["hand_verts"], z["obj_verts"]
    poses = z["obj_poses"]
    T = len(hand_v)
    K = np.load(os.path.join(clip, "intrin.npy")).astype(np.float64)
    obj_err, hand_err, sizes_rc, sizes_gt = [], [], [], []
    for t in range(T):
        Vh = obj_v @ poses[t][:3, :3].T + poses[t][:3, 3]
        og = masked_gt_stats(clip, t, "object", K)
        hg = masked_gt_stats(clip, t, "hand", K)
        if og is not None:
            obj_err.append(float(np.median(Vh[:, 2])) - og[0])
            sizes_rc.append(extent_size(Vh))
            sizes_gt.append(og[1])
        if hg is not None:
            hand_err.append(float(np.median(hand_v[t][:, 2])) - hg[0])
    if len(obj_err) < 10 or len(hand_err) < 10:
        return None
    obj_err = np.array(obj_err); hand_err = np.array(hand_err)
    return {"T": T, "frames_obj": len(obj_err),
            "obj_depth_MAE_cm": float(np.abs(obj_err).mean() * 100),
            "obj_depth_bias_cm": float(obj_err.mean() * 100),
            "obj_wiggle_cm": float(obj_err.std() * 100),
            "obj_size_rc_m": float(np.median(sizes_rc)),
            "obj_size_gt_m": float(np.median(sizes_gt)),
            "obj_size_err_pct": float((np.median(sizes_rc) / np.median(sizes_gt) - 1) * 100),
            "obj_size_flicker_pct": float(np.std(sizes_rc) / np.mean(sizes_rc) * 100),
            "hand_depth_MAE_cm": float(np.abs(hand_err).mean() * 100),
            "hand_wiggle_cm": float(hand_err.std() * 100)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default=os.path.join(RC_ROOT, "runs", "hoi4d_matrix"))
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    if args.validate:
        r = eval_run(os.path.join(RC_ROOT, "runs", "kettle_gt"),
                     os.path.join(CLIPS_ROOT, "kettle_N15"))
        print(json.dumps(r, indent=1))
        print("\nexpected from depth_eval.md: size_rc ~0.240m, size_gt ~0.240m, "
              "obj depth err ~0 (GT-injected run)")
        return

    conds = ("moge", "vggt", "gt")
    rows = {}
    for run in sorted(glob.glob(os.path.join(args.runs_root, "*__*"))):
        base = os.path.basename(run)
        clip_name, cond = base.rsplit("__", 1)
        r = eval_run(run, os.path.join(CLIPS_ROOT, clip_name))
        if r:
            rows.setdefault(clip_name, {})[cond] = r

    metrics = ("obj_depth_MAE_cm", "obj_wiggle_cm", "obj_size_err_pct",
               "hand_depth_MAE_cm", "hand_wiggle_cm")
    hdr = f"{'clip':22s} {'cond':5s} " + " ".join(f"{m.split('_cm')[0][:14]:>15s}" for m in metrics)
    print(hdr); print("-" * len(hdr))
    for clip_name, by_cond in rows.items():
        for cond in conds:
            if cond not in by_cond:
                continue
            r = by_cond[cond]
            print(f"{clip_name:22s} {cond:5s} " +
                  " ".join(f"{r[m]:15.2f}" for m in metrics))
        print()

    print("=" * len(hdr))
    med = {c: {m: float(np.median([rows[cl][c][m] for cl in rows if c in rows[cl]]))
               for m in metrics} for c in conds}
    for c in conds:
        n = sum(1 for cl in rows if c in rows[cl])
        print(f"median[{c:5s}] (n={n}) " +
              " ".join(f"{med[c][m]:15.2f}" for m in metrics))
    print("\noracle-gap closure (moge -> vggt vs moge -> gt), |value|-based per metric:")
    closure = {}
    for m in metrics:
        gaps_num, gaps_den = [], []
        for cl in rows:
            if all(c in rows[cl] for c in conds):
                a, v, g = (abs(rows[cl]["moge"][m]), abs(rows[cl]["vggt"][m]),
                           abs(rows[cl]["gt"][m]))
                gaps_num.append(a - v); gaps_den.append(a - g)
        if sum(gaps_den) > 0:
            closure[m] = sum(gaps_num) / sum(gaps_den)
            print(f"  {m:22s}: {closure[m]:6.2f}   (pooled over "
                  f"{len(gaps_num)} clips with all 3 conds)")
    out = {"per_clip": rows, "medians": med, "closure": closure}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "rc_matrix_results.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote rc_matrix_results.json")


if __name__ == "__main__":
    main()
