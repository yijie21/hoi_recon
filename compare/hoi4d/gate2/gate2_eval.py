"""Gate-2 evaluation — do modern feed-forward 4D models already solve the
per-frame foreground depth gauge on HOI clips?

Inputs: <clip>/gate2/<source>_samples.npz from gate2_extract.py (same frames,
same pixel populations for every source).

Per source, per clip (machinery identical to the b6/b5 kill tests):
  global gauge : ONE Huber-affine (a,b) fitted on pooled background samples
                 (GT ~ a*pred + b) — normalises away each source's overall
                 scale convention, exactly like the kill tests did for MoGe.
  obj/hand/union metrics after the global gauge:
                 MAE_cm, wiggle_cm = std over frames of the per-frame mean
                 signed error (the gauge-instability readout).
  ko oracle    : per-frame 1-DOF scale fitted on the object's own pixels vs
                 GT -> R_ko = MAE reduction it still achieves. If a source has
                 already solved the per-frame gauge, R_ko ~ 0 and wiggle is
                 small; large R_ko = per-frame gauge error still present.
  raw metric   : MAE with NO gauge fit (is the output actually metric?).
  kappa corr   : corr of the source's per-frame ko-kappa trace with MoGe's —
                 do the new models wobble in the same mode?

PRE-REGISTERED CRITERION (fixed before looking at any model output; MoGe
reference numbers from the b6/b5 runs: wiggle ~3-5 cm, oracle residual
~0.4 cm). A source SOLVES the foreground gauge iff, median over the 12 clips:
  (i)   wiggle_obj (global-gauged) <= 1.0 cm, AND
  (ii)  R_ko <= 0.15 (little per-frame scale left to fix), AND
  (iii) obj MAE (global-gauged) <= MoGe's obj MAE on the same samples.
PARTIAL if it at least halves MoGe's median wiggle with MAE <= MoGe's:
  "commodity substrate materially improves the gauge" -> update the program.
Else FAIL: the per-frame foreground gauge wobble persists in this model
generation and the b5 photometric direction stands (with this SOTA control).

Usage: python gate2_eval.py [--sources moge,d4rt,vggt]
"""
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kill_test"))
from kill_test import huber_affine

CLIPS_ROOT = "/workspace/hoi4d/clips"
MIN_PX = 200


def per_frame_stats(t, zp, zg, ag, bg):
    """Per-frame MAE/bias after global gauge + per-frame ko scale fit."""
    zgl = ag * zp + bg
    frames = np.unique(t)
    rows = []
    for f in frames:
        m = t == f
        if m.sum() < MIN_PX:
            continue
        zf, gf = zgl[m], zg[m]
        res = zf - gf
        ko = float(np.median(gf / np.maximum(zf, 1e-6)))
        rko = ko * zf - gf
        rows.append((int(f), float(np.mean(np.abs(res))), float(np.mean(res)),
                     float(np.mean(np.abs(rko))), ko, float(np.mean(rko))))
    return rows


def eval_source(clip, source):
    p = os.path.join(clip, "gate2", f"{source}_samples.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    ok_pred = {}
    for r in ("obj", "hand", "bg"):
        zp = z[f"{r}_zp"]
        ok = np.isfinite(zp) & (zp > 0)
        ok_pred[r] = {k: z[f"{r}_{k}"][ok] for k in ("t", "y", "x", "zg")} | {"zp": zp[ok]}
    g = huber_affine(ok_pred["bg"]["zp"], ok_pred["bg"]["zg"])
    if g is None:
        return None
    ag, bg = g
    out = {"global_gauge": {"a": ag, "b": bg},
           "valid_frac": {r: float((np.isfinite(z[f"{r}_zp"]) & (z[f"{r}_zp"] > 0)).mean())
                          for r in ("obj", "hand", "bg")}}
    uni = {k: np.concatenate([ok_pred["obj"][k], ok_pred["hand"][k]])
           for k in ("t", "zg", "zp")}
    for name, d in (("obj", ok_pred["obj"]), ("hand", ok_pred["hand"]), ("union", uni)):
        rows = per_frame_stats(d["t"], d["zp"], d["zg"], ag, bg)
        if not rows:
            continue
        mae = np.array([r[1] for r in rows]); bias = np.array([r[2] for r in rows])
        mko = np.array([r[3] for r in rows]); bko = np.array([r[5] for r in rows])
        raw = np.abs(d["zp"] - d["zg"])
        out[name] = {"MAE_cm": float(mae.mean() * 100),
                     "wiggle_cm": float(bias.std() * 100),
                     "MAE_ko_cm": float(mko.mean() * 100),
                     "R_ko": float(1 - mko.sum() / mae.sum()),
                     "wiggle_ko_cm": float(bko.std() * 100),
                     "MAE_raw_cm": float(raw.mean() * 100),
                     "frames": len(rows),
                     "ko_trace": {int(r[0]): float(r[4]) for r in rows}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="moge,d4rt,vggt")
    args = ap.parse_args()
    sources = args.sources.split(",")
    clips = sorted(d for d in glob.glob(os.path.join(CLIPS_ROOT, "*")) if os.path.isdir(d))

    results = {}
    for clip in clips:
        name = os.path.basename(clip)
        row = {}
        for s in sources:
            r = eval_source(clip, s)
            if r:
                row[s] = r
        if row:
            results[name] = row

    # kappa-trace correlations vs moge
    for name, row in results.items():
        if "moge" not in row:
            continue
        base = row["moge"].get("obj", {}).get("ko_trace", {})
        for s in sources:
            if s == "moge" or s not in row or "obj" not in row[s]:
                continue
            tr = row[s]["obj"]["ko_trace"]
            common = sorted(set(base) & set(tr))
            if len(common) > 3:
                a = np.array([base[f] for f in common]); b = np.array([tr[f] for f in common])
                row[s]["obj"]["corr_ko_vs_moge"] = float(np.corrcoef(a, b)[0, 1])

    print(f"{'clip':22s} {'src':5s} {'MAE':>6s} {'wig':>6s} {'R_ko':>6s} {'MAEko':>6s} "
          f"{'raw':>7s} {'corr_m':>6s} {'val%':>5s}")
    print("-" * 78)
    for name, row in results.items():
        for s in sources:
            if s not in row or "obj" not in row[s]:
                continue
            o = row[s]["obj"]
            corr = o.get("corr_ko_vs_moge")
            print(f"{name:22s} {s:5s} {o['MAE_cm']:6.2f} {o['wiggle_cm']:6.2f} "
                  f"{o['R_ko']:6.3f} {o['MAE_ko_cm']:6.2f} {o['MAE_raw_cm']:7.1f} "
                  f"{(corr if corr is not None else float('nan')):6.2f} "
                  f"{row[s]['valid_frac']['obj'] * 100:5.1f}")
        print()

    # pre-registered verdict per source
    print("=" * 78)
    verdicts = {}
    med = {}
    for s in sources:
        rows = [results[c][s]["obj"] for c in results if s in results[c] and "obj" in results[c][s]]
        if not rows:
            continue
        med[s] = {"MAE": float(np.median([r["MAE_cm"] for r in rows])),
                  "wig": float(np.median([r["wiggle_cm"] for r in rows])),
                  "R_ko": float(np.median([r["R_ko"] for r in rows])),
                  "raw": float(np.median([r["MAE_raw_cm"] for r in rows])),
                  "n": len(rows)}
    for s in sources:
        if s == "moge" or s not in med:
            continue
        m, ref = med[s], med.get("moge", {})
        solved = m["wig"] <= 1.0 and m["R_ko"] <= 0.15 and m["MAE"] <= ref.get("MAE", 1e9)
        partial = (not solved) and m["wig"] <= 0.5 * ref.get("wig", 0) and m["MAE"] <= ref.get("MAE", 1e9)
        verdicts[s] = "SOLVES the gauge" if solved else ("PARTIAL improvement" if partial else "FAIL — gauge persists")
    for s, m in med.items():
        print(f"median[{s:5s}] MAE={m['MAE']:5.2f}cm wiggle={m['wig']:5.2f}cm "
              f"R_ko={m['R_ko']:5.3f} raw={m['raw']:6.1f}cm (n={m['n']})"
              + (f"  ->  {verdicts[s]}" if s in verdicts else "  (baseline)"))

    out = {"per_clip": results, "medians": med, "verdicts": verdicts,
           "criterion": "solve: wiggle<=1.0cm & R_ko<=0.15 & MAE<=MoGe; "
                        "partial: wiggle<=0.5*MoGe & MAE<=MoGe"}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gate2_results.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote gate2_results.json")


if __name__ == "__main__":
    main()
