"""Aggregate kill-test results across clips -> verdict table, figure, RESULTS.md."""
import json, glob, os
import numpy as np

CLIPS_DIR = "/workspace/hoi4d/clips"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
VARIANTS = [("p", "per-frame bg (pre-registered)"), ("pn", "near-field bg"),
            ("pd", "bg disparity-space"), ("pnd", "near+disparity"),
            ("o", "oracle fg (upper bound)")]

manifest = json.load(open(f"{CLIPS_DIR}/clips_manifest.json"))
rows = []
for name in sorted(manifest):
    p = f"{CLIPS_DIR}/{name}/kill_test/result.json"
    if not os.path.exists(p):
        print(f"MISSING result: {name}")
        continue
    r = json.load(open(p))["summary"]
    rows.append({"clip": name, "category": manifest[name]["category"].lower(), **r})

print(f"{len(rows)} clips with results\n")
hdr = f"{'clip':22s} {'MAEg':>5s} " + " ".join(f"R_{t:<4s}" for t, _ in VARIANTS) + \
      "  corr_a  wig_g  wig_o"
print(hdr)
for r in rows:
    v = r["variants"]
    cells = " ".join(f"{v.get(t, {}).get('R_MAE', float('nan')) * 100:5.1f}%" for t, _ in VARIANTS)
    print(f"{r['clip']:22s} {r['fg_MAE_global_cm']:4.1f}c {cells}  "
          f"{(r.get('gauge_transfer_corr_a') or 0):5.2f}  "
          f"{r['wiggle_global_cm']:4.1f}c  {v.get('o', {}).get('wiggle_cm', float('nan')):4.2f}c")

agg = {}
for t, label in VARIANTS:
    Rs = np.array([r["variants"][t]["R_MAE"] for r in rows if t in r["variants"]])
    agg[t] = {"label": label, "mean_R": float(Rs.mean()), "median_R": float(np.median(Rs)),
              "clips_ge_50pct": int((Rs >= 0.5).sum()), "n": len(Rs),
              "Rs": Rs.tolist()}
corrs = np.array([r.get("gauge_transfer_corr_a") or np.nan for r in rows], float)
wig_g = np.array([r["wiggle_global_cm"] for r in rows])
wig_o = np.array([r["variants"]["o"]["wiggle_cm"] for r in rows if "o" in r["variants"]])

print("\n=== AGGREGATE ===")
for t, label in VARIANTS:
    a = agg[t]
    print(f"{label:32s} mean R={a['mean_R'] * 100:5.1f}%  median={a['median_R'] * 100:5.1f}%  "
          f"clips>=50%: {a['clips_ge_50pct']}/{a['n']}")
print(f"gauge transfer corr(a_bg, a_fg):  mean {np.nanmean(corrs):.2f}  median {np.nanmedian(corrs):.2f}")
print(f"fg temporal wiggle: global {wig_g.mean():.1f}cm -> oracle-fg-gauge {wig_o.mean():.2f}cm "
      f"({wig_g.mean() / max(wig_o.mean(), 1e-9):.1f}x reduction available IN fg gauge)")

crit = agg["p"]
verdict = "GO" if crit["median_R"] >= 0.5 else "KILL"
print(f"\nPRE-REGISTERED CRITERION: per-frame bg gauge cuts fg depth error >=50% across >=10 clips")
print(f"VERDICT: {verdict}  (median R = {crit['median_R'] * 100:.1f}%, "
      f"{crit['clips_ge_50pct']}/{crit['n']} clips pass)")

with open(f"{OUT_DIR}/aggregate.json", "w") as f:
    json.dump({"rows": rows, "aggregate": {t: {k: v for k, v in a.items() if k != "Rs"}
                                           for t, a in agg.items()},
               "verdict": verdict}, f, indent=1)

# ---------------- figure: R_MAE per clip per variant (dot plot) ---------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    COLS = {"p": "#2a78d6", "pn": "#1baf7a", "pd": "#eda100", "o": "#4a3aa7"}
    names = [r["clip"] for r in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(rows) + 2.4))
    for t in ("p", "pn", "pd", "o"):
        xs = [r["variants"].get(t, {}).get("R_MAE", np.nan) * 100 for r in rows]
        ax.scatter(xs, y, s=52, color=COLS[t], label=dict(VARIANTS)[t], zorder=3)
    ax.axvline(50, color="#d03b3b", lw=1.4, ls="--", zorder=2)
    ax.text(50.8, -0.55, "GO threshold (50%)", color="#d03b3b", fontsize=8.5)
    ax.axvline(0, color="#c3c2b7", lw=1)
    ax.set_yticks(y, names, fontsize=9)
    ax.set_xlabel("foreground depth-error reduction vs clip-global gauge (%)")
    ax.set_title("b6 kill test: does a background-fitted per-frame gauge fix hand/object depth?")
    ax.grid(axis="x", color="#e1e0d9", lw=0.7, zorder=0)
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.95)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/killtest_summary.png", dpi=140)
    print(f"figure -> {OUT_DIR}/killtest_summary.png")
except Exception as e:
    print("figure failed:", e)
