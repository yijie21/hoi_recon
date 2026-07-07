"""Aggregate the b5-pivot kill test over all clips and apply the
pre-registered v2 criterion (see pivot_test.py docstring).

Two verdicts, both reported:
  registered : PRIMARY = eb_s on all 12 clips (as frozen before real data).
               The pilot clip revealed eb_s is structurally degenerate — the
               rig anchor is ANTI-correlated with the true gauge under grasp
               occlusion, so EB's disagreement shrinkage collapses to the
               baseline (R = 0). Reported for the record.
  amended    : PRIMARY = flow_s, selected on the declared pilot clip
               (kettle_N22) ONLY; verdict computed on the 11 held-out clips.
               This is the headline verdict: the pilot is the config-selection
               set, the holdout is the test set.

Eligibility for ratio statistics: R_obj(ko) >= 0.10 (clips below the floor
are reported but excluded from headroom ratios — unstable denominator).
"""
import json, glob, os
import numpy as np

CLIPS_ROOT = "/workspace/hoi4d/clips"
PRIMARY = "eb_s"
AMENDED_PRIMARY = "flow_s"
PILOT = "kettle_N22_S157_T1"
KO_FLOOR = 0.10


def evaluate(rows, primary, label):
    per_clip, eligible = [], []
    for r in rows:
        mo, mu = r["metrics"]["obj"], r["metrics"]["union"]
        g, p, ko = mo["global"], mo[primary], mo["ko"]
        e = {
            "clip": r["clip"],
            "MAE_g": g["MAE_cm"], "MAE_prim": p["MAE_cm"], "MAE_ko": ko["MAE_cm"],
            "MAE_smooth": mo["smooth"]["MAE_cm"],
            "R_prim": p["R_MAE"], "R_ko": ko["R_MAE"],
            "R_so": mo["so"]["R_MAE"], "R_oracle": mo["oracle"]["R_MAE"],
            "R_dc0_prim": p["R_dc0"],
            "MAE_dc0_g": g["MAE_dc0_cm"], "MAE_dc0_prim": p["MAE_dc0_cm"],
            "MAE_dc0_ko": ko["MAE_dc0_cm"],
            "ratio": p["R_MAE"] / ko["R_MAE"] if ko["R_MAE"] > 0 else None,
            "wig_g": g["wiggle_cm"], "wig_prim": p["wiggle_cm"],
            "wig_ko": ko["wiggle_cm"], "wig_ko_lp_ceiling": None,
            "union_ok": mu[primary]["MAE_cm"] <= 1.05 * mu["global"]["MAE_cm"],
            "obj_ok": p["MAE_cm"] <= 1.05 * g["MAE_cm"],
            "corr_lp": r["diag"]["corr_kappa_ko_lp"].get(primary),
            "cv_r_gt": r["diag"]["cv_r_gt_obj"],
            "lowfreq_frac": r["diag"]["ko_spectrum"]["lowfreq_frac"],
            "flow_cov": r["diag"]["coverage"]["flow"],
            "eligible": ko["R_MAE"] >= KO_FLOOR,
        }
        per_clip.append(e)
        if e["eligible"]:
            eligible.append(e)

    def pooled_ratio(rows_, num_key, den_key, base_key="MAE_g"):
        num = sum(x[base_key] - x[num_key] for x in rows_)
        den = sum(x[base_key] - x[den_key] for x in rows_)
        return num / den if den > 0 else None

    pr = pooled_ratio(eligible, "MAE_prim", "MAE_ko")
    pr_dc0 = None
    den = sum(x["MAE_dc0_g"] - x["MAE_dc0_ko"] for x in eligible)
    if den > 0:
        pr_dc0 = sum(x["MAE_dc0_g"] - x["MAE_dc0_prim"] for x in eligible) / den
    ratios = [x["ratio"] for x in eligible if x["ratio"] is not None]
    med_ratio = float(np.median(ratios)) if ratios else None
    wig_reds = [x["wig_g"] / max(x["wig_prim"], 1e-6) for x in per_clip]
    med_wig = float(np.median(wig_reds))
    wig_ko_reds = [x["wig_g"] / max(x["wig_ko"], 1e-6) for x in per_clip]

    # bootstrap CI over eligible clips for the pooled ratio
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(2000):
        s = [eligible[i] for i in rng.integers(0, len(eligible), len(eligible))]
        v = pooled_ratio(s, "MAE_prim", "MAE_ko")
        if v is not None:
            boots.append(v)
    ci = [float(np.percentile(boots, q)) for q in (2.5, 97.5)] if boots else None

    crit_a = (pr is not None and pr >= 0.5) and (med_ratio is not None and med_ratio >= 0.5)
    crit_b = med_wig >= 2.5
    crit_c = all(x["obj_ok"] and x["union_ok"] for x in per_clip)
    smooth_gain = sum(x["MAE_g"] - x["MAE_smooth"] for x in per_clip)
    prim_gain = sum(x["MAE_g"] - x["MAE_prim"] for x in per_clip)
    crit_d = prim_gain > smooth_gain
    cond_a = pr_dc0 is not None and pr_dc0 >= 0.5

    if crit_a and crit_b and crit_c and crit_d:
        verdict = "GO"
    elif crit_b and crit_c and crit_d and cond_a:
        verdict = "CONDITIONAL GO (shape recovered; DC needs an absolute anchor)"
    else:
        verdict = "KILL"

    print(f"\n===== {label} (primary = {primary}, {len(rows)} clips) =====")
    hdr = (f"{'clip':22s} {'MAEg':>5s} {'prim':>5s} {'ko':>5s} {'R_prim':>7s} {'R_ko':>6s} "
           f"{'R/Rko':>6s} {'R_so':>6s} {'wig g/p':>8s} {'corr_lp':>7s} {'cv_rgt':>6s} "
           f"{'lof':>5s} {'elig':>4s}")
    print(hdr); print("-" * len(hdr))
    for x in per_clip:
        print(f"{x['clip']:22s} {x['MAE_g']:5.1f} {x['MAE_prim']:5.1f} {x['MAE_ko']:5.1f} "
              f"{x['R_prim']:7.3f} {x['R_ko']:6.3f} "
              f"{(x['ratio'] if x['ratio'] is not None else float('nan')):6.2f} "
              f"{x['R_so']:6.3f} {x['wig_g'] / max(x['wig_prim'], 1e-6):8.2f} "
              f"{(x['corr_lp'] if x['corr_lp'] is not None else float('nan')):7.2f} "
              f"{x['cv_r_gt']:6.3f} {x['lowfreq_frac']:5.2f} {str(x['eligible'])[:4]:>4s}")
    print()
    print(f"eligible clips (R_ko >= {KO_FLOOR}): {len(eligible)}/{len(per_clip)}")
    print(f"(a) pooled headroom ratio = {pr:.3f} (95% CI {ci})  |  median per-clip = {med_ratio:.3f}"
          f"  -> {'PASS' if crit_a else 'FAIL'}")
    print(f"    pooled ratio after DC removal = {pr_dc0:.3f}")
    print(f"(b) median wiggle reduction = {med_wig:.2f}x  (ko oracle achieves "
          f"{float(np.median(wig_ko_reds)):.2f}x)  -> {'PASS' if crit_b else 'FAIL'}")
    print(f"(c) no clip degraded >5% (obj & union) -> {'PASS' if crit_c else 'FAIL'}")
    print(f"(d) beats smooth control (pooled gain {prim_gain:.2f} vs {smooth_gain:.2f} cm) "
          f"-> {'PASS' if crit_d else 'FAIL'}")
    print(f"\nVERDICT [{label}]: {verdict}")

    # all configs, pooled over eligible clips (context, not verdict)
    print("\nall configs (pooled headroom ratio on eligible clips / median wiggle red.):")
    for cfg in ("rig_o", "rig_o_s", "rig_u_s", "rig_h_s", "flow", "flow_s",
                "fuse", "eb", "eb_s", "hyb", "hyb_s", "smooth", "so", "ko", "oracle"):
        try:
            num = sum(x["MAE_g"] - r["metrics"]["obj"][cfg]["MAE_cm"]
                      for x, r in zip(per_clip, rows) if x["eligible"])
            den = sum(x["MAE_g"] - x["MAE_ko"] for x in per_clip if x["eligible"])
            wr = float(np.median([x["wig_g"] / max(r["metrics"]["obj"][cfg]["wiggle_cm"], 1e-6)
                                  for x, r in zip(per_clip, rows)]))
            print(f"  {cfg:8s}: {num / den:6.3f}   {wr:5.2f}x")
        except KeyError:
            pass

    return {"label": label, "primary": primary, "per_clip": per_clip,
            "criteria": {"a": crit_a, "a_pooled": pr, "a_ci": ci,
                         "a_median": med_ratio, "a_dc0": pr_dc0,
                         "b": crit_b, "b_median_wig_red": med_wig,
                         "c": crit_c, "d": crit_d,
                         "d_gains_cm": [prim_gain, smooth_gain]},
            "verdict": verdict}


def main():
    rows = []
    for rj in sorted(glob.glob(os.path.join(CLIPS_ROOT, "*", "pivot_test", "result.json"))):
        with open(rj) as f:
            rows.append(json.load(f))
    if not rows:
        print("no results found"); return
    registered = evaluate(rows, PRIMARY, "REGISTERED: eb_s, all clips")
    holdout = [r for r in rows if r["clip"] != PILOT]
    amended = evaluate(holdout, AMENDED_PRIMARY,
                       "AMENDED (headline): flow_s, holdout = all but pilot")
    out = {"registered": registered, "amended_holdout": amended}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "aggregate.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote aggregate.json")


if __name__ == "__main__":
    main()
