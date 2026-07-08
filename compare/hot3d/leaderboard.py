"""Leaderboard + acceptance gate for the HOT3D improvement loop (see spec
docs/superpowers/specs/2026-07-08-hot3d-improvement-loop-design.md).

Noise floor: SAM-3D mesh generation is GPU-nondeterministic (same seed, same masks
→ slightly different mesh → object geometry shifts). Measured redraw noise: chamfer
shifts up to ~2mm, rot_traj up to ~17° on near-symmetric objects. Regression thresholds
use absolute floors (sub-floor changes are measurement noise, not real regressions).
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REG = 1.20          # per-clip regression tolerance
TIE_MM = 1.0        # worst-chamfer tie band
ABS_FLOOR = {"chamfer_mm": 2.0, "rot_traj_med": 5.0}  # measured SAM-3D redraw noise floors


def load_arm(arm):
    p = f"{HERE}/batch_summary_{arm}.json"
    rows = json.load(open(p))
    return {r["cat"]: r for r in rows if "error" not in r}


def gate(cand, best):
    if set(cand) != set(best):
        return False, f"clip sets differ: {sorted(set(best) ^ set(cand))}"
    for c in best:
        for k in ("chamfer_mm", "rot_traj_med"):
            # Regression only if BOTH relative and absolute thresholds exceeded
            rel_regress = cand[c][k] > best[c][k] * REG
            abs_regress = cand[c][k] > best[c][k] + ABS_FLOOR[k]
            if rel_regress and abs_regress:
                return False, f"{c} regresses on {k}: {best[c][k]} -> {cand[c][k]}"
    wc_c = max(v["chamfer_mm"] for v in cand.values())
    wc_b = max(v["chamfer_mm"] for v in best.values())
    if wc_c < wc_b - TIE_MM:
        return True, f"worst-clip chamfer {wc_b:.1f} -> {wc_c:.1f} mm"
    if abs(wc_c - wc_b) <= TIE_MM:
        mr_c = sum(v["rot_traj_med"] for v in cand.values()) / len(cand)
        mr_b = sum(v["rot_traj_med"] for v in best.values()) / len(best)
        if mr_c < mr_b:
            return True, f"chamfer tied; mean rot_traj {mr_b:.1f} -> {mr_c:.1f} deg"
        return False, f"chamfer tied; mean rot_traj {mr_b:.1f} -> {mr_c:.1f} (no gain)"
    return False, f"worst-clip chamfer worsens {wc_b:.1f} -> {wc_c:.1f} mm"


def best_arm():
    p = f"{HERE}/BEST_ARM"
    return open(p).read().strip() if os.path.exists(p) else "icpj"


def render():
    lines = ["# HOT3D 6-clip leaderboard", "",
             "| arm | clip | chamfer mm | centroid cm | rot_traj | p90 | rot_abs |",
             "|---|---|---|---|---|---|---|"]
    for p in sorted(glob.glob(f"{HERE}/batch_summary_*.json")):
        arm = os.path.basename(p)[len("batch_summary_"):-len(".json")]
        tag = f"**{arm}**" if arm == best_arm() else arm
        for r in json.load(open(p)):
            if "error" in r:
                lines.append(f"| {tag} | {r['cat']} | ERROR | | | | |")
                continue
            lines.append(f"| {tag} | {r['cat']} | {r['chamfer_mm']} | "
                         f"{r['centroid_cm']} | {r['rot_traj_med']} | "
                         f"{r['rot_traj_p90']} | {r['rot_abs_med']} |")
    open(f"{HERE}/LEADERBOARD.md", "w").write("\n".join(lines) + "\n")
    print(f"rendered LEADERBOARD.md (best={best_arm()})")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "render":
        render()
    elif cmd == "check":
        ok, why = gate(load_arm(sys.argv[2]), load_arm(best_arm()))
        print(("GATE PASS: " if ok else "GATE FAIL: ") + why)
        if ok:
            open(f"{HERE}/BEST_ARM", "w").write(sys.argv[2])
            render()
        sys.exit(0 if ok else 1)
