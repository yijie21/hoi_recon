"""Batch: run the best-strategy pipeline (real_forehoi_icp_joint) on N
selected HOT3D single-object interaction clips and score each against mocap
GT. Per clip: adapter (rectify + GT-depth raycast) -> pipeline -> side-by-side
overlay -> gt_pose_eval. Sequential (one GPU).

Stage-cache promotion: when a new run is created, if the best incumbent has identical
masks, reuse its stage2_hand and stage3_object directories. This removes SAM-3D GPU
nondeterminism noise from stage comparisons.

Usage: run_batch.py <selection.json> [--arm NAME] [--config PATH]
  selection.json: [{"clip": "clip-002025", "uid": "7", "cat": "bowl"}, ...]
"""
import argparse
import json
import os
import shutil
import subprocess

DS = "/workspace/datasets/hot3d"
RC = "/workspace/code/hoi_recon/render_and_compare"
HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/workspace/miniconda3/envs/rc5090/bin/python"


def sh(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def masks_identical(run_a, run_b):
    """Check if two runs have identical masks (same files, same content)."""
    import numpy as np
    masks_a = f"{run_a}/stage1_detect_track/masks"
    masks_b = f"{run_b}/stage1_detect_track/masks"

    # Both dirs must exist
    if not os.path.isdir(masks_a) or not os.path.isdir(masks_b):
        return False

    # Same file list
    files_a = sorted(os.listdir(masks_a))
    files_b = sorted(os.listdir(masks_b))
    if files_a != files_b:
        return False

    # Every .npy must be array-equal
    for fname in files_a:
        path_a = os.path.join(masks_a, fname)
        path_b = os.path.join(masks_b, fname)
        if not (os.path.isfile(path_a) and os.path.isfile(path_b)):
            return False
        if fname.endswith('.npy'):
            try:
                arr_a = np.load(path_a)
                arr_b = np.load(path_b)
                if not np.array_equal(arr_a, arr_b):
                    return False
            except Exception:
                return False
    return True


def promote_cached_stages(run, cat, num, arm, best_arm):
    """Promote stage2_hand and stage3_object from best incumbent if masks identical.
    Returns True if promotion happened, False otherwise."""
    if arm == best_arm:
        return False  # Don't promote when arm == best_arm

    # Try to locate incumbent: first exact name, then legacy name
    incumbent = f"{RC}/runs/hot3d_{cat}_{num}_{best_arm}"
    legacy_incumbent = f"{RC}/runs/hot3d_{cat}_{best_arm}"

    # Handle symlinks (e.g., hot3d_vase_002500_icpj may be a symlink)
    if os.path.islink(legacy_incumbent):
        incumbent_to_check = os.path.realpath(legacy_incumbent)
    elif os.path.isdir(incumbent):
        incumbent_to_check = incumbent
    elif os.path.isdir(legacy_incumbent):
        incumbent_to_check = legacy_incumbent
    else:
        incumbent_to_check = None

    if incumbent_to_check is None or not os.path.isdir(incumbent_to_check):
        return False

    # Check if masks are identical
    if not masks_identical(incumbent_to_check, run):
        return False

    # Copy stage2 and stage3
    for stage in ("stage2_hand", "stage3_object"):
        src = os.path.join(incumbent_to_check, stage)
        dst = os.path.join(run, stage)
        if os.path.isdir(src) and not os.path.isdir(dst):
            shutil.copytree(src, dst)

    print(f"PROMOTE {cat}: stage2+3 reused from {best_arm} (masks identical — removes SAM-3D redraw noise)", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("selection")
    ap.add_argument("--arm", default="icpj")
    ap.add_argument("--config", default="configs/real_forehoi_icp_joint.yaml")
    a = ap.parse_args()
    sel = json.load(open(a.selection))
    summary = []

    # Read best arm for cache promotion
    best_arm_path = f"{HERE}/scores/BEST_ARM"
    best_arm = open(best_arm_path).read().strip() if os.path.exists(best_arm_path) else "icpj"

    for item in sel:
        clip, uid, cat = item["clip"], str(item["uid"]), item["cat"]
        num = clip.split("-")[1]
        inp = f"{DS}/rc_input_{num}_{cat}"
        run = f"{RC}/runs/hot3d_{cat}_{num}_{a.arm}"
        lp = f"{run}.log"
        try:
            if not os.path.exists(f"{inp}/meta.json"):
                sh([PY, f"{HERE}/make_rc_input.py", f"{DS}/clips/{clip}", uid, inp])
            meta = json.load(open(f"{inp}/meta.json"))
            px, py_ = meta["prompt"]
            if not (0 <= px < meta["out"] and 0 <= py_ < meta["out"]):
                print(f"SKIP {clip} {cat}: prompt {meta['prompt']} outside frame",
                      flush=True)
                continue

            # Pipeline execution with optional stage-cache promotion
            if not os.path.exists(f"{run}/stage8_eval/pseudo_gt.npz"):
                env = dict(os.environ,
                           RC_GT_DEPTH_DIR=f"{inp}/depth_png",
                           RC_GT_INTRINSICS=f"{inp}/intrinsics.npy")

                # Phase 1: Run stages 0-1 to generate masks
                with open(lp, "w") as lf:
                    subprocess.run([PY, "-m", "hoi_recon.cli", "--video",
                        f"{inp}/rgb.mp4", "--out", run, "--real", "--config",
                        a.config, "--depth", "gt", "--object-prompt",
                        f"{px:.1f}", f"{py_:.1f}", "--stages", "0-1"], check=True, cwd=RC,
                        env=env, stdout=lf, stderr=subprocess.STDOUT)

                # Phase 2: Try to promote cached stages from best incumbent
                promote_cached_stages(run, cat, num, a.arm, best_arm)

                # Phase 3: Run full pipeline (resumes from cached 0-3 if promoted)
                with open(lp, "a") as lf:
                    subprocess.run([PY, "-m", "hoi_recon.cli", "--video",
                        f"{inp}/rgb.mp4", "--out", run, "--real", "--config",
                        a.config, "--depth", "gt", "--object-prompt",
                        f"{px:.1f}", f"{py_:.1f}"], check=True, cwd=RC,
                        env=env, stdout=lf, stderr=subprocess.STDOUT)

            log_txt = ""
            if os.path.exists(lp):
                log_txt = open(lp, errors="ignore").read()
            if "falling back to depth-lift" in log_txt:
                raise RuntimeError("INVALID: stage3 fell back to depth-lift")
            if os.path.exists(f"{HERE}/make_rc_vs_gt_overlay.py"):  # diagnostic overlay (dev branch)
                sh([PY, f"{HERE}/make_rc_vs_gt_overlay.py", inp, run,
                    f"{HERE}/overlays/rc_vs_gt_{cat}_{num}_{a.arm}.mp4"])
            sh([PY, f"{HERE}/gt_pose_eval_hot3d.py", inp, run])
            r = json.load(open(f"{HERE}/gt_pose_hot3d_{os.path.basename(run)}.json"))
            v = list(r.values())[0]
            summary.append({"clip": clip, "cat": cat,
                            "chamfer_mm": round(v["chamfer_mm_med"], 1),
                            "centroid_cm": round(v["centroid_cm_med"], 2),
                            "rot_traj_med": round(v["rot_traj_deg_med"], 1),
                            "rot_traj_p90": round(v["rot_traj_deg_p90"], 1),
                            "rot_abs_med": round(v["rot_deg_med"], 1)})
            print("RESULT", json.dumps(summary[-1]), flush=True)
        except (subprocess.CalledProcessError, RuntimeError) as e:
            print(f"FAIL {clip} {cat}: {e}", flush=True)
            summary.append({"clip": clip, "cat": cat, "error": str(e)})
    json.dump(summary, open(f"{HERE}/scores/batch_summary_{a.arm}.json", "w"), indent=1)
    print("\n== batch summary ==")
    for s in summary:
        print(json.dumps(s))


if __name__ == "__main__":
    main()
