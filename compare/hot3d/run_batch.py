"""Batch: run the best-strategy pipeline (real_forehoi_icp_joint) on N
selected HOT3D single-object interaction clips and score each against mocap
GT. Per clip: adapter (rectify + GT-depth raycast) -> pipeline -> side-by-side
overlay -> gt_pose_eval. Sequential (one GPU).

Usage: run_batch.py <selection.json> [--arm NAME] [--config PATH]
  selection.json: [{"clip": "clip-002025", "uid": "7", "cat": "bowl"}, ...]
"""
import argparse
import json
import os
import subprocess
import sys

DS = "/workspace/datasets/hot3d"
RC = "/workspace/code/hoi_recon/render_and_compare"
HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/workspace/miniconda3/envs/rc5090/bin/python"


def sh(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("selection")
    ap.add_argument("--arm", default="icpj")
    ap.add_argument("--config", default="configs/real_forehoi_icp_joint.yaml")
    a = ap.parse_args()
    sel = json.load(open(a.selection))
    summary = []
    for item in sel:
        clip, uid, cat = item["clip"], str(item["uid"]), item["cat"]
        num = clip.split("-")[1]
        inp = f"{DS}/rc_input_{num}_{cat}"
        run = f"{RC}/runs/hot3d_{cat}_{num}_{a.arm}"
        try:
            if not os.path.exists(f"{inp}/meta.json"):
                sh([PY, f"{HERE}/make_rc_input.py", f"{DS}/clips/{clip}", uid, inp])
            meta = json.load(open(f"{inp}/meta.json"))
            px, py_ = meta["prompt"]
            if not (0 <= px < meta["out"] and 0 <= py_ < meta["out"]):
                print(f"SKIP {clip} {cat}: prompt {meta['prompt']} outside frame",
                      flush=True)
                continue
            if not os.path.exists(f"{run}/stage8_eval/pseudo_gt.npz"):
                env = dict(os.environ,
                           RC_GT_DEPTH_DIR=f"{inp}/depth_png",
                           RC_GT_INTRINSICS=f"{inp}/intrinsics.npy")
                lp = f"{run}.log"
                with open(lp, "w") as lf:
                    subprocess.run([PY, "-m", "hoi_recon.cli", "--video",
                        f"{inp}/rgb.mp4", "--out", run, "--real", "--config",
                        a.config, "--depth", "gt", "--object-prompt",
                        f"{px:.1f}", f"{py_:.1f}"], check=True, cwd=RC,
                        env=env, stdout=lf, stderr=subprocess.STDOUT)
            log_txt = ""
            lp = f"{run}.log"
            if os.path.exists(lp):
                log_txt = open(lp, errors="ignore").read()
            if "falling back to depth-lift" in log_txt:
                raise RuntimeError("INVALID: stage3 fell back to depth-lift")
            sh([PY, f"{HERE}/make_rc_vs_gt_overlay.py", inp, run,
                f"{HERE}/rc_vs_gt_{cat}_{num}_{a.arm}.mp4"])
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
        except subprocess.CalledProcessError as e:
            print(f"FAIL {clip} {cat}: {e}", flush=True)
            summary.append({"clip": clip, "cat": cat, "error": str(e)})
    json.dump(summary, open(f"{HERE}/batch_summary_{a.arm}.json", "w"), indent=1)
    print("\n== batch summary ==")
    for s in summary:
        print(json.dumps(s))


if __name__ == "__main__":
    main()
