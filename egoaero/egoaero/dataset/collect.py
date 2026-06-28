"""EgoAERO Sec-3 closed-loop data collection: reconstruct -> online quality assess ->
accept / repairable_accept / recapture, writing accepted sequences into a mock EgoDex-R."""
from __future__ import annotations
import json, os
import numpy as np

from ..config import load_config
from ..pipeline import run_pipeline
from . import schema, capture
from .difficulty import difficulty_score


def recon_summary(run_dir, quality_report):
    """Compute reconstruction summary metrics from a completed pipeline run.

    Args:
        run_dir: Path to the pipeline run directory.
        quality_report: Dict from quality.json (used for per_finger Q_rec).

    Returns:
        Dict with keys: 'occlusion', 'obj_motion_m', 'contact_richness'.
    """
    from ..bundle import Bundle
    s0 = Bundle.load(os.path.join(run_dir, "stage0_ego_io"))
    om = s0["obj_mask"]; hm = s0["hand_mask"]
    inter = (om & hm).reshape(om.shape[0], -1).sum(1)
    area = np.maximum(om.reshape(om.shape[0], -1).sum(1), 1)
    occlusion = float(np.mean(inter / area))
    with np.load(os.path.join(run_dir, "contract", "object_traj.npz")) as z:
        poses = z["obj_poses_t"]
    obj_motion_m = float(np.linalg.norm(poses[:, :3, 3] - poses[0, :3, 3], axis=1).max())
    qrec = [v["Q_rec"] for v in quality_report.get("per_finger", {}).values()]
    contact_richness = float(np.mean(qrec)) if qrec else 0.0
    return {"occlusion": occlusion, "obj_motion_m": obj_motion_m,
            "contact_richness": contact_richness}


def run_collection(out_dir, n_target, dataset_cfg, seed=0, work_root=None):
    """Run the closed-loop collection loop: synthetic source -> reconstruct -> quality assess ->
    accept/repairable_accept/recapture, writing accepted sequences into out_dir.

    Args:
        out_dir: Output directory to write accepted sequences and summary.json.
        n_target: Target number of accepted sequences to collect.
        dataset_cfg: Dataset configuration dict (from dataset.yaml).
        seed: Random seed for synthetic source generation.
        work_root: Working directory for pipeline run artifacts (temporary).

    Returns:
        Summary dict with keys: n_accepted, n_attempts, decisions, difficulty_hist,
        capabilities, total_frames.
    """
    os.makedirs(out_dir, exist_ok=True)
    work_root = work_root or os.path.join(out_dir, "_work")
    col = dataset_cfg["collection"]
    cap = dataset_cfg["capture"]
    dw = dataset_cfg["difficulty"]
    max_attempts = int(col["max_attempts"])
    clips = capture.synthetic_source(
        max_attempts, seed, int(col["num_frames"]),
        float(cap["tightness_min"]), float(cap["tightness_max"])
    )
    decisions = {"accept": 0, "repairable_accept": 0, "recapture": 0}
    diff_hist = {i: 0 for i in range(1, 6)}
    n_accepted = 0
    n_attempts = 0
    total_frames = 0
    for clip in clips:
        if n_accepted >= int(n_target):
            break
        n_attempts += 1
        run_dir = os.path.join(work_root, f"attempt_{n_attempts:04d}")
        cfg = load_config(overrides=capture.clip_overrides(clip))
        run_pipeline(cfg, run_dir, "all")
        with open(os.path.join(run_dir, "quality.json")) as f:
            q = json.load(f)
        decision = q["decision"]
        decisions[decision] = decisions.get(decision, 0) + 1
        if decision == "recapture":
            continue
        rs = recon_summary(run_dir, q)
        difficulty = difficulty_score(q, rs, dw)
        diff_hist[difficulty] = diff_hist.get(difficulty, 0) + 1
        seq_id = f"seq_{n_accepted:04d}"
        meta = {
            "task_description": clip["task_description"],
            "manipulated_object": clip["manipulated_object"],
            "relational_objects": clip["relational_objects"],
            "difficulty": difficulty,
            "decision": decision,
            "frames": int(clip["num_frames"]),
            "seq_id": seq_id,
        }
        schema.write_sequence(out_dir, seq_id, run_dir, meta)
        n_accepted += 1
        total_frames += int(clip["num_frames"])
    summary = {
        "n_accepted": n_accepted,
        "n_attempts": n_attempts,
        "decisions": decisions,
        "difficulty_hist": diff_hist,
        "total_frames": total_frames,
        "capabilities": {
            "obj_state": True,
            "asset_free": True,
            "depth": True,
            "slam": True,
            "contact_eval": True,
        },
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary
