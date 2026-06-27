"""Stage 8 (Sec 3 / App E): online quality assessment. Scores the reconstructed clip
on bounded recoverability (coarse=stage5 vs repaired=stage6) and emits an
accept / repairable_accept / recapture decision. Supplementary diagnostic — the 4D-HOI
contract output is unchanged."""
from __future__ import annotations
import json, os
import numpy as np

from ..bundle import Bundle
from .. import quality as Q
from .stage6_contact import active_window, _obj_world

NAME = "stage8_quality"; INDEX = 8


def run(ctx) -> Bundle:
    cfg = ctx.cfg; qc = cfg.quality
    s5 = ctx.load("stage5_ego_comp"); s6 = ctx.load("stage6_contact")
    coarse = s5["hand_verts_t"]; repaired = s6["hand_verts_t"]
    obj_poses = s5["obj_poses_t"]; ov = s5["obj_verts"]; of = s5["obj_faces"].astype(int)
    fidx = {k: (np.asarray(v, float) if k == "z_norm" else np.asarray(v, int))
            for k, v in s5.meta["finger_idx"].items()}
    labels = s5.meta["stage_labels"]; T = coarse.shape[0]
    window = active_window(labels)

    # object surface points+normals per active frame (reuse stage6 helper)
    obj_world_seq = [None] * T
    for t in window:
        obj_world_seq[t] = _obj_world(ov, of, obj_poses[t])

    gap_before = Q.per_finger_gap(coarse, fidx, obj_world_seq, window)
    gap_after = Q.per_finger_gap(repaired, fidx, obj_world_seq, window)
    delta = Q.per_finger_delta(coarse, repaired, fidx, window)

    rec = Q.recoverability(gap_after, delta, qc.eps_g_m, qc.eps_delta_m)
    budget = Q.repair_budget(delta, qc.delta_max_m)
    R_after = Q.residual_after(s6.meta["pen_after_mm"], gap_after, qc.pen_ref_mm, qc.gap_ref_mm)

    # object-moving flag per active frame (translation speed between consecutive frames)
    moving = np.zeros(len(window), bool)
    for jpos, t in enumerate(window):
        tp = max(t - 1, 0)
        speed = float(np.linalg.norm(obj_poses[t][:3, 3] - obj_poses[tp][:3, 3]))
        moving[jpos] = speed > qc.obj_move_thresh_m_per_frame
    U = Q.unresolved_ratio(gap_after, delta, moving, qc.eps_g_m, qc.eps_delta_m)

    Qval = Q.quality_score(R_after, budget, U, qc.alpha, qc.beta, qc.gamma)
    label, attr = Q.decision(Qval, rec, qc.q_accept, qc.q_repairable)

    per_finger = {f: {"gap_before_mm": float(np.median(gap_before[f]) * 1000) if len(window) else 0.0,
                      "gap_after_mm": float(np.median(gap_after[f]) * 1000) if len(window) else 0.0,
                      "Q_rec": rec[f]} for f in Q.H.FINGERS}
    report = {"Q": Qval, "decision": label, "per_finger": per_finger,
              "B_repair": budget, "R_after": R_after, "U_unresolved": U,
              "failure_attribution": attr}

    print(f"  quality: decision={label}  Q={Qval:.3f}  B_repair={budget:.3f}  "
          f"R_after={R_after:.3f}  U={U:.3f}")
    with open(os.path.join(ctx.run_dir, "quality.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    return Bundle(meta={"quality": report})
