"""Stage 8 — Evaluation & error attribution (the research payload).

Mock mode: re-derives the synthetic ground truth and measures each stage's error,
showing the error-reduction story (raw perception -> contact-aware optimization):
  * hand joint error + jitter   (stage2 raw  vs  stage5 smoothed)
  * object translation error     (stage3 raw  vs  stage7 optimized)
  * penetration depth            (stage5      vs  stage7)
  * contact F1                   (stage6/stage7 prediction vs GT)
  * contact-frame surface gap    (stage5      vs  stage7)
Real mode: reports self-consistency diagnostics (gap, penetration, #contacts) and
exports stage7 as pseudo-GT + intermediate signals for the feed-forward model.
"""
from __future__ import annotations

import json
import os

import numpy as np

from ..bundle import Bundle
from ..logging_utils import log
from ._scene import all_object_world, radial_penetration
from ..mock.scene import generate_mock_hoi

NAME = "stage8_eval"
INDEX = 8


def _mpjpe_mm(a, b):
    return float(np.mean(np.linalg.norm(a - b, axis=-1)) * 1000)


def _penetration_sum(bundle):
    ow, on = all_object_world(bundle["obj_verts"], bundle["obj_faces"].astype(int),
                              bundle["obj_poses"])
    hv = bundle["hand_verts"]
    tot = 0.0
    for i in range(hv.shape[0]):
        depth, _ = radial_penetration(hv[i], ow[i])
        tot += float(depth.sum())
    return tot


def _prf(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def run(ctx) -> Bundle:
    cfg = ctx.cfg
    s2 = ctx.load("stage2_hand")
    s5 = ctx.load("stage5_coarse_fit")
    s6 = ctx.load("stage6_rectify")
    s7 = ctx.load("stage7_contact_optim")
    s3 = ctx.load("stage3_object")
    T = s7["hand_joints"].shape[0]

    report = {"mock": bool(cfg.mock), "T": T}

    if cfg.mock:
        gt = generate_mock_hoi(T, seed=cfg.seed)
        gt_centers = gt.obj_poses[:, :3, 3]
        gt_contact = gt.gt_contact_mask

        report["hand"] = {
            "mpjpe_mm_raw_stage2": _mpjpe_mm(s2["joints"], gt.hand_joints),
            "mpjpe_mm_smoothed_stage5": _mpjpe_mm(s5["hand_joints"], gt.hand_joints),
            "jitter_accel_stage2": float(np.mean(np.abs(np.diff(s2["joints"], 2, 0)))),
            "jitter_accel_stage5": float(np.mean(np.abs(np.diff(s5["hand_joints"], 2, 0)))),
        }
        report["object"] = {
            "transl_err_mm_raw_stage3": _mpjpe_mm(s3["poses"][:, :3, 3], gt_centers),
            "transl_err_mm_optim_stage7": _mpjpe_mm(s7["obj_poses"][:, :3, 3], gt_centers),
        }
        report["penetration_depth_sum"] = {
            "stage5": _penetration_sum(s5),
            "stage7": _penetration_sum(s7),
        }
        report["contact_f1"] = {
            "stage6_rectify": _prf(s6["contact_map"], gt_contact),
            "stage7_optim": _prf(s7["contact_map"], gt_contact),
        }
        any_gt = gt_contact.any(1)
        report["contact_frame_gap_mm"] = {
            "stage5": float(np.median(s5["gaps"][any_gt]) * 1000) if any_gt.any() else None,
            "stage7": float(np.median(s7["gaps"][any_gt]) * 1000) if any_gt.any() else None,
        }
    else:
        report["self_consistency"] = {
            "n_active_contacts": int(s7["contact_map"].sum()),
            "gap_median_mm": float(np.median(s7["gaps"]) * 1000),
            "penetration_depth_sum": _penetration_sum(s7),
        }

    # export pseudo-GT + distillation signals for the feed-forward model
    pg = os.path.join(ctx.stage_dir(NAME), "pseudo_gt.npz")
    os.makedirs(ctx.stage_dir(NAME), exist_ok=True)
    np.savez(pg, hand_verts=s7["hand_verts"], hand_joints=s7["hand_joints"],
             obj_verts=s7["obj_verts"], obj_faces=s7["obj_faces"],
             obj_poses=s7["obj_poses"], contact_map=s7["contact_map"],
             rectify_delta=s6["rectify_delta"], object_delta=s7["object_delta"])

    with open(os.path.join(ctx.stage_dir(NAME), "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    _print_report(report)

    return Bundle(meta=report, assets={"pseudo_gt": pg})


def _row(label, raw, fixed, unit="", better="down"):
    arrow = "↓" if better == "down" else "↑"
    delta = ""
    if isinstance(raw, (int, float)) and isinstance(fixed, (int, float)):
        if better == "down" and raw > 0:
            delta = f"  ({100*(raw-fixed)/raw:+.0f}%)"
        elif better == "up":
            delta = f"  ({fixed-raw:+.2f})"
    return f"  {label:34s} {raw:>10.3f} → {fixed:>10.3f} {unit}{delta}"


def _print_report(r):
    log("════════ STAGE 8: error attribution report ════════", "stage")
    if not r["mock"]:
        sc = r["self_consistency"]
        log(f"  [real] active contacts={sc['n_active_contacts']}  "
            f"gap median={sc['gap_median_mm']:.1f}mm  "
            f"penetration={sc['penetration_depth_sum']:.3f}", "info")
        log("  exported stage7 as pseudo-GT for the feed-forward model.", "ok")
        return
    h, o = r["hand"], r["object"]
    print()
    print("  metric                                raw(percep) →  refined")
    print("  " + "-" * 60)
    print(_row("hand MPJPE (mm)", h["mpjpe_mm_raw_stage2"], h["mpjpe_mm_smoothed_stage5"], "mm"))
    print(_row("hand jitter/accel", h["jitter_accel_stage2"], h["jitter_accel_stage5"]))
    print(_row("object transl err (mm)", o["transl_err_mm_raw_stage3"], o["transl_err_mm_optim_stage7"], "mm"))
    print(_row("penetration depth sum", r["penetration_depth_sum"]["stage5"], r["penetration_depth_sum"]["stage7"]))
    g = r["contact_frame_gap_mm"]
    if g["stage5"] is not None:
        print(_row("contact-frame gap (mm)", g["stage5"], g["stage7"], "mm"))
    f6 = r["contact_f1"]["stage6_rectify"]["f1"]
    f7 = r["contact_f1"]["stage7_optim"]["f1"]
    print(_row("contact F1", f6, f7, "", better="up"))
    print("  " + "-" * 60)
    print("  ↓ lower is better except contact F1 (↑). raw = composed perception,")
    print("    refined = after contact-aware reasoning. pseudo-GT exported for distillation.")
    print()
