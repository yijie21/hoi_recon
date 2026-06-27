"""Stage 7: reconstruction error report — penetration / contact-gap / jitter
before vs after adaptive contact optimization (the 'watch error fall' table)."""
from __future__ import annotations
import numpy as np
from ..bundle import Bundle

NAME = "stage7_eval"; INDEX = 7

def _jitter(joints):
    return float(np.mean(np.abs(np.diff(joints, 2, axis=0))))

def run(ctx) -> Bundle:
    s5 = ctx.load("stage5_ego_comp"); s6 = ctx.load("stage6_contact")
    report = {
        "pen_before_mm": s6.meta["pen_before_mm"], "pen_after_mm": s6.meta["pen_after_mm"],
        "gap_before_mm": s6.meta["gap_before_mm"], "gap_after_mm": s6.meta["gap_after_mm"],
        "hand_jitter_before": _jitter(s5["hand_joints_t"]),
        "hand_jitter_after": _jitter(s6["hand_joints_t"]),
    }
    print("  metric                    before ->  after")
    print(f"  penetration sum (mm)      {report['pen_before_mm']:.2f} -> {report['pen_after_mm']:.2f}")
    print(f"  contact gap (mm)          {report['gap_before_mm']:.2f} -> {report['gap_after_mm']:.2f}")
    print(f"  hand jitter               {report['hand_jitter_before']:.5f} -> {report['hand_jitter_after']:.5f}")
    return Bundle(meta={"report": report})
