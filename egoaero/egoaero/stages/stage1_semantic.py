"""Stage 1 (§2.1.1): MLLM semantic preprocessing. Mock returns GT masks + a
least-occluded seed frame; real mode calls MLLM + SAM3 (backends/real.py)."""
from __future__ import annotations
import numpy as np
from ..bundle import Bundle

NAME = "stage1_semantic"; INDEX = 1

def run(ctx) -> Bundle:
    cfg = ctx.cfg
    s0 = ctx.load("stage0_ego_io")
    if not cfg.mock:
        raise NotImplementedError("real MLLM+SAM3 backend — see backends/real.py")
    om = s0["obj_mask"]; hm = s0["hand_mask"]
    area = om.reshape(om.shape[0], -1).sum(1).astype(float)
    occ = (om & hm).reshape(om.shape[0], -1).sum(1).astype(float)
    score = area - occ                                   # least-occluded, most-visible
    seed = int(np.argmax(score))
    return Bundle(arrays={"obj_mask": om, "hand_mask": hm},
                  meta={"seed_frame": seed, "target_object": "object",
                        "related_objects": ["table"],
                        "stage_labels": s0.meta["stage_labels"]})
