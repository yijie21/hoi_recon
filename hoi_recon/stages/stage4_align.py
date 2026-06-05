"""Stage 4 — Spatial alignment into one metric world frame.

In:  hand motion (stage2), object trajectory (stage3), depth+camera (stage0).
Out: hand & object expressed in ONE world frame; resolved global scale gauge.
Method: lift both via camera extrinsics; solve a single global similarity to the
metric depth (Umeyama). In mock (world==camera, no metric depth) this is a
structural compose + scale=1, and we record the residual hand<->object contact
gap — the misalignment that stages 6-7 exist to fix.
"""
from __future__ import annotations

import numpy as np

from ..bundle import Bundle
from ..logging_utils import log
from ._scene import all_object_world, contact_gap

NAME = "stage4_align"
INDEX = 4


def run(ctx) -> Bundle:
    cfg = ctx.cfg
    s0 = ctx.load("stage0_preprocess")
    s2 = ctx.load("stage2_hand")
    s3 = ctx.load("stage3_object")
    T = int(s0.meta["T"])

    hand_verts = s2["verts"]
    hand_joints = s2["joints"]
    contact_idx = s2["contact_idx"].astype(int)
    obj_verts, obj_faces = s3["verts"], s3["faces"].astype(int)
    obj_poses = s3["poses"]

    # In real mode: transform hand (camera frame) and object into world via
    # extrinsics, then estimate one global metric scale from depth. In mock the
    # synthetic scene is already world-metric (extrinsics = I, scale = 1).
    world_scale = 1.0
    if not cfg.mock and s0.meta.get("has_depth"):
        # TODO(real): solve global similarity to metric depth (geometry.umeyama).
        log("real-mode metric scale solve not yet wired; using world_scale=1", "warn")

    objw, _ = all_object_world(obj_verts, obj_faces, obj_poses)
    gaps = np.array([contact_gap(hand_verts[i, contact_idx], objw[i]) for i in range(T)])
    log(f"contact-frame surface gap: min={gaps.min()*1000:.1f}mm "
        f"median={np.median(gaps)*1000:.1f}mm (pre-rectification)")

    meta = {"world_scale": world_scale,
            "gap_min_mm": float(gaps.min() * 1000),
            "gap_median_mm": float(np.median(gaps) * 1000)}
    arrays = {"hand_verts": hand_verts, "hand_joints": hand_joints,
              "contact_idx": contact_idx, "obj_verts": obj_verts,
              "obj_faces": obj_faces, "obj_poses": obj_poses,
              "obj_radius": s3.get("radius", np.array(0.0)),
              "gaps": gaps}
    obj_colors = s3.get("colors")              # present only for textured backends (sam3d)
    if obj_colors is not None:
        arrays["obj_colors"] = obj_colors
    hand_faces = s2.get("hand_faces")          # MANO mesh faces (real HaMeR hand)
    if hand_faces is not None:
        arrays["hand_faces"] = hand_faces
    return Bundle(arrays=arrays, meta=meta)
