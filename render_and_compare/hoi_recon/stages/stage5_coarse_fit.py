"""Stage 5 — Contact-agnostic 4D fit (coarse).  ← first watchable result.

In:  aligned scene (stage4) + masks/keypoints.
Out: temporally smooth hand motion + object 6D trajectory; constant betas; NO
     contact reasoning yet (hands/object may still float or interpenetrate).
Method: temporal smoothing (moving average) of hand verts/joints and object
translation; in real mode this is a joint optimization also minimizing silhouette
+ 2D-keypoint reprojection. The output is the coarse 4D HOI you watch first.
"""
from __future__ import annotations

import numpy as np

from ..bundle import Bundle
from ..logging_utils import log
from ._scene import all_object_world, contact_gap

NAME = "stage5_coarse_fit"
INDEX = 5


def _smooth(x, w):
    """Moving-average smoothing along axis 0 (reflect-padded)."""
    if w <= 1:
        return x
    k = np.ones(w) / w
    pad = w // 2
    xp = np.concatenate([x[pad:0:-1], x, x[-2:-pad - 2:-1]], axis=0)[:x.shape[0] + 2 * pad]
    out = np.empty_like(x)
    flat = xp.reshape(xp.shape[0], -1)
    of = np.empty((x.shape[0], flat.shape[1]))
    for c in range(flat.shape[1]):
        of[:, c] = np.convolve(flat[:, c], k, mode="valid")[:x.shape[0]]
    return of.reshape(x.shape)


def run(ctx) -> Bundle:
    cfg = ctx.cfg
    s4 = ctx.load("stage4_align")
    w = int(cfg.smoothing.window)
    T = s4["hand_verts"].shape[0]

    hand_verts = _smooth(s4["hand_verts"], w)
    hand_joints = _smooth(s4["hand_joints"], w)
    obj_poses = s4["obj_poses"].copy()
    obj_poses[:, :3, 3] = _smooth(obj_poses[:, :3, 3], w)   # smooth translation only

    contact_idx = s4["contact_idx"].astype(int)
    objw, _ = all_object_world(s4["obj_verts"], s4["obj_faces"].astype(int), obj_poses)
    gaps = np.array([contact_gap(hand_verts[i, contact_idx], objw[i]) for i in range(T)])

    accel = float(np.mean(np.abs(np.diff(hand_joints, 2, axis=0))))
    log(f"smoothed (w={w}); hand accel/jitter -> {accel:.5f}; "
        f"gap median={np.median(gaps)*1000:.1f}mm")

    arrays = {"hand_verts": hand_verts, "hand_joints": hand_joints,
              "contact_idx": contact_idx, "obj_verts": s4["obj_verts"],
              "obj_faces": s4["obj_faces"], "obj_poses": obj_poses,
              "obj_radius": s4.get("obj_radius", np.array(0.0)), "gaps": gaps}
    if s4.get("obj_colors") is not None:
        arrays["obj_colors"] = s4["obj_colors"]
    if s4.get("hand_faces") is not None:
        arrays["hand_faces"] = s4["hand_faces"]
    return Bundle(arrays=arrays,
        meta={"smoothing_window": w, "jitter_accel": accel,
              "gap_median_mm": float(np.median(gaps) * 1000)})
