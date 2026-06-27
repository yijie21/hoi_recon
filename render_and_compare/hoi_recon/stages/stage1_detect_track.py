"""Stage 1 — Detection, hand sides & segmentation (2D cues).

In:  frames + camera (stage0).
Out: hand_boxes[T,2,4] (left,right; xyxy), hand_valid[T,2], object_box[T,4],
     object masks (modal + amodal) and object point tracks (real mode).
Backends (real): WiLoR det-head, SAM 2, amodal video seg, CoTracker3.
Errors logged: mask IoU stability, hand-object mask overlap, track confidence.

Mock: project the synthetic hand/object to 2D boxes so the downstream contract is
exercised. Pixel masks are skipped in mock (stages 4-7 operate in 3D), but the
object silhouette is summarized by a projected box + radius.
"""
from __future__ import annotations

import numpy as np

from ..bundle import Bundle
from ..geometry import transform_points
from ..mock.scene import generate_mock_hoi

NAME = "stage1_detect_track"
INDEX = 1


def _project(K, pts):
    z = np.clip(pts[..., 2:3], 1e-6, None)
    uv = (pts / z) @ K.T
    return uv[..., :2]


def _box(uv):
    lo, hi = uv.min(0), uv.max(0)
    return np.array([lo[0], lo[1], hi[0], hi[1]])


def run(ctx) -> Bundle:
    cfg = ctx.cfg
    s0 = ctx.load("stage0_preprocess")
    K = s0["intrinsics"]
    T = int(s0.meta["T"])

    if cfg.mock:
        scene = generate_mock_hoi(T, seed=cfg.seed,
                                  image_size=(s0.meta["H"], s0.meta["W"]),
                                  fps=s0.meta["fps"])
        hand_boxes = np.full((T, 2, 4), np.nan)
        hand_valid = np.zeros((T, 2), bool)
        object_box = np.zeros((T, 4))
        for i in range(T):
            huv = _project(K, scene.hand_verts[i])
            hand_boxes[i, 1] = _box(huv)         # slot 1 = right hand
            hand_valid[i, 1] = True
            ow = transform_points(scene.obj_verts, scene.obj_poses[i])
            object_box[i] = _box(_project(K, ow))
        meta = {"has_masks": False, "hands": ["left", "right"],
                "mask_iou_stability": None, "hand_object_overlap": None}
        return Bundle(
            arrays={"hand_boxes": hand_boxes, "hand_valid": hand_valid,
                    "object_box": object_box},
            meta=meta)

    # --- real: YOLO hand detection + SAM2 object segmentation ---
    from ..backends.real_perception import (detect_hands, segment_object,
                                            _object_prompt, list_frames)
    from ..logging_utils import log
    frames_dir = s0.assets["frames_dir"]
    frame_paths = list_frames(frames_dir)
    H, W = int(s0.meta["H"]), int(s0.meta["W"])

    hand_boxes, hand_valid = detect_hands(cfg, frame_paths)
    log(f"detected hands: L={int(hand_valid[:,0].sum())} R={int(hand_valid[:,1].sum())} frames")
    prompt = _object_prompt(hand_boxes, hand_valid, (H, W))
    log(f"SAM2 object prompt @ ({prompt[0]:.0f},{prompt[1]:.0f})")
    masks_dir, mask_paths = segment_object(cfg, frames_dir, frame_paths, prompt,
                                           ctx.stage_dir(NAME))

    object_box = np.full((T, 4), np.nan)
    for i, mp in enumerate(mask_paths):
        if mp is None:
            continue
        m = np.load(mp)
        ys, xs = np.where(m)
        if len(xs):
            object_box[i] = [xs.min(), ys.min(), xs.max(), ys.max()]

    return Bundle(
        arrays={"hand_boxes": hand_boxes, "hand_valid": hand_valid,
                "object_box": object_box},
        meta={"has_masks": True, "hands": ["left", "right"],
              "object_prompt": list(prompt)},
        assets={"masks_dir": masks_dir})
