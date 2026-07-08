"""Mask quality gates for stage-1 object segmentation.

The two catastrophic HOT3D failures (mug+forearm merge, spatula->table leak)
were both silent stage-1 mask defects. This module quantifies the defects the
overlay videos showed: area jumps (leak), low temporal IoU (identity drift),
and high hand-box overlap (arm absorbed into the object mask)."""
import numpy as np

AREA_JUMP = 2.5        # x median area
TIOU_MIN = 0.45
HAND_OVERLAP_MAX = 0.55


def reprompt_point(mask):
    """Re-prompt point for a contaminated object mask -> ((x, y), case).

    The naive centroid of a mug+forearm merged mask lands on the forearm (the
    bigger blob), so a QA re-prompt from it tracks the arm exclusively (the
    HOT3D mug screen failure). Arms always enter from the frame edge; a held /
    desk object does not. So: split into 8-connected components, discard those
    touching the image border, and return the centroid of the largest
    surviving component (case "interior"). If nothing survives, fall back to
    the largest component's centroid (case "border-fallback")."""
    import cv2
    n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    best_interior, best_interior_area = 0, 0
    best_any, best_any_area = 0, 0
    for c in range(1, n):
        comp = labels == c
        a = int(comp.sum())
        if a > best_any_area:
            best_any, best_any_area = c, a
        touches_border = (comp[0].any() or comp[-1].any()
                          or comp[:, 0].any() or comp[:, -1].any())
        if not touches_border and a > best_interior_area:
            best_interior, best_interior_area = c, a
    if best_any == 0:                       # empty mask: degenerate fallback
        H, W = mask.shape
        return (W / 2.0, H / 2.0), "empty"
    if best_interior:
        c, case = best_interior, "interior"
    else:
        c, case = best_any, "border-fallback"
    ys, xs = np.where(labels == c)
    return (float(xs.mean()), float(ys.mean())), case


def qa_report(mask_paths, hand_boxes, hand_valid):
    T = len(mask_paths)
    area = np.zeros(T)
    tiou = np.zeros(max(T - 1, 0))
    hov = np.zeros(T)
    masks = [np.load(mp) if mp else np.zeros((1, 1), bool) for mp in mask_paths]
    for i, m in enumerate(masks):
        area[i] = m.sum()
    prev = None
    for i, m in enumerate(masks):
        if prev is not None and prev.shape == m.shape:
            u = (prev | m).sum()
            tiou[i - 1] = (prev & m).sum() / u if u else 0.0
        if hand_boxes is not None and hand_valid is not None and m.any():
            hb = np.zeros(m.shape, bool)
            H, W = m.shape
            for h in range(hand_boxes.shape[1]):
                b = hand_boxes[i, h]
                if hand_valid[i, h] and np.isfinite(b).all():
                    x0, y0, x1, y1 = np.clip(
                        b, 0, [W, H, W, H]).astype(int)
                    hb[y0:y1, x0:x1] = True
            hov[i] = (m & hb).sum() / m.sum()
        prev = m

    # Frame-to-frame area *ratio* jump (leak/merge onset), rather than a
    # global-median threshold: a global median is not robust when the clip
    # splits close to 50/50 between a stable phase and a leaked phase (a
    # leaked run of frames can then out-vote the stable run and hide its own
    # explosion behind an inflated median). A consecutive-ratio jump flags
    # the leak at the frame it actually starts, independent of run lengths.
    jump_at = None
    for i in range(1, T):
        a0, a1 = area[i - 1], area[i]
        if a0 <= 0 or a1 <= 0:
            continue
        if max(a0, a1) / min(a0, a1) > AREA_JUMP:
            jump_at = i
            break

    med = np.median(area[area > 0]) if (area > 0).any() else 0.0
    bad = bool(jump_at is not None
               or (len(tiou) and np.median(tiou) < TIOU_MIN)
               or (hov.mean() > HAND_OVERLAP_MAX))

    # Anchor candidate: large, stable, hand-free mask *before* any detected
    # jump — frames at/after a leak onset are untrustworthy re-prompt sources.
    stable_end = jump_at if jump_at is not None else T
    score = area / max(med, 1.0)
    score[stable_end:] = 0.0
    score = score * (1.0 - hov)
    return {"area": area, "tiou": tiou, "hand_overlap": hov,
            "bad": bad, "best_frame": int(np.argmax(score))}
