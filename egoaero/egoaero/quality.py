"""EgoAERO App-E online quality assessment — pure scoring functions.

Reads the coarse hand (stage5) and the repaired hand (stage6) — App E's bounded
projection is exactly the stage6 contact optimization — and scores bounded
recoverability into a quality value Q and an accept/repairable/recapture decision.
No I/O, no re-optimization, deterministic. Constants live in cfg.quality (documented).
"""
from __future__ import annotations
import numpy as np

from .stages.stage6_contact import signed_distance
from .core import hand as H


def per_finger_gap(hand_verts_seq, finger_idx, obj_world_seq, window):
    """Per finger: median |distance| of its fingertip-pad vertices to the object
    surface, for each frame in `window`. Returns {finger: array[len(window)]} (metres)."""
    out = {f: np.zeros(len(window)) for f in H.FINGERS}
    for jpos, t in enumerate(window):
        ow, on = obj_world_seq[t]
        for f in H.FINGERS:
            pad = H.fingertip_pad_idx(finger_idx, f)
            if len(pad) == 0:
                out[f][jpos] = 0.0
                continue
            s, _ = signed_distance(hand_verts_seq[t][pad], ow, on)
            out[f][jpos] = float(np.median(np.abs(s)))
    return out


def per_finger_delta(coarse_verts_seq, repaired_verts_seq, finger_idx, window):
    """Per finger: mean pad-vertex Euclidean displacement between coarse and repaired
    hand, for each frame in `window`. Returns {finger: array[len(window)]} (metres)."""
    out = {f: np.zeros(len(window)) for f in H.FINGERS}
    for jpos, t in enumerate(window):
        for f in H.FINGERS:
            pad = H.fingertip_pad_idx(finger_idx, f)
            if len(pad) == 0:
                out[f][jpos] = 0.0
                continue
            disp = repaired_verts_seq[t][pad] - coarse_verts_seq[t][pad]
            out[f][jpos] = float(np.mean(np.linalg.norm(disp, axis=1)))
    return out


def recoverability(gap_after, delta, eps_g, eps_delta):
    """Q_rec^f = fraction of active frames where g_after < eps_g AND ||delta|| < eps_delta."""
    out = {}
    for f in H.FINGERS:
        ga, df = gap_after[f], delta[f]
        ok = (ga < eps_g) & (df < eps_delta)
        out[f] = float(np.mean(ok)) if len(ok) else 0.0
    return out


def repair_budget(delta, delta_max):
    """B_repair = median over all (frame, finger) of ||delta|| / delta_max."""
    alld = np.concatenate([delta[f] for f in H.FINGERS]) if delta else np.zeros(1)
    if alld.size == 0:
        return 0.0
    return float(np.median(alld) / max(delta_max, 1e-9))


def residual_after(pen_after_mm, gap_after, pen_ref_mm, gap_ref_mm):
    """R_after = pen_after/pen_ref + (median-over-fingers median-over-frames gap)*1000/gap_ref.
    Dimensionless remaining penetration + contact-gap residual."""
    per_finger_med = [np.median(gap_after[f]) for f in H.FINGERS if len(gap_after[f])]
    gap_med_m = float(np.median(per_finger_med)) if per_finger_med else 0.0
    return float(pen_after_mm / max(pen_ref_mm, 1e-9) + (gap_med_m * 1000.0) / max(gap_ref_mm, 1e-9))


def unresolved_ratio(gap_after, delta, object_moving, eps_g, eps_delta):
    """Fraction of object-moving active frames where NO finger has recoverable contact."""
    moving = np.asarray(object_moving, bool)
    n = len(moving)
    if n == 0 or moving.sum() == 0:
        return 0.0
    unresolved = 0
    for j in range(n):
        if not moving[j]:
            continue
        any_rec = any((gap_after[f][j] < eps_g) and (delta[f][j] < eps_delta) for f in H.FINGERS)
        if not any_rec:
            unresolved += 1
    return float(unresolved / moving.sum())
