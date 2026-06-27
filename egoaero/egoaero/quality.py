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
