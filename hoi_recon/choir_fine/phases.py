# hoi_recon/choir_fine/phases.py
"""Segment a clip into CHOIR's five interaction phases from per-frame contact presence
(and optional per-frame motion magnitude). Contact terms are applied only on
manipulation frames downstream; the static phases let the optimizer skip moving-only
terms where the hand is at rest. CHOIR §7.3."""
from __future__ import annotations

import numpy as np

PHASES = ["pre_static", "approach", "manipulation", "release", "post_static"]


def segment_phases(contact_present, motion=None, static_thresh=1e-3):
    """contact_present: (T,) bool. motion: optional (T,) float per-frame motion magnitude.
    Returns labels (T,) int indexing PHASES."""
    contact_present = np.asarray(contact_present, bool)
    T = len(contact_present)
    labels = np.full(T, PHASES.index("approach"), int)        # default: approach
    idx = np.where(contact_present)[0]

    if len(idx) == 0:
        if motion is not None:
            labels[np.asarray(motion) < static_thresh] = PHASES.index("pre_static")
        return labels

    f0, f1 = int(idx[0]), int(idx[-1])
    labels[f0:f1 + 1] = PHASES.index("manipulation")
    labels[:f0] = PHASES.index("approach")
    labels[f1 + 1:] = PHASES.index("release")

    if motion is not None:
        motion = np.asarray(motion, float)
        for t in range(f0):                                   # leading static run
            if motion[t] < static_thresh:
                labels[t] = PHASES.index("pre_static")
            else:
                break
        for t in range(T - 1, f1, -1):                        # trailing static run
            if motion[t] < static_thresh:
                labels[t] = PHASES.index("post_static")
            else:
                break
    return labels
