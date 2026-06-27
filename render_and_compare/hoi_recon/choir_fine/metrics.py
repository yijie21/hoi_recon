# hoi_recon/choir_fine/metrics.py
"""Fine-stage proxy metrics for the evaluation harness (no GT needed):
  contact_gap        — median nearest hand->object surface distance on contact frames
  penetration_depth  — summed one-sided penetration (hand inside object) over the clip
These feed scripts/object_confidence.py for per-preset A/B + ablation."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def contact_gap(hand_contact, obj_surface, contact_present):
    """hand_contact: (T,Nh,3) hand contact verts. obj_surface: (T,No,3) object surface
    points (world). contact_present: (T,) bool. Returns the median over contact frames
    of the mean nearest-surface distance, or nan if no contact frames."""
    hand_contact = np.asarray(hand_contact, float)
    obj_surface = np.asarray(obj_surface, float)
    contact_present = np.asarray(contact_present, bool)
    gaps = []
    for t in np.where(contact_present)[0]:
        if obj_surface[t].shape[0] == 0:
            continue
        d, _ = cKDTree(obj_surface[t]).query(hand_contact[t], k=1)
        gaps.append(float(np.mean(d)))
    return float(np.median(gaps)) if gaps else float("nan")


def penetration_depth(hand_verts, nearest_surface, surface_normal):
    """hand_verts: (T,Nh,3). nearest_surface/surface_normal: (T,Nh,3) the nearest object
    surface point and its outward normal for each hand vertex. Returns summed one-sided
    penetration: sum of max(0, (surface - hand) . normal) over all vertices/frames
    (positive when the hand vertex is inside the object). Normals are normalized internally
    so non-unit normals do not scale the depth."""
    hand_verts = np.asarray(hand_verts, float)
    nearest_surface = np.asarray(nearest_surface, float)
    surface_normal = np.asarray(surface_normal, float)
    n = surface_normal / (np.linalg.norm(surface_normal, axis=-1, keepdims=True) + 1e-12)
    signed = ((nearest_surface - hand_verts) * n).sum(-1)                # >0 => inside
    return float(np.clip(signed, 0.0, None).sum())
