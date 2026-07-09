"""Stable-grasp detection: frames where the object moves rigidly with a
wrist. During such segments the hand's rotation carries the azimuth signal
that symmetric object geometry hides from depth+silhouette (HOT3D batch
finding: rotation error is a direct function of shape symmetry).

Related work (15-min pass, 2026-07-09): "ComPose: When to Trust Hands for
Object Pose Tracking" (arXiv 2605.23523) frames the same underlying idea —
decide, per frame, whether the hand or the object signal is more reliable for
6-DoF object tracking, and lean on the hand's rigid motion when it is. This
module's stable_grasp_mask is a lightweight, unsupervised proxy for that
"trust hands" decision: constant relative wrist<->object translation over a
centred window, while the wrist itself is actually moving, is exactly the
kinematic signature of a firm (no-slip) grasp under which object rotation
should track wrist rotation (see also the classic rigid-grasp constraint:
pose of a part is fixed relative to the end-effector while grasped with no
sliding). No learned trust-classifier is used here — the geometric velocity
test is deliberately simple, deterministic, and cheap."""
import numpy as np


def stable_grasp_mask(wrist, obj_t, v_rel_max=0.015, win=5):
    rel = obj_t - wrist
    v = np.zeros(len(rel))
    v[1:] = np.linalg.norm(np.diff(rel, axis=0), axis=1)
    k = np.ones(win) / win
    vs = np.convolve(v, k, mode="same")
    moving = np.zeros(len(rel))
    moving[1:] = np.linalg.norm(np.diff(wrist, axis=0), axis=1)
    ms = np.convolve(moving, k, mode="same")
    # grasp = relative motion small WHILE the hand actually moves
    return (vs < v_rel_max) & (ms > v_rel_max)
