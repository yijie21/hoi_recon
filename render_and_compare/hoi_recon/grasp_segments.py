"""Stable-grasp detection for the T3 grasp-rigidity prior.

During frames where the hand firmly holds the object, the object co-rotates
with the wrist — supplying the per-frame azimuth signal that symmetric object
geometry hides from depth+silhouette (HOT3D batch finding: rotation error is a
direct function of shape symmetry). `_joint_refine` (object_icp.py) turns that
into a loss: on consecutive stable-grasp frame pairs it pulls the object's
rotation delta toward the wrist's.

Detection is CONTACT-based (v2), not velocity-based. An earlier
relative-velocity detector (`stable_grasp_mask`, removed) mis-fired on real
clips for two structural reasons:
  (1) object<->wrist RELATIVE velocity is swamped by ICP jitter in the object
      trajectory — the very noise the prior exists to fix — so a "small
      relative motion" test failed on genuinely grasped frames; and
  (2) gating on TRANSLATIONAL hand speed excluded in-hand ROTATION (the wrist
      pivots ~in place: ~1 cm/frame translation while rotating ~5 deg/frame) —
      exactly the frames the rigidity term must catch.
On the vase clip (held+rotated the whole time) that detector fired only
~10/150 frames, barely touching the f100-150 in-hand-rotation blow-up.

Contact detection sidesteps both: the per-frame minimum distance between hand
vertices and the posed object surface is a RELATIVE hand-to-object quantity,
invariant to absolute-trajectory jitter, and it says nothing about hand speed
so in-hand rotation is included. When neither the hand nor the object is
rotating during a held span the rigidity term is naturally inert (dRo ≈ dRw ≈
I), so contact alone is a safe grasp definition.

Related work (lit pass): "ComPose: When to Trust Hands for Object Pose
Tracking" (arXiv 2605.23523) frames the same decision — when to trust the
hand's rigid motion for object pose — with a learned per-frame trust
classifier. `grasp_mask_contact` is an unsupervised, geometry-only proxy for
that decision, and is consistent with the classic robotics rigid-grasp
constraint (a part's pose is fixed relative to the end-effector while grasped
with no sliding)."""
import numpy as np


def _close_gaps(mask, max_gap):
    """Fill runs of False shorter than `max_gap` that are flanked by True on
    both sides — bridges brief contact dropouts (a stray hand-vertex frame
    pushing the min distance just over threshold) so a continuously held object
    reads as one long span rather than many fragments."""
    mask = np.asarray(mask, bool).copy()
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            j = i
            while j < n and not mask[j]:
                j += 1
            if i > 0 and j < n and (j - i) < max_gap:
                mask[i:j] = True
            i = j
        else:
            i += 1
    return mask


def _min_run(mask, min_run):
    """Keep only contiguous True runs of length >= `min_run`; drop shorter ones
    (flicker rejection)."""
    mask = np.asarray(mask, bool)
    out = np.zeros_like(mask)
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_run:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def grasp_mask_contact(hand_verts, obj_points_world, d_contact=0.03, win=5,
                       min_run=4):
    """Contact-based stable-grasp mask.

    hand_verts:        [T, Nh, 3] hand mesh vertices per frame (world frame).
    obj_points_world:  [T, No, 3] posed object surface points per frame
                       (world frame; subsample to ~500 upstream — this does a
                       KD-tree nearest-neighbour query per frame).
    d_contact:         contact threshold in metres (default 0.03 = 3 cm).
    win:               max False-gap length bridged (contact-dropout closing).
    min_run:           minimum contiguous grasp-span length kept (flicker).

    Returns bool[T]: True where the hand is in contact with the object over a
    span of at least `min_run` frames. Deterministic (no randomness)."""
    from scipy.spatial import cKDTree

    T = len(hand_verts)
    dmin = np.full(T, np.inf)
    for t in range(T):
        hv = np.asarray(hand_verts[t], np.float64)
        ov = np.asarray(obj_points_world[t], np.float64)
        if hv.size == 0 or ov.size == 0:
            continue
        d, _ = cKDTree(ov).query(hv, workers=-1)   # nearest obj pt per hand vtx
        dmin[t] = float(d.min())
    raw = dmin < d_contact
    return _min_run(_close_gaps(raw, win), min_run)


def grasp_spans(mask):
    """(n_frames, n_spans) for a bool grasp mask — n_spans = number of
    contiguous True runs. Small helper for logging that the detector engaged."""
    mask = np.asarray(mask, bool)
    n_frames = int(mask.sum())
    padded = np.concatenate([[0], mask.astype(int), [0]])
    n_spans = int((np.diff(padded) == 1).sum())
    return n_frames, n_spans
