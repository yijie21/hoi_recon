"""Stable-grasp detector (hoi_recon.grasp_segments.stable_grasp_mask) and the
consecutive-pair selection it feeds into object_icp._joint_refine's rigidity
term."""
import numpy as np
import sys

sys.path.insert(0, "/workspace/code/hoi_recon/render_and_compare")
from hoi_recon.grasp_segments import stable_grasp_mask


def test_rigid_motion_detected():
    t = np.linspace(0, 1, 60)[:, None]
    wrist = np.concatenate([t, t * 0.5, 1.0 + 0 * t], 1)   # moving hand
    obj = wrist + np.array([0.03, 0.01, 0.02])             # rigid offset
    m = stable_grasp_mask(wrist, obj)
    assert m[10:50].all()


def test_independent_motion_rejected():
    rng = np.random.default_rng(0)
    wrist = rng.normal(0, 0.05, (60, 3)).cumsum(0)
    obj = np.zeros((60, 3))                                # static object
    m = stable_grasp_mask(wrist, obj)
    assert not m[10:50].any()


def test_static_static_not_grasped():
    """The hand-motion gate: `& (ms > v_rel_max)`. When BOTH the object and
    the hand are perfectly still, relative velocity is ~0 (which passes the
    vs<v_rel_max stability test), but the hand isn't moving — so ms<v_rel_max
    must VETO it. This is the case that separates 'rigidly grasped AND moving'
    from 'everything just sitting static on a table'. Without the ms gate this
    would be flagged all-True; with it, all-False.

    (Guards the gate directly: the two motion tests above still pass if the
    `& (ms > v_rel_max)` clause is stripped — this one does not.)"""
    wrist = np.zeros((60, 3))                              # hand perfectly still
    obj = np.full((60, 3), [0.10, -0.05, 0.30])           # object still, offset
    m = stable_grasp_mask(wrist, obj)
    assert not m.any()

    # also with a tiny constant offset of zero motion at a nonzero location for
    # the hand: still no motion anywhere -> never grasped
    wrist2 = np.full((60, 3), [1.0, 2.0, 3.0])
    m2 = stable_grasp_mask(wrist2, wrist2 + np.array([0.02, 0.0, 0.0]))
    assert not m2.any()


def test_consecutive_grasp_pair_selection():
    """_joint_refine builds its rigidity-term index set as
    gi = where(mask[:-1] & mask[1:]) — consecutive TRUE pairs only. A single
    isolated grasp frame (no adjacent grasp frame on either side) must
    contribute zero pairs, and a solid run of length n must contribute
    exactly n-1 pairs."""
    mask = np.array([False, True, False, True, True, True, False])
    gi = np.where(mask[:-1] & mask[1:])[0]
    assert list(gi) == [3, 4]                 # pairs (3,4) and (4,5)
    assert len(gi) == 2

    isolated = np.array([False, True, False])
    gi2 = np.where(isolated[:-1] & isolated[1:])[0]
    assert len(gi2) == 0
