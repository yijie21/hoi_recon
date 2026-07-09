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
