"""Contact-based stable-grasp detector
(hoi_recon.grasp_segments.grasp_mask_contact) and the consecutive-pair
selection it feeds into object_icp._joint_refine's rigidity term.

v2 detection is CONTACT-based (min hand-vertex-to-object-surface distance),
not velocity-based: a relative hand-to-object quantity, immune to the
absolute-trajectory ICP jitter that made the old velocity detector mis-fire,
and it includes in-hand rotation (no hand-speed gate)."""
import numpy as np
import sys

sys.path.insert(0, "/workspace/code/hoi_recon/render_and_compare")
from hoi_recon.grasp_segments import grasp_mask_contact, grasp_spans

# small fixed point clouds for a hand and an object; per-frame centres below
_HOFF = np.array([[0., 0., 0.], [0.01, 0., 0.],
                  [0., 0.01, 0.], [0., 0., 0.01]])      # 4 "hand vertices"
_OOFF = np.array([[0., 0., 0.], [0.005, 0., 0.], [0., 0.005, 0.]])  # 3 obj pts


def _centres(T):
    """A moving trajectory (hand pivots/translates through space)."""
    s = np.linspace(0, 1, T)[:, None]
    return s * np.array([1.0, 0.5, 0.0]) + np.array([0.0, 0.0, 1.0])


def test_contact_span_detected():
    """Hand and object move TOGETHER within contact distance (2 cm offset <
    3 cm) for the whole clip -> flagged over the contiguous span."""
    T = 40
    c = _centres(T)
    hand = c[:, None, :] + _HOFF[None]                          # (T,4,3)
    obj = c[:, None, :] + np.array([0.02, 0.0, 0.0]) + _OOFF[None]
    m = grasp_mask_contact(hand, obj)
    assert m.all()
    assert grasp_spans(m) == (T, 1)


def test_independent_far_apart_rejected():
    """Object held 1 m away from the hand the whole clip -> never grasped."""
    T = 40
    c = _centres(T)
    hand = c[:, None, :] + _HOFF[None]
    obj = c[:, None, :] + np.array([1.0, 0.0, 0.0]) + _OOFF[None]
    m = grasp_mask_contact(hand, obj)
    assert not m.any()


def test_brief_contact_flicker_rejected():
    """A 2-frame brush (< min_run=4) must NOT count as a grasp span."""
    T = 40
    c = _centres(T)
    hand = c[:, None, :] + _HOFF[None]
    obj = c[:, None, :] + np.array([1.0, 0.0, 0.0]) + _OOFF[None]   # far
    obj = obj.copy()
    for f in (20, 21):                                             # brief contact
        obj[f] = c[f][None, :] + _OOFF
    m = grasp_mask_contact(hand, obj)                             # min_run=4
    assert not m.any()


def test_min_run_boundary_span_kept():
    """A contiguous run of exactly min_run (4) frames IS kept — the flicker
    filter's boundary."""
    T = 40
    c = _centres(T)
    hand = c[:, None, :] + _HOFF[None]
    obj = c[:, None, :] + np.array([1.0, 0.0, 0.0]) + _OOFF[None]   # far
    obj = obj.copy()
    for f in range(20, 24):                                        # 4-frame grasp
        obj[f] = c[f][None, :] + _OOFF
    m = grasp_mask_contact(hand, obj)
    assert m[20:24].all()
    assert m.sum() == 4
    assert grasp_spans(m) == (4, 1)


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
