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
from hoi_recon.object_icp import (_grasp_rigidity_pairs, _rot_angle_deg,
                                  ROT_GATE_DEG)


def _rotz(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])

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


def test_rot_angle_deg():
    """Geodesic angle of a pure z-rotation matches its degree amount."""
    for d in (2.5, 4.0, 6.7, 30.0):
        assert abs(float(_rot_angle_deg(_rotz(d))) - d) < 1e-4
    # batched
    ang = _rot_angle_deg(np.stack([_rotz(3.0), _rotz(5.0)]))
    assert np.allclose(ang, [3.0, 5.0], atol=1e-4)


def test_speed_gated_pairs_keep_only_fast_rotation():
    """v3 speed gate: given a grasp span covering BOTH a slow-hold segment
    (3 deg/fr wrist rotation, depth already tracks) and a fast in-hand-rotation
    segment (6 deg/fr, depth azimuth-blind), only the fast-segment consecutive
    pairs survive _grasp_rigidity_pairs (>= ROT_GATE_DEG = 4 deg/fr)."""
    T = 20
    # cumulative wrist orientation: frames 0-9 rotate 3 deg/fr, 10-19 rotate 6
    speeds = np.array([3.0] * 10 + [6.0] * 10)
    ang = np.concatenate([[0.0], np.cumsum(speeds)[:-1]])   # per-frame absolute
    wrist_R = np.stack([_rotz(a) for a in ang])
    grasp = np.ones(T, bool)                                # grasped throughout

    gi = _grasp_rigidity_pairs(grasp, wrist_R)
    # pair i has wrist delta angle = speeds[i]: i=0..9 -> 3 deg/fr (dropped),
    # i=10..18 -> 6 deg/fr (kept). Pairs run 0..T-2 = 0..18.
    assert gi.min() >= 10               # nothing from the slow 3-deg/fr stretch
    assert set(gi.tolist()) == set(range(10, 19))
    # every surviving pair is genuinely fast
    dRw = wrist_R[gi + 1] @ np.swapaxes(wrist_R[gi], -1, -2)
    assert (_rot_angle_deg(dRw) >= ROT_GATE_DEG).all()


def test_speed_gate_all_slow_selects_nothing():
    """A purely slow-hold grasp (all pairs below the gate) selects no pairs —
    the term then contributes nothing (needs >=2)."""
    T = 12
    ang = np.arange(T) * 2.5                                # 2.5 deg/fr, all slow
    wrist_R = np.stack([_rotz(a) for a in ang])
    gi = _grasp_rigidity_pairs(np.ones(T, bool), wrist_R)
    assert len(gi) == 0


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
