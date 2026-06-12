import numpy as np
from hoi_recon.choir_fine import metrics


def test_contact_gap_zero_when_touching():
    # hand point coincident with a surface point on the single contact frame
    hand_c = np.zeros((1, 1, 3))                       # (T=1, Nh=1, 3)
    surf = np.zeros((1, 1, 3))                         # (T=1, No=1, 3)
    cp = np.array([True])
    assert metrics.contact_gap(hand_c, surf, cp) == 0.0


def test_contact_gap_is_distance():
    hand_c = np.zeros((1, 1, 3))
    surf = np.full((1, 1, 3), 0.0); surf[0, 0, 2] = 0.05    # 5cm away
    assert metrics.contact_gap(hand_c, surf, np.array([True])) == \
        __import__("pytest").approx(0.05)


def test_contact_gap_nan_when_no_contact_frames():
    hand_c = np.zeros((2, 1, 3)); surf = np.zeros((2, 1, 3))
    assert np.isnan(metrics.contact_gap(hand_c, surf, np.array([False, False])))


def test_penetration_positive_inside():
    # one hand vertex 1cm inside a surface whose outward normal is +z, at z=0
    hand = np.array([[[0.0, 0.0, -0.01]]])             # (T=1, Nh=1, 3), below surface
    surf = np.array([[[0.0, 0.0, 0.0]]])               # nearest surface point
    nrm = np.array([[[0.0, 0.0, 1.0]]])                # outward normal +z
    pen = metrics.penetration_depth(hand, surf, nrm)
    assert pen == __import__("pytest").approx(0.01, abs=1e-6)


def test_penetration_zero_outside():
    hand = np.array([[[0.0, 0.0, 0.02]]])              # above surface (outside)
    surf = np.array([[[0.0, 0.0, 0.0]]])
    nrm = np.array([[[0.0, 0.0, 1.0]]])
    assert metrics.penetration_depth(hand, surf, nrm) == 0.0
