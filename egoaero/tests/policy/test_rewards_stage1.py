import numpy as np
from egoaero.policy import rewards as R


def test_r_wrist_perfect_is_one_and_decreasing():
    p = np.zeros(3); Rm = np.eye(3); v = np.zeros(3)
    assert abs(R.r_wrist(p, Rm, v, p, Rm, v, 40.0, 1.0, 1.0) - 1.0) < 1e-12
    worse = R.r_wrist(p + 0.05, Rm, v, p, Rm, v, 40.0, 1.0, 1.0)
    assert 0.0 < worse < 1.0


def test_r_finger_mean_of_exp():
    x = np.zeros((5, 3)); xh = np.zeros((5, 3))
    assert abs(R.r_finger(x, xh, 40.0) - 1.0) < 1e-12
    xh2 = xh.copy(); xh2[0] = [0.1, 0, 0]
    assert R.r_finger(x, xh2, 40.0) < 1.0


def test_r_smooth_and_stage1_weighting():
    a = np.zeros(4); s = R.r_smooth(a, a, np.zeros(4), np.zeros(4), 0.1, 0.001)
    assert abs(s - 1.0) < 1e-12
    assert abs(R.r_stage1(1.0, 1.0, 1.0, 1.0, 1.0, 0.1) - 2.1) < 1e-12
