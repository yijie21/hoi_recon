import numpy as np
from egoaero.policy import rewards as R


def test_r_obj_perfect_is_one():
    p = np.zeros(3)
    Rm = np.eye(3)
    v = np.zeros(3)
    assert abs(R.r_obj(p, Rm, v, p, Rm, v, 40.0, 1.0, 1.0) - 1.0) < 1e-12


def test_r_contact_empty_active_is_zero():
    assert R.r_contact(np.zeros(5), np.zeros(5), [], 200.0, 1.0) == 0.0


def test_r_contact_rewards_close_and_forceful():
    dists = np.array([0.0, 0.2, 0.2, 0.2, 0.2])     # thumb touching
    forces = np.array([5.0, 0, 0, 0, 0])
    val = R.r_contact(dists, forces, [0], 200.0, 1.0)   # only thumb active
    assert 0.0 < val <= 1.0
    far = R.r_contact(np.array([0.2, 0, 0, 0, 0]), forces, [0], 200.0, 1.0)
    assert far < val                                   # farther -> lower


def test_r_res_and_stage2_weighting():
    assert abs(R.r_res(np.zeros(4), 1.0) - 1.0) < 1e-12
    assert abs(R.r_stage2(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1) - 3.1) < 1e-12
