import numpy as np
from egoaero.policy import metrics as M


def test_translation_error_cm():
    p = np.zeros((3, 3))
    pref = np.zeros((3, 3))
    pref[:, 0] = 0.01   # 1cm off
    assert abs(M.object_translation_error(p, pref) - 1.0) < 1e-9       # cm


def test_rotation_error_deg():
    I = np.tile(np.eye(3), (2, 1, 1))
    Rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    seq = np.stack([Rz, Rz])
    assert abs(M.object_rotation_error(seq, I) - 90.0) < 1e-3


def test_joint_and_fingertip_cm():
    a = np.zeros((2, 5, 3))
    b = a.copy()
    b[:, :, 0] = 0.02          # 2cm
    assert abs(M.mean_joint_error(a, b) - 2.0) < 1e-9
    assert abs(M.mean_fingertip_error(a, b) - 2.0) < 1e-9


def test_success_thresholds_and_rate():
    assert M.success(10.0, 1.0, 2.0, 1.0) is True
    assert M.success(40.0, 1.0, 2.0, 1.0) is False     # Er over 30
    rows = [(10, 1, 2, 1), (40, 1, 2, 1)]
    assert abs(M.success_rate(rows) - 0.5) < 1e-9
