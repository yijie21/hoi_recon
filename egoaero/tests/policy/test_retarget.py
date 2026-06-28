import numpy as np
from egoaero.policy import retarget as RT


def toy_fk(q):
    # 1 "fingertip": a 2-joint planar arm in xy, link lengths 1,1
    x = np.cos(q[0]) + np.cos(q[0] + q[1])
    y = np.sin(q[0]) + np.sin(q[0] + q[1])
    return np.array([[x, y, 0.0]])


def test_solve_ik_reaches_target():
    target = np.array([[1.0, 1.0, 0.0]])      # reachable
    q = RT.solve_ik(np.array([0.1, 0.1]), toy_fk, target, damping=0.05, iters=200)
    err = np.linalg.norm(toy_fk(q) - target)
    assert err < 1e-2


def test_retarget_sequence_shape():
    seq = np.array([[[1.0, 1.0, 0.0]], [[0.5, 1.2, 0.0]]])   # T=2, K=1
    qs = RT.retarget_sequence(seq, toy_fk, np.array([0.1, 0.1]), n_q=2,
                              damping=0.05, iters=100)
    assert qs.shape == (2, 2)
