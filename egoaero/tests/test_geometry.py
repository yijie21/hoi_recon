import numpy as np
from egoaero.core import geometry as g

def test_se3_inv_roundtrip():
    T = g.se3(g.rotvec_to_R(np.array([0.1, -0.2, 0.3])), np.array([1., 2., 3.]))
    assert np.allclose(g.se3_inv(T) @ T, np.eye(4), atol=1e-9)

def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(50, 3))
    s, R, t = 1.7, g.rotvec_to_R(np.array([0.2, 0.1, -0.3])), np.array([0.5, -1., 2.])
    dst = (s * (R @ src.T).T) + t
    s2, R2, t2 = g.umeyama(src, dst, with_scale=True)
    assert abs(s2 - s) < 1e-6 and np.allclose(R2, R, atol=1e-6)

def test_geodesic_deg_zero_and_90():
    I = np.eye(3)
    Rz = g.rotvec_to_R(np.array([0, 0, np.pi/2]))
    assert g.geodesic_deg(I, I) < 1e-6
    assert abs(g.geodesic_deg(I, Rz) - 90.0) < 1e-4

def test_se3_log_exp_roundtrip():
    T = g.se3(g.rotvec_to_R(np.array([0.3, 0.1, -0.2])), np.array([0.4, -0.1, 0.2]))
    assert np.allclose(g.se3_exp(g.se3_log(T)), T, atol=1e-8)
