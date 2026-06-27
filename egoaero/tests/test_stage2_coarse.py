import numpy as np
from egoaero.core.geometry import se3, rotvec_to_R, transform_points
from egoaero.stages.stage2_track import coarse_pose

def test_ransac_recovers_rigid_with_outliers():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(200, 3)) * 0.05
    T = se3(rotvec_to_R(np.array([0.05, -0.1, 0.07])), np.array([0.01, 0.0, 0.02]))
    dst = transform_points(src, T)
    dst[:40] += rng.normal(size=(40, 3)) * 0.5            # 20% gross outliers
    Test, inl = coarse_pose(src, dst, ransac_thresh=0.01, iters=200, rng=rng)
    assert np.allclose(Test, T, atol=1e-2)
    assert inl.sum() >= 150
