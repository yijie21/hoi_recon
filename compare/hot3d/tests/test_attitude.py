"""Pure-function coverage for the T2 anchor attitude search (object_icp).

Only the deterministic, GPU-free helpers are exercised here: the canonical-axis
rotation builder and the azimuth-hypothesis grid. The in-loop photometric term
and full _score_hypothesis need real frames/masks/depth and are screened on the
bench, not unit-tested."""
import numpy as np, sys

sys.path.insert(0, "/workspace/code/hoi_recon/render_and_compare")
from hoi_recon.object_icp import _rotz, _rot_axis, _attitude_hypotheses


def _is_rotation(R):
    return (R.shape == (3, 3)
            and np.allclose(R @ R.T, np.eye(3), atol=1e-9)
            and abs(np.linalg.det(R) - 1.0) < 1e-9)


def test_rotz():
    assert np.allclose(_rotz(0.0), np.eye(3))
    assert np.allclose(_rotz(2 * np.pi), np.eye(3), atol=1e-9)
    # +90deg about z sends +x -> +y
    assert np.allclose(_rotz(np.pi / 2) @ np.array([1.0, 0, 0]),
                       np.array([0.0, 1, 0]), atol=1e-9)
    assert _is_rotation(_rotz(0.7))
    # _rotz is exactly the z-axis (axis 2) case of _rot_axis
    assert np.allclose(_rotz(0.7), _rot_axis(2, 0.7))


def test_rot_axis_valid_and_axis_fixed():
    for axis in range(3):
        R = _rot_axis(axis, 1.234)
        assert _is_rotation(R)
        e = np.eye(3)[axis]
        assert np.allclose(R @ e, e, atol=1e-9)      # rotation axis is fixed


def test_attitude_hypotheses_contains_identity_and_valid():
    hyps = _attitude_hypotheses(12)
    # 1 identity + 12//3 non-zero rotations about each of the 3 axes
    assert len(hyps) == 1 + 3 * (12 // 3)
    assert any(np.allclose(R, np.eye(3)) for R in hyps)   # identity present
    assert all(_is_rotation(R) for R in hyps)             # all valid SO(3)


def test_attitude_hypotheses_counts_scale_with_n():
    for n in (9, 12, 24):
        assert len(_attitude_hypotheses(n)) == 1 + 3 * (n // 3)
    # degenerate small n still yields identity + one per axis (per>=1)
    assert len(_attitude_hypotheses(1)) == 1 + 3
