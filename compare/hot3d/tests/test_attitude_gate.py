"""Spread-gate for the attitude search (object_icp._attitude_apply).

Calibrated on the mesh-controlled 6-clip HOT3D A/B: apply the chroma-scored
azimuth correction only for a non-identity winner whose chroma spread clears
SPREAD_MIN (bottle 31 LAB → apply; masher 10 / uniform objects 1-2 → keep the
depth-ICP attitude)."""
import sys

sys.path.insert(0, "/workspace/code/hoi_recon/render_and_compare")
from hoi_recon.object_icp import _attitude_apply, SPREAD_MIN


def test_strong_spread_nonidentity_applies():
    # bottle: spread 31 LAB, a non-identity winner
    assert _attitude_apply(best_is_identity=False, spread=31.4)


def test_moderate_spread_below_threshold_kept_identity():
    # masher: spread 9.9 LAB corrected but hurt chamfer — must be rejected
    assert not _attitude_apply(best_is_identity=False, spread=9.9)


def test_noise_spread_kept_identity():
    # vase/mug/cube: spread ~1-2 LAB, non-discriminative
    assert not _attitude_apply(best_is_identity=False, spread=1.6)


def test_identity_winner_never_applies():
    # even huge spread: if the best hypothesis IS identity, nothing to apply
    assert not _attitude_apply(best_is_identity=True, spread=99.0)


def test_threshold_is_eighteen():
    assert SPREAD_MIN == 18.0
    assert not _attitude_apply(False, SPREAD_MIN - 0.01)
    assert _attitude_apply(False, SPREAD_MIN)
