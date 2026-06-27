import pytest
from hoi_recon.choir_fine import presets


def test_choir_faithful_weights_locked():
    """CHOIR arXiv:2605.20992 §7.3 weights — verbatim. This test is the anti-drift
    guard; do NOT change these numbers without changing the paper reference."""
    w = presets.CHOIR_FAITHFUL
    assert w["contact"] == 1000.0
    assert w["pen"] == 500.0
    assert w["sil"] == 500.0
    assert w["anc_2d"] == 0.5
    assert w["anc_anat"] == 30.0
    assert w["anc_pose_h"] == 100.0
    assert w["anc_pose_o"] == 100.0
    assert w["temp_pose_vel"] == 500.0
    assert w["temp_obj_vel"] == 500.0
    assert w["temp_wrist_anchor"] == 200.0
    assert w["temp_hand_tr_vel"] == 200.0
    assert w["temp_root_R_vel"] == 500.0
    assert w["temp_pose_acc"] == 1000.0
    assert w["temp_hand_tr_acc"] == 5000.0
    assert w["temp_root_R_acc"] == 3000.0
    assert w["hand_sil"] == 0.0          # CHOIR has no hand-silhouette term
    assert presets.LRS_FAITHFUL == {"object": 3e-4, "finger": 5e-4, "wrist": 5e-5}
    assert presets.ITERS_FAITHFUL == 800


def test_every_term_has_a_weight():
    """Every registered term name must have a weight in every preset (no silent gaps)."""
    for name in ("choir_faithful", "combined_v2"):
        w = presets.get_preset(name)
        assert set(w) >= set(presets.TERMS), set(presets.TERMS) - set(w)


def test_combined_v2_enables_our_toggles():
    """combined_v2 = faithful + our improvements on (hand silhouette here)."""
    w = presets.get_preset("combined_v2")
    assert w["hand_sil"] > 0.0           # T3 hand image registration on
    # faithful weights preserved
    assert w["contact"] == 1000.0


def test_get_preset_unknown_raises():
    with pytest.raises(KeyError):
        presets.get_preset("nope")
