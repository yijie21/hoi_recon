# tests/test_choir_fine_configs.py
import pathlib
from hoi_recon.config import load_config
from hoi_recon.choir_fine import presets

_REPO = pathlib.Path(__file__).resolve().parent.parent


def test_choir_faithful_config_names_locked_preset():
    cfg = load_config(str(_REPO / "configs/choir_faithful.yaml"))
    assert cfg.get("fine_preset") == "choir_faithful"
    w = presets.get_preset(cfg["fine_preset"])
    assert w["contact"] == 1000.0 and w["hand_sil"] == 0.0   # faithful: no hand-sil


def test_combined_v2_config_enables_toggles():
    cfg = load_config(str(_REPO / "configs/combined_v2.yaml"))
    assert cfg.get("fine_preset") == "combined_v2"
    w = presets.get_preset(cfg["fine_preset"])
    assert w["hand_sil"] > 0.0                                # our improvement on


def test_both_configs_are_real_mode_with_a_resolvable_preset():
    for name in ("configs/choir_faithful.yaml", "configs/combined_v2.yaml"):
        cfg = load_config(str(_REPO / name))
        assert cfg.mock is False
        presets.get_preset(cfg["fine_preset"])               # must not raise
