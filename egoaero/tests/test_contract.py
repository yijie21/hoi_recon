# egoaero/tests/test_contract.py
from egoaero import config
from egoaero.pipeline import run_pipeline
from egoaero import contract


def test_contract_written_and_valid(tmp_path):
    cfg = config.load_config(overrides={"num_frames": 16})
    ctx = run_pipeline(cfg, str(tmp_path / "run"), stages="all")
    out = contract.write(ctx)
    assert contract.validate(str(tmp_path / "run"))
    assert "object_traj" in out and "hand_mano" in out


def test_contract_not_called_without_stage7(tmp_path):
    """Pipeline without stage7 should not auto-write contract."""
    cfg = config.load_config(overrides={"num_frames": 16})
    ctx = run_pipeline(cfg, str(tmp_path / "run"), stages="0-6")
    import os
    assert not os.path.exists(os.path.join(str(tmp_path / "run"), "contract"))


def test_validate_returns_false_when_missing(tmp_path):
    assert not contract.validate(str(tmp_path / "nonexistent"))
