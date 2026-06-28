# egoaero/tests/policy/test_train.py
"""Smoke test for two-stage SB3 PPO training driver (gated)."""
import os
import pytest


def _policy_cfg():
    import os
    import yaml
    # test file: egoaero/tests/policy/test_train.py
    # here (3 dirname calls) = egoaero/
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cfg_path = os.path.join(here, "egoaero", "configs", "policy.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def test_two_stage_smoke_trains_and_saves(tmp_path):
    """Smoke: two-stage PPO completes at budget='smoke'; both zips saved."""
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("torch")
    from tests.policy.test_env_stage1 import _setup
    from egoaero.policy.train import train_two_stage

    task = _setup(tmp_path)
    out = str(tmp_path / "pol")
    pi_I, pi_R = train_two_stage(task, _policy_cfg(), budget="smoke", out_dir=out, seed=0)

    assert os.path.exists(os.path.join(out, "pi_I.zip")), "pi_I.zip not found"
    assert os.path.exists(os.path.join(out, "pi_R.zip")), "pi_R.zip not found"
