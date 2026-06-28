# egoaero/tests/policy/test_evaluate.py
"""Smoke test for policy/evaluate.py — rollout, evaluate, ablation (gated)."""
import pytest


def _policy_cfg():
    import os
    import yaml
    # test file: egoaero/tests/policy/test_evaluate.py
    # 3 dirname calls → egoaero/
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cfg_path = os.path.join(here, "egoaero", "configs", "policy.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def test_evaluate_returns_metrics(tmp_path):
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("torch")
    from tests.policy.test_env_stage1 import _setup
    from egoaero.policy.train import train_two_stage
    from egoaero.policy.evaluate import evaluate

    task = _setup(tmp_path)
    pi_I, pi_R = train_two_stage(task, _policy_cfg(), budget="smoke", seed=0)
    m = evaluate(task, pi_I, pi_R, seeds=(0,))
    assert set(["Er", "Et", "Ej", "Eft", "SR"]).issubset(m)
    assert 0.0 <= m["SR"] <= 1.0


def test_evaluate_none_policies(tmp_path):
    """Regression: evaluate(task, None, None) should not raise action dim mismatch."""
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("torch")
    pytest.importorskip("mujoco")
    from tests.policy.test_env_stage1 import _setup
    from egoaero.policy.evaluate import evaluate

    task = _setup(tmp_path)
    # Call evaluate with both policies None (zero actions)
    # This exercises the pi_I is None fallback which should use len(task.finger_act_ids)
    m = evaluate(task, None, None, seeds=(0,))
    assert set(["Er", "Et", "Ej", "Eft", "SR"]).issubset(m)
    assert 0.0 <= m["SR"] <= 1.0
