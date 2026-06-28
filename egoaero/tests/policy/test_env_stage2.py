# egoaero/tests/policy/test_env_stage2.py
"""Tests for policy/env.py — StageIIEnv (residual RL, gated)."""
import numpy as np
import pytest


def test_stage2_residual_and_api(tmp_path):
    """reset returns 77-dim obs; step gives finite reward; check_env passes."""
    pytest.importorskip("mujoco")
    pytest.importorskip("gymnasium")
    sb3 = pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.env_checker import check_env
    from tests.policy.test_env_stage1 import _setup
    from egoaero.policy.env import StageIIEnv

    task = _setup(tmp_path)
    # No-op Stage-I policy: returns zeros over the 18 finger actuators
    pi_I = lambda obs: np.zeros(len(task.finger_act_ids), np.float32)

    env = StageIIEnv(task, pi_I)
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape, (
        f"reset obs shape {obs.shape} != space shape {env.observation_space.shape}"
    )
    assert obs.dtype == np.float32, f"obs dtype {obs.dtype} != float32"

    o2, r, term, trunc, _ = env.step(env.action_space.sample())
    assert o2.shape == obs.shape, f"step obs shape mismatch"
    assert o2.dtype == np.float32
    assert np.isfinite(r), f"reward not finite: {r}"
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)

    check_env(env, warn=True, skip_render_check=True)
