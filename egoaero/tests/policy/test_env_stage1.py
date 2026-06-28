# egoaero/tests/policy/test_env_stage1.py
"""Tests for policy/env.py — StageIEnv (gymnasium + mujoco, gated)."""
import os
import numpy as np
import pytest


def _setup(tmp_path):
    """Build and return a Task from a mock pipeline run."""
    pytest.importorskip("mujoco")
    pytest.importorskip("gymnasium")
    # test file lives at egoaero/tests/policy/; here = egoaero/tests
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hand_xml = os.path.join(os.path.dirname(here), "assets", "shadow_hand", "right_hand.xml")
    if not os.path.exists(hand_xml):
        pytest.skip("shadow hand not vendored")

    from egoaero import config
    from egoaero.pipeline import run_pipeline
    from egoaero import contract

    ctx = run_pipeline(
        config.load_config(overrides={"num_frames": 10}),
        str(tmp_path / "run"),
        "all",
    )
    contract.write(ctx)

    from egoaero.policy.task import build_task
    return build_task(str(tmp_path / "run"), hand_xml, config.load_config())


def test_stage1_env_api(tmp_path):
    """reset() returns correct shape; step() returns finite reward + same-shape obs."""
    from egoaero.policy.env import StageIEnv

    env = StageIEnv(_setup(tmp_path))
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape, (
        f"reset obs shape {obs.shape} != observation_space.shape {env.observation_space.shape}"
    )
    assert obs.dtype == np.float32, f"obs dtype should be float32, got {obs.dtype}"

    a = env.action_space.sample()
    obs2, rew, term, trunc, info = env.step(a)
    assert np.isfinite(rew), f"reward not finite: {rew}"
    assert obs2.shape == obs.shape, f"step obs shape {obs2.shape} != {obs.shape}"
    assert obs2.dtype == np.float32
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)


def test_stage1_env_full_episode(tmp_path):
    """Run a full episode; all rewards finite; episode terminates by truncation."""
    from egoaero.policy.env import StageIEnv

    env = StageIEnv(_setup(tmp_path))
    obs, _ = env.reset(seed=1)
    done = False
    rewards = []
    steps = 0
    while not done:
        a = env.action_space.sample()
        obs, rew, term, trunc, _ = env.step(a)
        rewards.append(rew)
        done = term or trunc
        steps += 1
        assert steps <= 200, "episode did not terminate within 200 steps"

    assert all(np.isfinite(r) for r in rewards), "some rewards are not finite"
    assert trunc, "episode should end by truncation (trunc=True), not by termination"
    assert not term


def test_stage1_check_env(tmp_path):
    """SB3 check_env passes (no warnings treated as errors)."""
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.env_checker import check_env
    from egoaero.policy.env import StageIEnv

    check_env(StageIEnv(_setup(tmp_path)), warn=True, skip_render_check=True)
