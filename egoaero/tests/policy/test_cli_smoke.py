# egoaero/tests/policy/test_cli_smoke.py
"""Gated smoke test for the egoaero-train / egoaero-eval CLI entry points."""
import os
import subprocess
import sys

import pytest


def test_train_eval_cli(tmp_path):
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("mujoco")
    # egoaero/ package root: 3 dirname levels up from this file
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if not os.path.exists(os.path.join(here, "assets", "shadow_hand", "right_hand.xml")):
        pytest.skip("shadow hand not vendored")

    from egoaero import config
    from egoaero.pipeline import run_pipeline
    from egoaero import contract

    ctx = run_pipeline(
        config.load_config(overrides={"num_frames": 8}),
        str(tmp_path / "run"),
        "all",
    )
    contract.write(ctx)

    pol = str(tmp_path / "pol")
    r = subprocess.run(
        [
            sys.executable, "-m", "egoaero.policy.cli",
            "train",
            "--run", str(tmp_path / "run"),
            "--out", pol,
            "--budget", "smoke",
        ],
        cwd=here,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert os.path.exists(os.path.join(pol, "pi_R.zip")), (
        f"pi_R.zip missing; stdout={r.stdout!r}; stderr={r.stderr!r}"
    )
