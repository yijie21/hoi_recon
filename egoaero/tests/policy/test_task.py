# egoaero/tests/policy/test_task.py
"""Tests for policy/task.py — MuJoCo scene builder + reference loader."""
import os
import numpy as np
import pytest

from egoaero import config
from egoaero.pipeline import run_pipeline
from egoaero import contract


def _recon(tmp_path):
    """Run the full mock pipeline + contract write; return the run dir path."""
    cfg = config.load_config(overrides={"num_frames": 12})
    ctx = run_pipeline(cfg, str(tmp_path / "run"), stages="all")
    contract.write(ctx)
    return str(tmp_path / "run")


# ---------------------------------------------------------------------------
# test_load_reference: pure numpy — no mujoco required
# ---------------------------------------------------------------------------

def test_load_reference(tmp_path):
    run = _recon(tmp_path)
    from egoaero.policy.task import load_reference
    ref = load_reference(run)
    assert ref["wrist_pos"].shape == (12, 3)
    assert ref["fingertips_h"].shape == (12, 5, 3)
    assert ref["obj_R"].shape == (12, 3, 3)
    assert ref["obj_pos"].shape == (12, 3)
    assert isinstance(ref["contact_active"], list)
    assert len(ref["contact_active"]) == 12
    assert isinstance(ref["contact_active"][0], list)
    assert ref["T"] == 12
    assert os.path.exists(ref["mesh_path"])


# ---------------------------------------------------------------------------
# test_build_task: gated on mujoco being installed + shadow hand asset present
# ---------------------------------------------------------------------------

def test_build_task(tmp_path):
    pytest.importorskip("mujoco")
    # The test file is at egoaero/tests/policy/test_task.py
    # here = egoaero/tests  (two dirnames up from this file)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hand_xml = os.path.join(os.path.dirname(here), "assets", "shadow_hand", "right_hand.xml")
    if not os.path.exists(hand_xml):
        pytest.skip("shadow hand not vendored")

    run = _recon(tmp_path)
    from egoaero.policy.task import build_task
    from egoaero.config import load_config
    import mujoco

    task = build_task(run, hand_xml, load_config().get("quality", {}))

    # Basic model validity
    assert task.model.nu > 0, "model must have actuators"
    assert task.ref["T"] == 12, "reference must have 12 frames"

    # Finger actuators (no WRJ): Shadow Hand has 20 total, 2 wrist → 18 finger
    assert len(task.finger_act_ids) == 18, (
        f"expected 18 finger actuators, got {len(task.finger_act_ids)}"
    )

    # All 5 fingertip body IDs must resolve on the composed model
    assert len(task.fingertip_body_ids) == 5
    for finger, bid in task.fingertip_body_ids.items():
        assert bid >= 0, f"fingertip body id for '{finger}' not found (bid={bid})"

    # Wrist mocap id is valid
    assert task.wrist_mocap_id >= 0

    # obj_qadr is a non-negative qpos index
    assert task.obj_qadr >= 0

    # Scene must step without error
    data = mujoco.MjData(task.model)
    mujoco.mj_step(task.model, data)  # raises if model is broken
