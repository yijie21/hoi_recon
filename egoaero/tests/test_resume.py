# egoaero/tests/test_resume.py
"""Regression test: stage6 resume path must not crash (FIX 1 — z_norm threading).

On a resumed / partial run stage6 loads stage5 output from disk and reconstructs
finger_idx from JSON-serialized lists.  Before the fix, z_norm was cast to int
(corrupting float data) causing wrong fingertip-pad selection and potential errors
downstream.  After the fix z_norm is correctly restored as a float array.
"""
from egoaero import config
from egoaero.pipeline import RunContext
from egoaero.stages import (
    stage0_ego_io, stage2_track, stage3_mesh,
    stage4_hand, stage5_ego_comp, stage6_contact,
)


def test_stage6_resume_from_disk(tmp_path):
    """Full pipeline -> save -> fresh RunContext -> run only stage6 (resume path)."""
    cfg = config.load_config(overrides={"num_frames": 16})

    # Full run: produce and persist stage0..stage5 outputs
    ctx1 = RunContext(cfg, str(tmp_path))
    stage0_ego_io.run(ctx1).save(ctx1.stage_dir("stage0_ego_io"))
    stage2_track.run(ctx1).save(ctx1.stage_dir("stage2_track"))
    stage3_mesh.run(ctx1).save(ctx1.stage_dir("stage3_mesh"))
    stage4_hand.run(ctx1).save(ctx1.stage_dir("stage4_hand"))
    stage5_ego_comp.run(ctx1).save(ctx1.stage_dir("stage5_ego_comp"))

    # Simulate resume: fresh RunContext pointing at the same run_dir, run ONLY stage6
    ctx2 = RunContext(cfg, str(tmp_path))
    b = stage6_contact.run(ctx2)   # must NOT raise AttributeError / TypeError

    assert "contact_mask" in b.arrays, "resume path must produce a contact_mask bundle"
    assert b["contact_mask"].shape[0] == 16
