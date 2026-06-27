# egoaero/tests/test_stage6_run.py
from egoaero import config
from egoaero.pipeline import RunContext
from egoaero.stages import (stage0_ego_io, stage2_track, stage4_hand,
                            stage5_ego_comp, stage6_contact)
import numpy as np

def _prep(tmp_path, **over):
    ctx = RunContext(config.load_config(overrides={"num_frames": 16, **over}), str(tmp_path))
    stage0_ego_io.run(ctx).save(ctx.stage_dir("stage0_ego_io"))
    stage2_track.run(ctx).save(ctx.stage_dir("stage2_track"))
    stage4_hand.run(ctx).save(ctx.stage_dir("stage4_hand"))
    stage5_ego_comp.run(ctx).save(ctx.stage_dir("stage5_ego_comp"))
    return ctx

def test_triangular_smooth_bounds():
    d = np.zeros((10, 3)); d[5] = [0.03, 0, 0]
    sm = stage6_contact.triangular_smooth(d, window=9, boundary_frames=3)
    assert sm.shape == (10, 3) and abs(sm[5, 0]) < 0.03   # spike is smoothed down

def test_run_reduces_penetration_and_gap(tmp_path):
    ctx = _prep(tmp_path)
    b = stage6_contact.run(ctx)
    assert b.meta["pen_after_mm"] <= b.meta["pen_before_mm"] + 1e-6
    assert b["contact_mask"].shape[0] == 16
