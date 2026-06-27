from egoaero import config
from egoaero.pipeline import RunContext
from egoaero.stages import stage0_ego_io, stage4_hand


def test_depth_correction_reduces_bias(tmp_path):
    ctx = RunContext(config.load_config(overrides={"num_frames": 14}), str(tmp_path))
    stage0_ego_io.run(ctx).save(ctx.stage_dir("stage0_ego_io"))
    b = stage4_hand.run(ctx)
    assert b.meta["transl_err_after_mm"] < b.meta["transl_err_before_mm"]
