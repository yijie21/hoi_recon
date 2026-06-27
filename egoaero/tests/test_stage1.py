from egoaero import config
from egoaero.pipeline import RunContext
from egoaero.stages import stage0_ego_io, stage1_semantic

def test_seed_frame_and_passthrough(tmp_path):
    ctx = RunContext(config.load_config(overrides={"num_frames": 16}), str(tmp_path))
    stage0_ego_io.run(ctx).save(ctx.stage_dir("stage0_ego_io"))
    b = stage1_semantic.run(ctx)
    assert 0 <= b.meta["seed_frame"] < 16
    assert b.meta["target_object"] == "object"
    assert b["obj_mask"].shape[0] == 16
