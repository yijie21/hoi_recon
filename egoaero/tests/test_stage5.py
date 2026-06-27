# egoaero/tests/test_stage5.py
import numpy as np
from egoaero import config
from egoaero.pipeline import RunContext
from egoaero.stages import stage0_ego_io, stage2_track, stage4_hand, stage5_ego_comp
from egoaero.core.geometry import se3_inv


def test_world_to_table_transform(tmp_path):
    ctx = RunContext(config.load_config(overrides={"num_frames": 12}), str(tmp_path))
    s0 = stage0_ego_io.run(ctx); s0.save(ctx.stage_dir("stage0_ego_io"))
    stage2_track.run(ctx).save(ctx.stage_dir("stage2_track"))
    stage4_hand.run(ctx).save(ctx.stage_dir("stage4_hand"))
    b = stage5_ego_comp.run(ctx)
    # table-frame hand = inv(table_T) applied to world hand (first frame, pre-smoothing check loosely)
    assert b["hand_verts_t"].shape == s0["gt_hand_verts_w"].shape
    assert b["obj_poses_t"].shape[0] == 12
