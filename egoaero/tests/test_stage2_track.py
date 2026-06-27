# egoaero/tests/test_stage2_track.py
import numpy as np
from egoaero import config
from egoaero.pipeline import RunContext
from egoaero.stages import stage0_ego_io, stage2_track


def test_pose_graph_reduces_drift(tmp_path):
    ctx = RunContext(config.load_config(overrides={"num_frames": 20, "seed": 0}), str(tmp_path))
    stage0_ego_io.run(ctx).save(ctx.stage_dir("stage0_ego_io"))
    b = stage2_track.run(ctx)
    assert b["obj_poses_w"].shape == (20, 4, 4)
    assert b.meta["track_err_deg_after"] < b.meta["track_err_deg_before"]
