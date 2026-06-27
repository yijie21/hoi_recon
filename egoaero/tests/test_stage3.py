import numpy as np
from egoaero import config
from egoaero.pipeline import RunContext
from egoaero.stages import stage0_ego_io, stage2_track, stage3_mesh


def test_mesh_alignment_recovered(tmp_path):
    ctx = RunContext(config.load_config(overrides={"num_frames": 12}), str(tmp_path))
    stage0_ego_io.run(ctx).save(ctx.stage_dir("stage0_ego_io"))
    stage2_track.run(ctx).save(ctx.stage_dir("stage2_track"))
    b = stage3_mesh.run(ctx)
    assert b["obj_verts"].shape[1] == 3
    assert b.meta["align_residual_m"] < 1e-3      # recovers the injected sam-mesh transform
