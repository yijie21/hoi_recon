import numpy as np
from egoaero import config
from egoaero.pipeline import RunContext
from egoaero.stages import stage0_ego_io

def _ctx(tmp_path, **over):
    cfg = config.load_config(overrides={"num_frames": 16, **over})
    return RunContext(cfg, str(tmp_path))

def test_stage0_mock(tmp_path):
    b = stage0_ego_io.run(_ctx(tmp_path))
    assert b.meta["T"] == 16
    assert b["depth"].shape[0] == 16
    assert b["cam_traj"].shape == (16, 4, 4)
    assert b["obj_mask"].shape[0] == 16
