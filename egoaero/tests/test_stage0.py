from egoaero import config
from egoaero.bundle import Bundle
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

def test_stage0_meta_roundtrip(tmp_path):
    b = stage0_ego_io.run(_ctx(tmp_path))
    b.save(str(tmp_path / "s0"))
    b2 = Bundle.load(str(tmp_path / "s0"))
    assert b2.meta["T"] == 16
    assert isinstance(b2.meta["stage_labels"], list)
    assert isinstance(b2.meta["finger_idx"], dict)
    assert isinstance(next(iter(b2.meta["finger_idx"].values())), list)
