from egoaero import config
from egoaero.pipeline import RunContext
from egoaero.stages import (stage0_ego_io, stage2_track, stage4_hand,
                            stage5_ego_comp, stage6_contact, stage7_eval)

def test_report_has_before_after(tmp_path):
    ctx = RunContext(config.load_config(overrides={"num_frames": 16}), str(tmp_path))
    for m in (stage0_ego_io, stage2_track, stage4_hand, stage5_ego_comp, stage6_contact):
        m.run(ctx).save(ctx.stage_dir(m.NAME))
    b = stage7_eval.run(ctx)
    r = b.meta["report"]
    assert "pen_before_mm" in r and "pen_after_mm" in r
    assert r["pen_after_mm"] <= r["pen_before_mm"] + 1e-6
