import json, os
from egoaero import config
from egoaero.pipeline import RunContext
from egoaero.stages import (stage0_ego_io, stage2_track, stage3_mesh, stage4_hand,
                            stage5_ego_comp, stage6_contact, stage8_quality)

def _prep(tmp_path):
    ctx = RunContext(config.load_config(overrides={"num_frames": 16}), str(tmp_path))
    for m in (stage0_ego_io, stage2_track, stage3_mesh, stage4_hand,
              stage5_ego_comp, stage6_contact):
        m.run(ctx).save(ctx.stage_dir(m.NAME))
    return ctx

def test_stage8_emits_decision_and_json(tmp_path):
    ctx = _prep(tmp_path)
    b = stage8_quality.run(ctx)
    q = b.meta["quality"]
    assert q["decision"] in ("accept", "repairable_accept", "recapture")
    assert 0.0 < q["Q"] <= 1.0
    for f in ("thumb", "index", "middle", "ring", "little"):
        assert 0.0 <= q["per_finger"][f]["Q_rec"] <= 1.0
    assert os.path.exists(os.path.join(ctx.run_dir, "quality.json"))
    with open(os.path.join(ctx.run_dir, "quality.json")) as fh:
        assert json.load(fh)["decision"] == q["decision"]
