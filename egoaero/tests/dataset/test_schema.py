from egoaero import config
from egoaero.pipeline import run_pipeline
from egoaero.dataset import schema


def _run(tmp_path):
    cfg = config.load_config(overrides={"num_frames": 12})
    run_pipeline(cfg, str(tmp_path / "run"), "all")   # auto-writes contract + quality.json
    return str(tmp_path / "run")


def _meta():
    return {"task_description": "pick up the object", "manipulated_object": "object",
            "relational_objects": ["table"], "difficulty": 3, "decision": "accept",
            "frames": 12, "seq_id": "seq_0000"}


def test_write_and_validate(tmp_path):
    run = _run(tmp_path)
    ds = str(tmp_path / "dataset")
    man = schema.write_sequence(ds, "seq_0000", run, _meta())
    assert "hand_mano" in man and "raw_obs" in man and "metadata" in man
    assert schema.validate_sequence(ds, "seq_0000") is True
    assert schema.read_metadata(ds, "seq_0000")["difficulty"] == 3


def test_validate_false_when_missing_field(tmp_path):
    run = _run(tmp_path)
    ds = str(tmp_path / "dataset")
    bad = _meta(); del bad["difficulty"]
    schema.write_sequence(ds, "seq_bad", run, bad)
    assert schema.validate_sequence(ds, "seq_bad") is False
