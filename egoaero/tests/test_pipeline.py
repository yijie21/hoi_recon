# egoaero/tests/test_pipeline.py
from egoaero import config
from egoaero.pipeline import run_pipeline


def test_full_mock_pipeline_runs(tmp_path):
    cfg = config.load_config(overrides={"num_frames": 16})
    ctx = run_pipeline(cfg, str(tmp_path / "run"), stages="all")
    rep = ctx.load("stage7_eval").meta["report"]
    assert rep["pen_after_mm"] <= rep["pen_before_mm"] + 1e-6
