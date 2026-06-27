"""Smoke tests: the whole mock pipeline runs end-to-end and refinement helps.

Run with:  pytest -q   (or: python -m pytest tests/)
These need only numpy + scipy + pyyaml (no model weights).
"""
import os
import tempfile

import numpy as np

from hoi_recon.config import load_config
from hoi_recon.pipeline import run_pipeline


def _run(tmp, frames=32):
    cfg = load_config(None, {"mock": True, "num_frames": frames, "force": True})
    ctx = run_pipeline(cfg, os.path.join(tmp, "run"), "all")
    return ctx


def test_pipeline_runs_and_produces_all_stages():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _run(tmp)
        for name in ["stage0_preprocess", "stage4_align", "stage6_rectify",
                     "stage7_contact_optim", "stage8_eval"]:
            assert ctx.has(name), f"missing {name}"
        rep = ctx.load("stage8_eval").meta
        assert rep["mock"] is True
        # pseudo-GT exported for the feed-forward model
        assert os.path.exists(ctx.load("stage8_eval").assets["pseudo_gt"])


def test_refinement_reduces_hand_and_penetration_error():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _run(tmp)
        rep = ctx.load("stage8_eval").meta
        # hand smoothing reduces joint error and jitter
        assert rep["hand"]["mpjpe_mm_smoothed_stage5"] < rep["hand"]["mpjpe_mm_raw_stage2"]
        assert rep["hand"]["jitter_accel_stage5"] < rep["hand"]["jitter_accel_stage2"]
        # contact-aware optimization reduces penetration
        assert rep["penetration_depth_sum"]["stage7"] < rep["penetration_depth_sum"]["stage5"]


def test_stage_caching_and_resume():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = load_config(None, {"mock": True, "num_frames": 24})
        run_dir = os.path.join(tmp, "run")
        run_pipeline(cfg, run_dir, "0-3")
        # resume the remaining stages; earlier ones are cached
        ctx = run_pipeline(cfg, run_dir, "all")
        assert ctx.has("stage8_eval")


def test_viser_viewer_builds_and_renders():
    pytest = __import__("pytest")
    if __import__("importlib").util.find_spec("viser") is None:
        pytest.skip("viser not installed")
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _run(tmp)
        from hoi_recon.viz import viser_app as V
        srv, render, gui_contacts = V.launch(ctx.run_dir, port=8147, block=False)
        try:
            for t in [0, 5, 10]:
                render(t)
            gui_contacts.value = True
            render(7)
        finally:
            srv.stop()


def test_geometry_umeyama_recovers_similarity():
    from hoi_recon.geometry import umeyama, rotvec_to_R, transform_points, se3
    rng = np.random.default_rng(0)
    src = rng.normal(size=(50, 3))
    R = rotvec_to_R(np.array([0.3, -0.7, 0.2]))
    s, t = 1.7, np.array([0.5, -1.0, 2.0])
    dst = s * src @ R.T + t
    s2, R2, t2 = umeyama(src, dst, with_scale=True)
    rec = s2 * src @ R2.T + t2
    assert np.allclose(rec, dst, atol=1e-6)
