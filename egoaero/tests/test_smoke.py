"""Task 18: end-to-end mock pipeline smoke test.

Runs all 8 stages on a minimal (24-frame) mock scene and asserts:
  - stage7 eval: penetration not increased after contact optimisation
  - stage2 track: centroid error decreased after pose-graph optimisation
  - stage4 hand: translation error decreased after depth-residual correction
  - contract output passes validate()
"""
from egoaero import config
from egoaero.pipeline import run_pipeline
from egoaero import contract


def test_end_to_end_mock_smoke(tmp_path):
    cfg = config.load_config(overrides={"num_frames": 24, "seed": 0})
    ctx = run_pipeline(cfg, str(tmp_path / "run"), stages="all")

    rep = ctx.load("stage7_eval").meta["report"]
    # contact optimisation must not worsen penetration
    assert rep["pen_after_mm"] <= rep["pen_before_mm"] + 1e-6, (
        f"penetration increased: before={rep['pen_before_mm']:.4f} "
        f"after={rep['pen_after_mm']:.4f}"
    )

    s2 = ctx.load("stage2_track").meta
    assert s2["track_err_deg_after"] < s2["track_err_deg_before"], (
        f"tracking error not reduced: before={s2['track_err_deg_before']:.4f} "
        f"after={s2['track_err_deg_after']:.4f}"
    )

    s4 = ctx.load("stage4_hand").meta
    assert s4["transl_err_after_mm"] < s4["transl_err_before_mm"], (
        f"hand translation error not reduced: before={s4['transl_err_before_mm']:.4f} "
        f"after={s4['transl_err_after_mm']:.4f}"
    )

    assert contract.validate(str(tmp_path / "run")), (
        "contract.validate() returned False — one or more required output files missing"
    )
