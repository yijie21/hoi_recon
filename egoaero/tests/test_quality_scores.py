# egoaero/tests/test_quality_scores.py
import numpy as np
from egoaero import quality as Q


def _gd(n=4):
    # 4 active frames; 'thumb' recoverable in 3 of them, 'index' never
    gap = {"thumb": np.array([0.001, 0.002, 0.001, 0.05]),
           "index": np.array([0.05, 0.05, 0.05, 0.05]),
           "middle": np.zeros(n), "ring": np.zeros(n), "little": np.zeros(n)}
    delta = {f: np.full(n, 0.003) for f in gap}
    return gap, delta


def test_recoverability_counts_indicator():
    gap, delta = _gd()
    rec = Q.recoverability(gap, delta, eps_g=0.004, eps_delta=0.012)
    assert abs(rec["thumb"] - 0.75) < 1e-9      # 3/4 frames pass
    assert rec["index"] == 0.0                  # gap never < eps_g
    assert rec["middle"] == 1.0                 # gap 0 < eps_g, delta < eps_delta


def test_repair_budget_normalizes():
    _, delta = _gd()
    b = Q.repair_budget(delta, delta_max=0.015)
    assert abs(b - (0.003 / 0.015)) < 1e-9      # median delta / delta_max


def test_residual_after_dimensionless():
    gap, _ = _gd()
    r = Q.residual_after(pen_after_mm=25000.0, gap_after=gap,
                         pen_ref_mm=50000.0, gap_ref_mm=40.0)
    assert r > 0 and np.isfinite(r)


def test_unresolved_ratio():
    gap, delta = _gd()
    moving = np.array([True, True, True, True])
    # frame 3: thumb gap 0.05 (not recoverable) but middle/ring/little gap 0 -> recoverable_any True
    # so with all-zero other fingers, every frame has a recoverable finger -> U = 0
    u = Q.unresolved_ratio(gap, delta, moving, eps_g=0.004, eps_delta=0.012)
    assert u == 0.0
    # now make ALL fingers unrecoverable on frame 3
    gap2 = {f: v.copy() for f, v in gap.items()}
    for f in gap2:
        gap2[f][3] = 0.05
    u2 = Q.unresolved_ratio(gap2, delta, moving, eps_g=0.004, eps_delta=0.012)
    assert abs(u2 - 0.25) < 1e-9                # 1 of 4 moving frames unresolved
