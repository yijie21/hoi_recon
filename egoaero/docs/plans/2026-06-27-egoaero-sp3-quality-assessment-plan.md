# EgoAERO SP3 — Online Quality Assessment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add EgoAERO's App-E online quality assessment to the `egoaero/` method: score a reconstructed clip on bounded recoverability and emit an accept / repairable_accept / recapture decision, reusing stage5 (coarse) and stage6 (repaired) as the before/after of App-E's bounded projection.

**Architecture:** A pure, unit-tested `quality.py` (App-E scoring functions, no I/O) plus a thin `stage8_quality` pipeline stage that loads stage5+stage6, runs the scorers, prints a verdict, and writes `meta["quality"]` + `<run>/quality.json`. New `cfg.quality` defaults; stage8 appended to the pipeline (9 stages, 0–8). The 4D-HOI contract output is unchanged.

**Tech Stack:** Python 3.9+, numpy, pytest. Pure-numpy (no torch).

## Global Constraints

- **Self-contained:** `egoaero/` MUST NOT import from `render_and_compare/` or any sibling. `quality.py` imports only `egoaero.*` (it reuses `signed_distance`/`active_window`/`_obj_world` from `egoaero.stages.stage6_contact` and `egoaero.core.hand`).
- **Faithful where specified; documented defaults at gaps**, logged in `egoaero/ASSUMPTIONS.md`. App-E specifies the equations but NO constants — every threshold/weight is a documented default.
- **No re-optimization:** SP3 reads stage5 (before) and stage6 (after); it never re-runs the contact optimization.
- **Units:** metres internally; mm only in reported/normalized metrics.
- **Determinism:** no randomness in SP3 (pure functions of the reconstruction).
- **Stage convention:** module defines `NAME: str`, `INDEX: int`, `run(ctx) -> Bundle`.
- **Decision labels (exact strings):** `"accept"`, `"repairable_accept"`, `"recapture"`.
- **Commits:** one per task, message prefix `egoaero:`.
- **Tests:** CWD-safe; run with `python -m pytest egoaero/tests/ -q` (default `python` has numpy/scipy/pyyaml/pytest/trimesh).

## Reuse reference (exact existing signatures — do not redefine)

From `egoaero/egoaero/stages/stage6_contact.py` (module-level):
- `signed_distance(points, obj_pts, obj_normals) -> (s[N], nn[N,3])` — sign: + outside, − inside.
- `active_window(stage_labels) -> list[int]` — frame indices with label in {grasp,move,place}.
- `_obj_world(obj_verts, obj_faces, pose) -> (ow[Npts,3], on[Npts,3])`.

From `egoaero/egoaero/core/hand.py`:
- `FINGERS = ["thumb","index","middle","ring","little"]`
- `fingertip_pad_idx(finger_idx, finger) -> np.ndarray` (distal pad vertex indices).

Stage bundles available:
- `stage5_ego_comp`: arrays `hand_verts_t[T,Nh,3]` (coarse), `hand_joints_t`, `obj_poses_t[T,4,4]`, `obj_verts[No,3]`, `obj_faces[Mo,3]`; meta `finger_idx` (dict of lists incl. float `z_norm`), `stage_labels` (list[str]).
- `stage6_contact`: arrays `hand_verts_t[T,Nh,3]` (repaired); meta `pen_after_mm`, `gap_after_mm`, `finger_idx`, `stage_labels`.

finger_idx rehydration (use verbatim — z_norm stays float):
```python
fidx = {k: (np.asarray(v, float) if k == "z_norm" else np.asarray(v, int))
        for k, v in meta["finger_idx"].items()}
```

---

## File Structure

```
egoaero/egoaero/
  config.py                       Task 1 (add cfg.quality block)
  quality.py                      Tasks 2-4 (NEW: pure App-E scorers)
  stages/stage8_quality.py        Task 5 (NEW stage)
  pipeline.py                     Task 6 (append stage8 to STAGES)
  stages/__init__.py              Task 6 (export stage8_quality)
egoaero/tests/
  test_quality.py                 Tasks 2-4 (unit tests)
  test_stage8.py                  Task 5
  test_smoke.py                   Task 6 (extend)
egoaero/README.md, ASSUMPTIONS.md Task 6 (docs)
```

---

### Task 1: `cfg.quality` defaults block

**Files:**
- Modify: `egoaero/egoaero/config.py` (add a `"quality"` key to `_DEFAULTS`, near `"contact"`)
- Modify: `egoaero/ASSUMPTIONS.md`
- Test: `egoaero/tests/test_quality_config.py`

**Interfaces:**
- Produces: `cfg.quality.{eps_g_m, eps_delta_m, delta_max_m, alpha, beta, gamma, pen_ref_mm, gap_ref_mm, obj_move_thresh_mps, q_accept, q_repairable}`.

- [ ] **Step 1: Write the failing test**

```python
# egoaero/tests/test_quality_config.py
from egoaero import config

def test_quality_defaults_present():
    q = config.load_config().quality
    assert q.eps_g_m == 0.004 and q.eps_delta_m == 0.012 and q.delta_max_m == 0.015
    assert q.alpha == 1.0 and q.beta == 0.5 and q.gamma == 1.0
    assert q.q_accept == 0.6 and q.q_repairable == 0.3
    assert q.pen_ref_mm == 50000.0 and q.gap_ref_mm == 40.0
    assert q.obj_move_thresh_mps == 0.01

def test_quality_override_merges():
    q = config.load_config(overrides={"quality": {"q_accept": 0.7}}).quality
    assert q.q_accept == 0.7 and q.q_repairable == 0.3   # other defaults kept
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest egoaero/tests/test_quality_config.py -v`
Expected: FAIL (`AttributeError: quality`).

- [ ] **Step 3: Write minimal implementation**

In `egoaero/egoaero/config.py`, add this block to `_DEFAULTS` immediately after the `"contact": {...}` entry:

```python
    "quality": {                  # App E — paper gives NO constants; all DOCUMENTED defaults
        "eps_g_m": 0.004,         # contact-gap recoverability threshold (4 mm)
        "eps_delta_m": 0.012,     # per-finger correction-budget threshold (12 mm)
        "delta_max_m": 0.015,     # budget normalizer (= contact.max_finger_disp_m)
        "alpha": 1.0,             # R_after weight in Q
        "beta": 0.5,              # B_repair weight in Q
        "gamma": 1.0,             # U_unresolved weight in Q
        "pen_ref_mm": 50000.0,    # R_after penetration normalizer (mock-scale)
        "gap_ref_mm": 40.0,       # R_after contact-gap normalizer
        "obj_move_thresh_mps": 0.01,  # object-moving threshold for U_unresolved
        "q_accept": 0.6,          # Q >= -> accept
        "q_repairable": 0.3,      # q_repairable <= Q < q_accept -> repairable_accept
    },
```

Append to `egoaero/ASSUMPTIONS.md` under a new `## SP3 — Online quality assessment (App E)` heading:

```markdown
## SP3 — Online quality assessment (App E)
App E specifies the equations (Q_rec, B_repair, Q=exp(-aR-bB-gU), 3 decisions) but NO constants.
Documented defaults (config.quality): eps_g=4mm, eps_delta=12mm, delta_max=15mm (=max_finger_disp),
alpha=1.0, beta=0.5, gamma=1.0, pen_ref=50000mm & gap_ref=40mm (R_after normalizers, mock-scaled),
obj_move_thresh=0.01 m/frame (U_unresolved), q_accept=0.6 / q_repairable=0.3 (decision thresholds).
U_unresolved heuristic (paper names but does not define): fraction of active frames where the object
is moving yet NO finger has recoverable contact. Tuned so a clean mock clip scores "accept".
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest egoaero/tests/test_quality_config.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add egoaero/egoaero/config.py egoaero/ASSUMPTIONS.md egoaero/tests/test_quality_config.py
git commit -m "egoaero: cfg.quality App-E defaults block"
```

---

### Task 2: `quality.py` — per-finger gap & delta

**Files:**
- Create: `egoaero/egoaero/quality.py`
- Test: `egoaero/tests/test_quality.py`

**Interfaces:**
- Produces:
  - `per_finger_gap(hand_verts_seq, finger_idx, obj_world_seq, window) -> dict[str, np.ndarray]` — per finger, array of length `len(window)` of median pad-vertex distance (metres, `|signed_distance|`) to the object surface.
  - `per_finger_delta(coarse_verts_seq, repaired_verts_seq, finger_idx, window) -> dict[str, np.ndarray]` — per finger, array length `len(window)` of mean pad-vertex displacement (metres) between coarse and repaired hand.
  - `obj_world_seq` is a list indexed by frame `t`, each element `(ow[Npts,3], on[Npts,3])` (object surface points + normals in world/table frame). Only frames in `window` are read.

- [ ] **Step 1: Write the failing test**

```python
# egoaero/tests/test_quality.py
import numpy as np
from egoaero import quality as Q
from egoaero.core import hand as H

def _hand_and_obj():
    v, j, fidx = H.procedural_hand(seed=0)
    # flat object plane at z=0, +z normals, covering xy
    grid = np.array([[x, y, 0.0] for x in np.linspace(-0.1, 0.1, 9)
                     for y in np.linspace(-0.1, 0.1, 9)])
    nrm = np.tile([0, 0, 1.0], (grid.shape[0], 1))
    return v, j, fidx, grid, nrm

def test_per_finger_gap_shapes_and_value():
    v, j, fidx, grid, nrm = _hand_and_obj()
    # place the hand so index-finger pad sits ~2cm above the plane
    verts_seq = np.stack([v, v], 0)                      # T=2 identical frames
    obj_seq = [(grid, nrm), (grid, nrm)]
    gaps = Q.per_finger_gap(verts_seq, fidx, obj_seq, window=[0, 1])
    assert set(H.FINGERS).issubset(gaps.keys())
    assert gaps["index"].shape == (2,)
    assert np.all(gaps["index"] >= 0)                    # distances are non-negative

def test_per_finger_delta_measures_displacement():
    v, j, fidx, grid, nrm = _hand_and_obj()
    coarse = np.stack([v, v], 0)
    repaired = coarse.copy()
    pad = H.fingertip_pad_idx(fidx, "thumb")
    repaired[:, pad] += np.array([0.0, 0.0, -0.01])      # move thumb pad 1cm
    d = Q.per_finger_delta(coarse, repaired, fidx, window=[0, 1])
    assert np.allclose(d["thumb"], 0.01, atol=1e-9)
    assert np.allclose(d["index"], 0.0, atol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest egoaero/tests/test_quality.py -v`
Expected: FAIL (`ModuleNotFoundError: egoaero.quality`).

- [ ] **Step 3: Write minimal implementation**

```python
# egoaero/egoaero/quality.py
"""EgoAERO App-E online quality assessment — pure scoring functions.

Reads the coarse hand (stage5) and the repaired hand (stage6) — App E's bounded
projection is exactly the stage6 contact optimization — and scores bounded
recoverability into a quality value Q and an accept/repairable/recapture decision.
No I/O, no re-optimization, deterministic. Constants live in cfg.quality (documented).
"""
from __future__ import annotations
import numpy as np

from .stages.stage6_contact import signed_distance
from .core import hand as H


def per_finger_gap(hand_verts_seq, finger_idx, obj_world_seq, window):
    """Per finger: median |distance| of its fingertip-pad vertices to the object
    surface, for each frame in `window`. Returns {finger: array[len(window)]} (metres)."""
    out = {f: np.zeros(len(window)) for f in H.FINGERS}
    for jpos, t in enumerate(window):
        ow, on = obj_world_seq[t]
        for f in H.FINGERS:
            pad = H.fingertip_pad_idx(finger_idx, f)
            if len(pad) == 0:
                out[f][jpos] = 0.0
                continue
            s, _ = signed_distance(hand_verts_seq[t][pad], ow, on)
            out[f][jpos] = float(np.median(np.abs(s)))
    return out


def per_finger_delta(coarse_verts_seq, repaired_verts_seq, finger_idx, window):
    """Per finger: mean pad-vertex Euclidean displacement between coarse and repaired
    hand, for each frame in `window`. Returns {finger: array[len(window)]} (metres)."""
    out = {f: np.zeros(len(window)) for f in H.FINGERS}
    for jpos, t in enumerate(window):
        for f in H.FINGERS:
            pad = H.fingertip_pad_idx(finger_idx, f)
            if len(pad) == 0:
                out[f][jpos] = 0.0
                continue
            disp = repaired_verts_seq[t][pad] - coarse_verts_seq[t][pad]
            out[f][jpos] = float(np.mean(np.linalg.norm(disp, axis=1)))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest egoaero/tests/test_quality.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add egoaero/egoaero/quality.py egoaero/tests/test_quality.py
git commit -m "egoaero: quality.py per-finger gap + delta (App E)"
```

---

### Task 3: `quality.py` — recoverability, budget, residual, unresolved

**Files:**
- Modify: `egoaero/egoaero/quality.py` (append four functions)
- Test: `egoaero/tests/test_quality_scores.py`

**Interfaces:**
- Consumes: `per_finger_gap`/`per_finger_delta` outputs (dict finger → array[len(window)]).
- Produces:
  - `recoverability(gap_after, delta, eps_g, eps_delta) -> dict[str, float]` — `Q_rec^f ∈ [0,1]`.
  - `repair_budget(delta, delta_max) -> float` — `median_{t,f}(delta)/delta_max`.
  - `residual_after(pen_after_mm, gap_after, pen_ref_mm, gap_ref_mm) -> float` — dimensionless `R_after`. `gap_after` is the `per_finger_gap` dict for the REPAIRED hand.
  - `unresolved_ratio(gap_after, delta, object_moving, eps_g, eps_delta) -> float` — `U ∈ [0,1]`. `object_moving` is a bool array of length `len(window)`.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest egoaero/tests/test_quality_scores.py -v`
Expected: FAIL (`AttributeError: recoverability`).

- [ ] **Step 3: Write minimal implementation**

Append to `egoaero/egoaero/quality.py`:

```python
def recoverability(gap_after, delta, eps_g, eps_delta):
    """Q_rec^f = fraction of active frames where g_after < eps_g AND ||delta|| < eps_delta."""
    out = {}
    for f in H.FINGERS:
        ga, df = gap_after[f], delta[f]
        ok = (ga < eps_g) & (df < eps_delta)
        out[f] = float(np.mean(ok)) if len(ok) else 0.0
    return out


def repair_budget(delta, delta_max):
    """B_repair = median over all (frame, finger) of ||delta|| / delta_max."""
    alld = np.concatenate([delta[f] for f in H.FINGERS]) if delta else np.zeros(1)
    if alld.size == 0:
        return 0.0
    return float(np.median(alld) / max(delta_max, 1e-9))


def residual_after(pen_after_mm, gap_after, pen_ref_mm, gap_ref_mm):
    """R_after = pen_after/pen_ref + (median-over-fingers median-over-frames gap)*1000/gap_ref.
    Dimensionless remaining penetration + contact-gap residual."""
    per_finger_med = [np.median(gap_after[f]) for f in H.FINGERS if len(gap_after[f])]
    gap_med_m = float(np.median(per_finger_med)) if per_finger_med else 0.0
    return float(pen_after_mm / max(pen_ref_mm, 1e-9) + (gap_med_m * 1000.0) / max(gap_ref_mm, 1e-9))


def unresolved_ratio(gap_after, delta, object_moving, eps_g, eps_delta):
    """Fraction of object-moving active frames where NO finger has recoverable contact."""
    moving = np.asarray(object_moving, bool)
    n = len(moving)
    if n == 0 or moving.sum() == 0:
        return 0.0
    unresolved = 0
    for j in range(n):
        if not moving[j]:
            continue
        any_rec = any((gap_after[f][j] < eps_g) and (delta[f][j] < eps_delta) for f in H.FINGERS)
        if not any_rec:
            unresolved += 1
    return float(unresolved / moving.sum())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest egoaero/tests/test_quality_scores.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add egoaero/egoaero/quality.py egoaero/tests/test_quality_scores.py
git commit -m "egoaero: quality.py recoverability + budget + residual + unresolved (App E)"
```

---

### Task 4: `quality.py` — quality_score & decision

**Files:**
- Modify: `egoaero/egoaero/quality.py` (append two functions)
- Test: `egoaero/tests/test_quality_decision.py`

**Interfaces:**
- Produces:
  - `quality_score(R_after, B_repair, U_unresolved, alpha, beta, gamma) -> float` — `Q = exp(−αR−βB−γU) ∈ (0,1]`.
  - `decision(Q, per_finger_Q_rec, q_accept, q_repairable) -> (str, dict)` — label in `{"accept","repairable_accept","recapture"}` + `failure_attribution` dict with keys `low_recoverability_fingers` (list of fingers with `Q_rec < 0.5`) and `Q` (the score).

- [ ] **Step 1: Write the failing test**

```python
# egoaero/tests/test_quality_decision.py
import math
from egoaero import quality as Q

def test_quality_score_monotonic_and_bounded():
    base = Q.quality_score(0.2, 0.2, 0.0, 1.0, 0.5, 1.0)
    assert 0 < base <= 1.0
    assert Q.quality_score(0.5, 0.2, 0.0, 1.0, 0.5, 1.0) < base   # higher R -> lower Q
    assert Q.quality_score(0.2, 0.9, 0.0, 1.0, 0.5, 1.0) < base   # higher B -> lower Q
    assert Q.quality_score(0.2, 0.2, 0.5, 1.0, 0.5, 1.0) < base   # higher U -> lower Q
    assert Q.quality_score(0.0, 0.0, 0.0, 1.0, 0.5, 1.0) == 1.0

def test_decision_thresholds():
    rec = {"thumb": 0.9, "index": 0.8, "middle": 0.1, "ring": 0.0, "little": 0.0}
    lab_a, info_a = Q.decision(0.7, rec, q_accept=0.6, q_repairable=0.3)
    lab_r, _ = Q.decision(0.45, rec, q_accept=0.6, q_repairable=0.3)
    lab_x, _ = Q.decision(0.1, rec, q_accept=0.6, q_repairable=0.3)
    assert lab_a == "accept" and lab_r == "repairable_accept" and lab_x == "recapture"
    assert set(info_a["low_recoverability_fingers"]) == {"middle", "ring", "little"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest egoaero/tests/test_quality_decision.py -v`
Expected: FAIL (`AttributeError: quality_score`).

- [ ] **Step 3: Write minimal implementation**

Append to `egoaero/egoaero/quality.py`:

```python
def quality_score(R_after, B_repair, U_unresolved, alpha, beta, gamma):
    """Q = exp(-alpha*R_after - beta*B_repair - gamma*U_unresolved) in (0, 1]."""
    return float(np.exp(-(alpha * R_after + beta * B_repair + gamma * U_unresolved)))


def decision(Q, per_finger_Q_rec, q_accept, q_repairable):
    """Map Q to a collection decision + failure attribution."""
    if Q >= q_accept:
        label = "accept"
    elif Q >= q_repairable:
        label = "repairable_accept"
    else:
        label = "recapture"
    low = [f for f, v in per_finger_Q_rec.items() if v < 0.5]
    return label, {"low_recoverability_fingers": low, "Q": float(Q)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest egoaero/tests/test_quality_decision.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add egoaero/egoaero/quality.py egoaero/tests/test_quality_decision.py
git commit -m "egoaero: quality.py quality_score + decision (App E)"
```

---

### Task 5: `stage8_quality` stage

**Files:**
- Create: `egoaero/egoaero/stages/stage8_quality.py`
- Test: `egoaero/tests/test_stage8.py`

**Interfaces:**
- Consumes: `stage5_ego_comp`, `stage6_contact` bundles; `cfg.quality`; reuses `stage6_contact.active_window`/`_obj_world` and `quality.py`.
- Produces: bundle `stage8_quality` with `meta["quality"] = {Q, decision, per_finger:{<finger>:{gap_before_mm, gap_after_mm, Q_rec}}, B_repair, R_after, U_unresolved, failure_attribution}`; writes `<run>/quality.json` with the same dict. `NAME="stage8_quality"; INDEX=8`.

- [ ] **Step 1: Write the failing test**

```python
# egoaero/tests/test_stage8.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest egoaero/tests/test_stage8.py -v`
Expected: FAIL (`ImportError: cannot import name 'stage8_quality'`).

- [ ] **Step 3: Write minimal implementation**

```python
# egoaero/egoaero/stages/stage8_quality.py
"""Stage 8 (Sec 3 / App E): online quality assessment. Scores the reconstructed clip
on bounded recoverability (coarse=stage5 vs repaired=stage6) and emits an
accept / repairable_accept / recapture decision. Supplementary diagnostic — the 4D-HOI
contract output is unchanged."""
from __future__ import annotations
import json, os
import numpy as np

from ..bundle import Bundle
from .. import quality as Q
from .stage6_contact import active_window, _obj_world

NAME = "stage8_quality"; INDEX = 8


def run(ctx) -> Bundle:
    cfg = ctx.cfg; qc = cfg.quality
    s5 = ctx.load("stage5_ego_comp"); s6 = ctx.load("stage6_contact")
    coarse = s5["hand_verts_t"]; repaired = s6["hand_verts_t"]
    obj_poses = s5["obj_poses_t"]; ov = s5["obj_verts"]; of = s5["obj_faces"].astype(int)
    fidx = {k: (np.asarray(v, float) if k == "z_norm" else np.asarray(v, int))
            for k, v in s5.meta["finger_idx"].items()}
    labels = s5.meta["stage_labels"]; T = coarse.shape[0]
    window = active_window(labels)

    # object surface points+normals per active frame (reuse stage6 helper)
    obj_world_seq = [None] * T
    for t in window:
        obj_world_seq[t] = _obj_world(ov, of, obj_poses[t])

    gap_before = Q.per_finger_gap(coarse, fidx, obj_world_seq, window)
    gap_after = Q.per_finger_gap(repaired, fidx, obj_world_seq, window)
    delta = Q.per_finger_delta(coarse, repaired, fidx, window)

    rec = Q.recoverability(gap_after, delta, qc.eps_g_m, qc.eps_delta_m)
    budget = Q.repair_budget(delta, qc.delta_max_m)
    R_after = Q.residual_after(s6.meta["pen_after_mm"], gap_after, qc.pen_ref_mm, qc.gap_ref_mm)

    # object-moving flag per active frame (translation speed between consecutive frames)
    moving = np.zeros(len(window), bool)
    for jpos, t in enumerate(window):
        tp = max(t - 1, 0)
        speed = float(np.linalg.norm(obj_poses[t][:3, 3] - obj_poses[tp][:3, 3]))
        moving[jpos] = speed > qc.obj_move_thresh_mps
    U = Q.unresolved_ratio(gap_after, delta, moving, qc.eps_g_m, qc.eps_delta_m)

    Qval = Q.quality_score(R_after, budget, U, qc.alpha, qc.beta, qc.gamma)
    label, attr = Q.decision(Qval, rec, qc.q_accept, qc.q_repairable)

    per_finger = {f: {"gap_before_mm": float(np.median(gap_before[f]) * 1000) if len(window) else 0.0,
                      "gap_after_mm": float(np.median(gap_after[f]) * 1000) if len(window) else 0.0,
                      "Q_rec": rec[f]} for f in Q.H.FINGERS}
    report = {"Q": Qval, "decision": label, "per_finger": per_finger,
              "B_repair": budget, "R_after": R_after, "U_unresolved": U,
              "failure_attribution": attr}

    print(f"  quality: decision={label}  Q={Qval:.3f}  B_repair={budget:.3f}  "
          f"R_after={R_after:.3f}  U={U:.3f}")
    with open(os.path.join(ctx.run_dir, "quality.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    return Bundle(meta={"quality": report})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest egoaero/tests/test_stage8.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add egoaero/egoaero/stages/stage8_quality.py egoaero/tests/test_stage8.py
git commit -m "egoaero: stage8 online quality assessment (App E verdict + quality.json)"
```

---

### Task 6: Pipeline wiring, smoke test, docs

**Files:**
- Modify: `egoaero/egoaero/pipeline.py` (append `stage8_quality` to `STAGES`)
- Modify: `egoaero/egoaero/stages/__init__.py` (export `stage8_quality`)
- Modify: `egoaero/tests/test_smoke.py` (assert stage8 verdict)
- Modify: `egoaero/README.md` (stage list 0–8 + quality section)
- Test: `egoaero/tests/test_smoke.py`

**Interfaces:**
- Consumes: all stage modules incl. `stage8_quality`.
- Produces: full 9-stage pipeline; `python -m egoaero.cli --mock` prints the quality verdict and writes `quality.json`.

- [ ] **Step 1: Write the failing test**

Append to `egoaero/tests/test_smoke.py`:

```python
def test_smoke_quality_verdict(tmp_path):
    from egoaero import config
    from egoaero.pipeline import run_pipeline
    cfg = config.load_config(overrides={"num_frames": 24, "seed": 0})
    ctx = run_pipeline(cfg, str(tmp_path / "run"), stages="all")
    q = ctx.load("stage8_quality").meta["quality"]
    assert q["decision"] in ("accept", "repairable_accept", "recapture")
    assert 0.0 < q["Q"] <= 1.0
    assert "thumb" in q["per_finger"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest egoaero/tests/test_smoke.py::test_smoke_quality_verdict -v`
Expected: FAIL (`stage8_quality` not in pipeline / `FileNotFoundError`).

- [ ] **Step 3: Write minimal implementation**

In `egoaero/egoaero/stages/__init__.py`, add the import line (with the others):
```python
from . import stage8_quality  # noqa: F401
```

In `egoaero/egoaero/pipeline.py`, update the import and `STAGES` list to include stage8:
```python
from .stages import (stage0_ego_io, stage1_semantic, stage2_track, stage3_mesh,
                     stage4_hand, stage5_ego_comp, stage6_contact, stage7_eval,
                     stage8_quality)

STAGES = [stage0_ego_io, stage1_semantic, stage2_track, stage3_mesh,
          stage4_hand, stage5_ego_comp, stage6_contact, stage7_eval, stage8_quality]
```
(Leave the existing contract hook keyed on stage7/stage6 unchanged — quality is supplementary.)

In `egoaero/README.md`: change the stage list to 0–8 (add "stage8 quality — App-E accept/repairable/recapture verdict") and add a short "Online quality assessment" subsection noting `quality.json` output and that thresholds are documented defaults (pointer to `ASSUMPTIONS.md`).

- [ ] **Step 4: Run tests**

Run: `python -m pytest egoaero/tests/ -q`
Expected: all green (prior suite + the new quality/stage8/smoke tests).
Also manually: `cd egoaero && python -m egoaero.cli --out /tmp/egoaero_q --mock --num-frames 24` — expect the `quality: decision=...` line and `/tmp/egoaero_q/quality.json` to exist.

- [ ] **Step 5: Commit**

```bash
git add egoaero/egoaero/pipeline.py egoaero/egoaero/stages/__init__.py egoaero/tests/test_smoke.py egoaero/README.md
git commit -m "egoaero: wire stage8 quality into pipeline + smoke + README"
```

---

## Self-Review

**Spec coverage (SP3 spec §2–§5 → tasks):**
- `cfg.quality` defaults (§2.3) → Task 1 ✓
- `per_finger_gap`/`per_finger_delta` (§2.1) → Task 2 ✓
- `recoverability`/`repair_budget`/`residual_after`/`unresolved_ratio` (§2.1) → Task 3 ✓
- `quality_score`/`decision` (§2.1) → Task 4 ✓
- `stage8_quality` stage + `quality.json` + `meta["quality"]` (§2.2) → Task 5 ✓
- pipeline wiring (9 stages), smoke, README, ASSUMPTIONS (§2,§5,§6) → Tasks 1 (ASSUMPTIONS) + 6 ✓
- Faithfulness/no-re-optimization (§4, global constraints) → Task 5 reads stage5/stage6 only ✓

**Placeholder scan:** none — every step has complete code/commands.

**Type consistency:** `per_finger_gap`/`per_finger_delta` return `dict[finger→array]`; `recoverability`/`unresolved_ratio` consume those dicts; `residual_after` takes the `gap_after` dict + `pen_after_mm` (matches stage6 meta key `pen_after_mm`); `decision` returns `(str, dict)` and stage8 unpacks `label, attr`; `quality.H.FINGERS` is accessible because `quality.py` does `from .core import hand as H`. `obj_world_seq` is a per-frame list indexed by `t`, written only for `window` frames and read only for those — consistent between Task 2 (consumer) and Task 5 (producer). Decision strings match the global-constraint exact labels.
