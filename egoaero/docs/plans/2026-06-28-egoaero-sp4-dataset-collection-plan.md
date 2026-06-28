# EgoAERO SP4 — EgoDex-R Dataset + Collection Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce EgoAERO's closed-loop data collection (Sec 3) and the EgoDex-R per-sequence schema (App F): reconstruct→quality-assess→accept/repairable_accept/recapture over a synthetic capture source, writing accepted sequences (full App-F fields + heuristic difficulty/metadata) into a mock EgoDex-R dataset with a Table-1-style summary.

**Architecture:** A pure-numpy `egoaero/egoaero/dataset/` subsystem reusing the SP1 pipeline (`run_pipeline` stages 0–8, which auto-writes `<run>/contract/`) and SP3 quality (`<run>/quality.json`). A small backward-compatible `mock_tightness` knob lets the synthetic capture source produce genuinely different contact quality (hence a real spread of decisions) without touching the SP3 thresholds. No new heavy dependencies.

**Tech Stack:** Python 3.9+, numpy, pyyaml, pytest. Pure-numpy.

## Global Constraints

- **Self-contained:** no imports from `render_and_compare/` or any sibling. SP4 lives under `egoaero/egoaero/dataset/`.
- **Pure-numpy, no heavy deps:** the pipeline + quality used here are numpy-only; SP4 adds nothing heavier. The base suite must stay green.
- **Faithful where specified; documented defaults at gaps**, logged in `egoaero/ASSUMPTIONS.md`. Faithful: the accept/repairable_accept/recapture closed loop (Sec 3), the App-F schema field set, the Table-1 capability flags. Documented defaults/substitutions: difficulty heuristic (vs MLLM), synthetic capture source (vs FastUMI-Ego), task-description templating, dataset scale.
- **No threshold gaming:** the synthetic capture source varies *clip quality* (via `mock_tightness`), never the SP3 decision thresholds.
- **Decision labels (exact):** `"accept"`, `"repairable_accept"`, `"recapture"` (from SP3).
- **Determinism:** all randomness seeded.
- **Commits:** one per task, prefix `egoaero:`. **Tests:** CWD-safe; `python -m pytest egoaero/tests/ -q`.

## Reuse reference (exact, verified)

- `egoaero.pipeline.run_pipeline(cfg, run_dir, "all")` runs stages 0–8 and AUTO-writes `<run>/contract/` (pipeline hook) and `<run>/quality.json` (stage8). After it, both exist.
- `<run>/contract/`: `hand_mano.npz`, `object_traj.npz`, `object_mesh.obj`, `contact.npz`, `manifest.json`.
- `<run>/quality.json`: `{Q, decision, per_finger:{<finger>:{gap_before_mm,gap_after_mm,Q_rec}}, B_repair, R_after, U_unresolved, failure_attribution}`.
- `<run>/stage0_ego_io/arrays.npz`: `intrinsics`, `cam_traj`, `table_T_gt`, `depth`, `obj_mask`, `hand_mask` (+ gt_* ). `<run>/stage0_ego_io/meta.json`: `fps`, `T`, `image_size`.
- `egoaero.config.load_config(overrides=...)` → attribute-dict `Config`; `egoaero.bundle.Bundle.load(dir)`.

---

## File structure

```
egoaero/egoaero/dataset/
  __init__.py                 Task 1
  difficulty.py               Task 2
  schema.py                   Task 3
  capture.py                  Task 4
  collect.py                  Task 5
  cli.py                      Task 6
egoaero/egoaero/configs/dataset.yaml   Task 1
egoaero/egoaero/config.py              Task 1 (add mock_tightness knob)
egoaero/egoaero/core/mock_scene.py     Task 1 (honor tightness)
egoaero/egoaero/stages/stage0_ego_io.py Task 1 (pass tightness through)
egoaero/tests/dataset/                 one test file per task
```

---

### Task 1: dataset package, config, and the `mock_tightness` knob

**Files:**
- Create: `egoaero/egoaero/dataset/__init__.py`, `egoaero/tests/dataset/__init__.py`, `egoaero/egoaero/configs/dataset.yaml`
- Modify: `egoaero/egoaero/config.py` (add `mock_tightness` default), `egoaero/egoaero/core/mock_scene.py` (honor tightness), `egoaero/egoaero/stages/stage0_ego_io.py` (pass tightness)
- Test: `egoaero/tests/dataset/test_tightness.py`

**Interfaces:**
- Produces: `cfg.mock_tightness` (float, default 0.0 = current behavior); `generate_ego_hoi(..., tightness=0.0)` deepens the grasp press by `tightness * 0.05 m` at the contact frames; `dataset.yaml` with `collection`/`difficulty`/`capture` blocks.

- [ ] **Step 1: Write the failing test**

```python
# egoaero/tests/dataset/test_tightness.py
import numpy as np
from egoaero.core.mock_scene import generate_ego_hoi

def test_tightness_deepens_contact():
    loose = generate_ego_hoi(num_frames=24, seed=0, tightness=0.0)
    tight = generate_ego_hoi(num_frames=24, seed=0, tightness=1.0)
    # mid-clip the tight hand sits closer to / inside the object than the loose hand
    mid = 12
    obj_c = tight.obj_poses_w[mid, :3, 3]
    d_loose = np.linalg.norm(loose.hand_verts_w[mid].mean(0) - obj_c)
    d_tight = np.linalg.norm(tight.hand_verts_w[mid].mean(0) - obj_c)
    assert d_tight < d_loose

def test_tightness_default_unchanged():
    a = generate_ego_hoi(num_frames=16, seed=1)
    b = generate_ego_hoi(num_frames=16, seed=1, tightness=0.0)
    assert np.allclose(a.hand_verts_w, b.hand_verts_w)   # default == current behavior
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest egoaero/tests/dataset/test_tightness.py -v`
Expected: FAIL (`generate_ego_hoi() got an unexpected keyword argument 'tightness'`).

- [ ] **Step 3: Write minimal implementation**

Create empty `egoaero/egoaero/dataset/__init__.py` and `egoaero/tests/dataset/__init__.py`.

In `egoaero/egoaero/core/mock_scene.py`, add a `tightness=0.0` parameter to `generate_ego_hoi` and deepen the press. Find where the hand root z-offset / grasp `gap` is computed (the `gap`/`root` lines) and subtract an extra press at the contact frames:
```python
def generate_ego_hoi(num_frames=48, seed=0, image_size=(480, 640), fps=30.0, tightness=0.0):
    ...
    # after the existing `gap = ...` and `bump = ...` lines, before computing root:
    gap = gap - float(tightness) * 0.05 * bump   # deepen grasp press by up to 5 cm at contact
    ...
```
(Keep everything else identical; with `tightness=0.0` the term is zero, so default behavior is unchanged.)

In `egoaero/egoaero/config.py` `_DEFAULTS`, add near `"num_frames"`:
```python
    "mock_tightness": 0.0,   # SP4: synthetic grasp tightness knob (0=loose .. 1=tight contact)
```

In `egoaero/egoaero/stages/stage0_ego_io.py`, pass it through:
```python
    s = generate_ego_hoi(num_frames=int(cfg.num_frames), seed=int(cfg.seed),
                         tightness=float(cfg.get("mock_tightness", 0.0)))
```
(`cfg` is a `Config`; use `cfg.mock_tightness` if `get` is unavailable — `Config` supports attribute access, so `float(cfg.mock_tightness)` is fine since the default now exists.)

Create `egoaero/egoaero/configs/dataset.yaml`:
```yaml
collection:
  n_target: 5          # accepted sequences to collect (CLI --n overrides)
  max_attempts: 40     # cap on capture attempts
  num_frames: 32       # frames per synthetic clip
difficulty:
  w_occlusion: 1.0     # documented heuristic weights
  w_motion: 1.0
  w_residual: 1.0
  w_contact: 1.0       # subtracted (richer recoverable contact => easier)
capture:
  tightness_min: 0.0   # synthetic clip quality spread (loose -> recapture)
  tightness_max: 1.0   # (tight -> accept)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest egoaero/tests/dataset/test_tightness.py -v`
Expected: PASS (2 tests). Then confirm no regression: `python -m pytest egoaero/tests/ -q -k "not policy"` stays green (default tightness preserves behavior).

- [ ] **Step 5: Commit**

```bash
git add egoaero/egoaero/dataset/__init__.py egoaero/tests/dataset egoaero/egoaero/configs/dataset.yaml egoaero/egoaero/config.py egoaero/egoaero/core/mock_scene.py egoaero/egoaero/stages/stage0_ego_io.py
git commit -m "egoaero: SP4 dataset skeleton + config + mock_tightness knob"
```

---

### Task 2: `difficulty.py` — heuristic difficulty score (1–5)

**Files:**
- Create: `egoaero/egoaero/dataset/difficulty.py`
- Test: `egoaero/tests/dataset/test_difficulty.py`

**Interfaces:**
- Produces: `difficulty_score(quality_report, recon_summary) -> int` in `1..5`. `quality_report` is the SP3 dict (uses `R_after`, `U_unresolved`). `recon_summary` is `{"occlusion": float in [0,1], "obj_motion_m": float, "contact_richness": float in [0,1]}`. Higher occlusion/motion/residual/unresolved → harder; higher contact_richness → easier. Deterministic.

- [ ] **Step 1: Write the failing test**

```python
# egoaero/tests/dataset/test_difficulty.py
from egoaero.dataset.difficulty import difficulty_score

W = {"w_occlusion": 1.0, "w_motion": 1.0, "w_residual": 1.0, "w_contact": 1.0}

def _q(R=0.5, U=0.0):
    return {"R_after": R, "U_unresolved": U}

def test_bounds_1_to_5():
    easy = difficulty_score(_q(R=0.0, U=0.0),
                            {"occlusion": 0.0, "obj_motion_m": 0.0, "contact_richness": 1.0}, W)
    hard = difficulty_score(_q(R=3.0, U=1.0),
                            {"occlusion": 1.0, "obj_motion_m": 0.5, "contact_richness": 0.0}, W)
    assert 1 <= easy <= 5 and 1 <= hard <= 5
    assert easy == 1 and hard == 5

def test_monotonic_in_occlusion():
    lo = difficulty_score(_q(), {"occlusion": 0.1, "obj_motion_m": 0.1, "contact_richness": 0.5}, W)
    hi = difficulty_score(_q(), {"occlusion": 0.9, "obj_motion_m": 0.1, "contact_richness": 0.5}, W)
    assert hi >= lo

def test_richer_contact_not_harder():
    poor = difficulty_score(_q(), {"occlusion": 0.5, "obj_motion_m": 0.2, "contact_richness": 0.0}, W)
    rich = difficulty_score(_q(), {"occlusion": 0.5, "obj_motion_m": 0.2, "contact_richness": 1.0}, W)
    assert rich <= poor
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest egoaero/tests/dataset/test_difficulty.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write minimal implementation**

```python
# egoaero/egoaero/dataset/difficulty.py
"""Heuristic difficulty score (1..5) for an EgoDex-R sequence. The paper uses an
MLLM-based evaluator (App F); this is a documented deterministic substitute combining
occlusion, object motion, residual quality, and (inversely) contact richness."""
from __future__ import annotations


def difficulty_score(quality_report, recon_summary, weights):
    occ = float(recon_summary.get("occlusion", 0.0))                 # [0,1]
    motion = min(float(recon_summary.get("obj_motion_m", 0.0)) / 0.5, 1.0)  # normalize ~0.5 m
    contact = float(recon_summary.get("contact_richness", 0.0))      # [0,1]
    resid = min(float(quality_report.get("R_after", 0.0)) / 3.0, 1.0)       # normalize ~3.0
    unresolved = float(quality_report.get("U_unresolved", 0.0))      # [0,1]
    hard = (weights["w_occlusion"] * occ
            + weights["w_motion"] * motion
            + 0.5 * weights["w_residual"] * (resid + unresolved)
            - weights["w_contact"] * contact)
    # max possible "hard" (contact=0): w_occ + w_motion + w_residual ; normalize to [0,1]
    denom = weights["w_occlusion"] + weights["w_motion"] + weights["w_residual"]
    frac = max(0.0, min(hard / denom, 1.0)) if denom > 0 else 0.0
    return int(round(1 + 4 * frac))                                  # 1..5
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest egoaero/tests/dataset/test_difficulty.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add egoaero/egoaero/dataset/difficulty.py egoaero/tests/dataset/test_difficulty.py
git commit -m "egoaero: SP4 heuristic difficulty score (1-5)"
```

---

### Task 3: `schema.py` — EgoDex-R sequence record (App F)

**Files:**
- Create: `egoaero/egoaero/dataset/schema.py`
- Test: `egoaero/tests/dataset/test_schema.py`

**Interfaces:**
- Produces:
  - `write_sequence(dataset_dir, seq_id, run_dir, metadata) -> dict` — creates `<dataset_dir>/<seq_id>/` with: a copy of `<run>/contract/` files (hand_mano.npz, object_traj.npz, object_mesh.obj, contact.npz), `quality.json` (copied), `raw_obs.npz` (depth, obj_mask, hand_mask, intrinsics, cam_traj from the stage0 bundle + `timestamps`), `metadata.json` (the passed metadata), and `manifest.json` listing files. Returns the manifest.
  - `validate_sequence(dataset_dir, seq_id) -> bool` — all required files present + metadata has keys `{task_description, manipulated_object, relational_objects, difficulty, decision, frames, seq_id}`.
  - `read_metadata(dataset_dir, seq_id) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# egoaero/tests/dataset/test_schema.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest egoaero/tests/dataset/test_schema.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write minimal implementation**

```python
# egoaero/egoaero/dataset/schema.py
"""EgoDex-R per-sequence record (App F): assemble raw observations, reconstructed
hand-object states, quality diagnostics, and task/difficulty metadata into one
sequence directory, with a writer / validator / reader."""
from __future__ import annotations
import json, os, shutil
import numpy as np

_CONTRACT_FILES = ["hand_mano.npz", "object_traj.npz", "object_mesh.obj", "contact.npz"]
_META_KEYS = {"task_description", "manipulated_object", "relational_objects",
              "difficulty", "decision", "frames", "seq_id"}


def _load_npz(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def write_sequence(dataset_dir, seq_id, run_dir, metadata):
    d = os.path.join(dataset_dir, seq_id)
    os.makedirs(d, exist_ok=True)
    contract_dir = os.path.join(run_dir, "contract")
    for fn in _CONTRACT_FILES:
        shutil.copyfile(os.path.join(contract_dir, fn), os.path.join(d, fn))
    # quality diagnostics
    shutil.copyfile(os.path.join(run_dir, "quality.json"), os.path.join(d, "quality.json"))
    # raw observations from the stage0 bundle
    s0 = _load_npz(os.path.join(run_dir, "stage0_ego_io", "arrays.npz"))
    with open(os.path.join(run_dir, "stage0_ego_io", "meta.json")) as f:
        s0_meta = json.load(f)
    fps = float(s0_meta.get("fps", 30.0)); T = int(s0_meta.get("T", s0["depth"].shape[0]))
    np.savez(os.path.join(d, "raw_obs.npz"),
             depth=s0["depth"], obj_mask=s0["obj_mask"], hand_mask=s0["hand_mask"],
             intrinsics=s0["intrinsics"], cam_traj=s0["cam_traj"],
             timestamps=np.arange(T) / fps)
    with open(os.path.join(d, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    manifest = {"hand_mano": "hand_mano.npz", "object_traj": "object_traj.npz",
                "object_mesh": "object_mesh.obj", "contact": "contact.npz",
                "quality": "quality.json", "raw_obs": "raw_obs.npz",
                "metadata": "metadata.json", "frames": int(metadata.get("frames", T))}
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def validate_sequence(dataset_dir, seq_id) -> bool:
    d = os.path.join(dataset_dir, seq_id)
    need = _CONTRACT_FILES + ["quality.json", "raw_obs.npz", "metadata.json", "manifest.json"]
    if not all(os.path.exists(os.path.join(d, n)) for n in need):
        return False
    try:
        with open(os.path.join(d, "metadata.json")) as f:
            meta = json.load(f)
    except Exception:
        return False
    return _META_KEYS.issubset(meta.keys())


def read_metadata(dataset_dir, seq_id) -> dict:
    with open(os.path.join(dataset_dir, seq_id, "metadata.json")) as f:
        return json.load(f)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest egoaero/tests/dataset/test_schema.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add egoaero/egoaero/dataset/schema.py egoaero/tests/dataset/test_schema.py
git commit -m "egoaero: SP4 EgoDex-R sequence schema (writer/validator/reader)"
```

---

### Task 4: `capture.py` — synthetic capture source

**Files:**
- Create: `egoaero/egoaero/dataset/capture.py`
- Test: `egoaero/tests/dataset/test_capture.py`

**Interfaces:**
- Produces: `synthetic_source(n, seed, num_frames, tightness_min, tightness_max) -> list[dict]` — `n` clip configs, each `{"seed": int, "num_frames": int, "mock_tightness": float, "task_description": str, "manipulated_object": str, "relational_objects": list}`, with `mock_tightness` swept linearly across `[tightness_min, tightness_max]` so quality (and decisions) genuinely span the range. `clip_overrides(cfg_dict) -> dict` returns the `load_config` overrides for a clip config.

- [ ] **Step 1: Write the failing test**

```python
# egoaero/tests/dataset/test_capture.py
from egoaero.dataset.capture import synthetic_source, clip_overrides

def test_source_yields_n_with_tightness_spread():
    clips = synthetic_source(n=5, seed=0, num_frames=24, tightness_min=0.0, tightness_max=1.0)
    assert len(clips) == 5
    ts = [c["mock_tightness"] for c in clips]
    assert min(ts) == 0.0 and max(ts) == 1.0 and ts == sorted(ts)
    assert all("task_description" in c and "seed" in c for c in clips)

def test_clip_overrides_keys():
    c = synthetic_source(2, 0, 16, 0.0, 1.0)[1]
    ov = clip_overrides(c)
    assert ov["mock_tightness"] == c["mock_tightness"]
    assert ov["num_frames"] == c["num_frames"] and ov["seed"] == c["seed"]
    assert ov["mock"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest egoaero/tests/dataset/test_capture.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write minimal implementation**

```python
# egoaero/egoaero/dataset/capture.py
"""Synthetic capture source — a documented stand-in for FastUMI-Ego capture. Yields
mock ego-clip configs whose grasp tightness sweeps a range, so the collection loop
sees a genuine spread of reconstruction quality (and thus accept/repair/recapture
decisions) WITHOUT changing the SP3 thresholds."""
from __future__ import annotations

_TASKS = ["pick up the object", "move the object", "place the object",
          "lift and hold the object", "grasp and relocate the object"]


def synthetic_source(n, seed, num_frames, tightness_min, tightness_max):
    n = int(n)
    clips = []
    for i in range(n):
        frac = i / (n - 1) if n > 1 else 1.0
        tightness = tightness_min + frac * (tightness_max - tightness_min)
        clips.append({
            "seed": int(seed) + i,
            "num_frames": int(num_frames),
            "mock_tightness": float(tightness),
            "task_description": _TASKS[i % len(_TASKS)],
            "manipulated_object": "object",
            "relational_objects": ["table"],
        })
    return clips


def clip_overrides(clip_cfg):
    return {"mock": True, "seed": clip_cfg["seed"],
            "num_frames": clip_cfg["num_frames"],
            "mock_tightness": clip_cfg["mock_tightness"]}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest egoaero/tests/dataset/test_capture.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add egoaero/egoaero/dataset/capture.py egoaero/tests/dataset/test_capture.py
git commit -m "egoaero: SP4 synthetic capture source (tightness-swept clips)"
```

---

### Task 5: `collect.py` — the Sec-3 closed loop

**Files:**
- Create: `egoaero/egoaero/dataset/collect.py`
- Test: `egoaero/tests/dataset/test_collect.py`

**Interfaces:**
- Consumes: `egoaero.config.load_config`, `egoaero.pipeline.run_pipeline`, `egoaero.bundle.Bundle`, `dataset.schema`, `dataset.difficulty`, `dataset.capture`.
- Produces: `run_collection(out_dir, n_target, dataset_cfg, seed=0, work_root=None) -> dict` (summary). Loops the synthetic source; per clip: run pipeline → read `quality.json` decision → compute difficulty → if `accept`/`repairable_accept` write the sequence (`seq_%04d`) and count it, else (`recapture`) skip; stop at `n_target` accepted or `max_attempts`. Writes `<out_dir>/summary.json` and returns it. Summary keys: `n_accepted`, `n_attempts`, `decisions` (counts per label), `difficulty_hist` (1..5 counts), `capabilities` (`{obj_state, asset_free, depth, slam, contact_eval}` all True), `total_frames`.
  - `recon_summary(run_dir) -> dict` helper: `{"occlusion","obj_motion_m","contact_richness"}` from stage0 masks + object_traj + quality per-finger Q_rec.

- [ ] **Step 1: Write the failing test**

```python
# egoaero/tests/dataset/test_collect.py
import json, os, yaml
from egoaero.dataset import schema
from egoaero.dataset.collect import run_collection

def _ds_cfg():
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(os.path.join(here, "egoaero", "configs", "dataset.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["collection"]["num_frames"] = 12
    cfg["collection"]["max_attempts"] = 6
    return cfg

def test_collection_builds_dataset(tmp_path):
    out = str(tmp_path / "egodexr")
    summary = run_collection(out, n_target=2, dataset_cfg=_ds_cfg(), seed=0,
                             work_root=str(tmp_path / "work"))
    assert os.path.exists(os.path.join(out, "summary.json"))
    # decisions counts sum to attempts; accepted <= attempts
    assert sum(summary["decisions"].values()) == summary["n_attempts"]
    assert summary["n_accepted"] <= summary["n_attempts"]
    assert summary["capabilities"]["contact_eval"] is True
    # every written sequence validates
    for sid in os.listdir(out):
        if sid.startswith("seq_"):
            assert schema.validate_sequence(out, sid)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest egoaero/tests/dataset/test_collect.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write minimal implementation**

```python
# egoaero/egoaero/dataset/collect.py
"""EgoAERO Sec-3 closed-loop data collection: reconstruct -> online quality assess ->
accept / repairable_accept / recapture, writing accepted sequences into a mock EgoDex-R."""
from __future__ import annotations
import json, os
import numpy as np

from ..config import load_config
from ..pipeline import run_pipeline
from . import schema, capture
from .difficulty import difficulty_score


def recon_summary(run_dir, quality_report):
    from ..bundle import Bundle
    s0 = Bundle.load(os.path.join(run_dir, "stage0_ego_io"))
    om = s0["obj_mask"]; hm = s0["hand_mask"]
    inter = (om & hm).reshape(om.shape[0], -1).sum(1)
    area = np.maximum(om.reshape(om.shape[0], -1).sum(1), 1)
    occlusion = float(np.mean(inter / area))
    with np.load(os.path.join(run_dir, "contract", "object_traj.npz")) as z:
        poses = z["obj_poses_t"]
    obj_motion_m = float(np.linalg.norm(poses[:, :3, 3] - poses[0, :3, 3], axis=1).max())
    qrec = [v["Q_rec"] for v in quality_report.get("per_finger", {}).values()]
    contact_richness = float(np.mean(qrec)) if qrec else 0.0
    return {"occlusion": occlusion, "obj_motion_m": obj_motion_m,
            "contact_richness": contact_richness}


def run_collection(out_dir, n_target, dataset_cfg, seed=0, work_root=None):
    os.makedirs(out_dir, exist_ok=True)
    work_root = work_root or os.path.join(out_dir, "_work")
    col = dataset_cfg["collection"]; cap = dataset_cfg["capture"]; dw = dataset_cfg["difficulty"]
    max_attempts = int(col["max_attempts"])
    clips = capture.synthetic_source(max_attempts, seed, int(col["num_frames"]),
                                     float(cap["tightness_min"]), float(cap["tightness_max"]))
    decisions = {"accept": 0, "repairable_accept": 0, "recapture": 0}
    diff_hist = {i: 0 for i in range(1, 6)}
    n_accepted = 0; n_attempts = 0; total_frames = 0
    for clip in clips:
        if n_accepted >= int(n_target):
            break
        n_attempts += 1
        run_dir = os.path.join(work_root, f"attempt_{n_attempts:04d}")
        cfg = load_config(overrides=capture.clip_overrides(clip))
        run_pipeline(cfg, run_dir, "all")
        with open(os.path.join(run_dir, "quality.json")) as f:
            q = json.load(f)
        decision = q["decision"]
        decisions[decision] = decisions.get(decision, 0) + 1
        if decision == "recapture":
            continue
        rs = recon_summary(run_dir, q)
        difficulty = difficulty_score(q, rs, dw)
        diff_hist[difficulty] = diff_hist.get(difficulty, 0) + 1
        seq_id = f"seq_{n_accepted:04d}"
        meta = {"task_description": clip["task_description"],
                "manipulated_object": clip["manipulated_object"],
                "relational_objects": clip["relational_objects"],
                "difficulty": difficulty, "decision": decision,
                "frames": int(clip["num_frames"]), "seq_id": seq_id}
        schema.write_sequence(out_dir, seq_id, run_dir, meta)
        n_accepted += 1; total_frames += int(clip["num_frames"])
    summary = {"n_accepted": n_accepted, "n_attempts": n_attempts, "decisions": decisions,
               "difficulty_hist": diff_hist, "total_frames": total_frames,
               "capabilities": {"obj_state": True, "asset_free": True, "depth": True,
                                "slam": True, "contact_eval": True}}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest egoaero/tests/dataset/test_collect.py -v`
Expected: PASS. NOTE: if the synthetic clips never reach `accept`/`repairable_accept` (so `n_accepted==0`), the test's per-sequence validation is vacuous but still passes; the controller will separately confirm a real spread. If you observe zero accepts even at `tightness=1.0`, increase the `0.05` press scale in `mock_scene` (Task 1) or the `tightness_max` so the tight clips genuinely achieve small post-repair contact gaps — document the value in ASSUMPTIONS.md. Do NOT change SP3 thresholds.

- [ ] **Step 5: Commit**

```bash
git add egoaero/egoaero/dataset/collect.py egoaero/tests/dataset/test_collect.py
git commit -m "egoaero: SP4 closed-loop collection (reconstruct->quality->accept/repair/recapture)"
```

---

### Task 6: `cli.py` (egoaero-collect), README, ASSUMPTIONS

**Files:**
- Create: `egoaero/egoaero/dataset/cli.py`
- Modify: `egoaero/pyproject.toml` (console script), `egoaero/README.md`, `egoaero/ASSUMPTIONS.md`
- Test: `egoaero/tests/dataset/test_cli_smoke.py`

**Interfaces:**
- Produces: `egoaero-collect --out <dir> --n <K> [--max-attempts M] [--seed S]` → `run_collection`, prints the summary JSON.

- [ ] **Step 1: Write the failing test**

```python
# egoaero/tests/dataset/test_cli_smoke.py
import json, os, subprocess, sys

def test_collect_cli(tmp_path):
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = str(tmp_path / "egodexr")
    r = subprocess.run([sys.executable, "-m", "egoaero.dataset.cli",
                        "--out", out, "--n", "1", "--max-attempts", "4"],
                       cwd=here, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert os.path.exists(os.path.join(out, "summary.json"))
    s = json.load(open(os.path.join(out, "summary.json")))
    assert "decisions" in s and s["n_attempts"] >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest egoaero/tests/dataset/test_cli_smoke.py -v`
Expected: FAIL (`No module named egoaero.dataset.cli`).

- [ ] **Step 3: Write minimal implementation**

```python
# egoaero/egoaero/dataset/cli.py
import argparse, json, os, sys, yaml
from .collect import run_collection


def _dataset_cfg():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "configs", "dataset.yaml")) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                 description="EgoAERO SP4 — EgoDex-R closed-loop collection")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--max-attempts", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = _dataset_cfg()
    n_target = a.n if a.n is not None else cfg["collection"]["n_target"]
    if a.max_attempts is not None:
        cfg["collection"]["max_attempts"] = a.max_attempts
    summary = run_collection(a.out, n_target, cfg, seed=a.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
```

Add to `egoaero/pyproject.toml` under `[project.scripts]`:
```toml
egoaero-collect = "egoaero.dataset.cli:main"
```

Update `egoaero/README.md`: add an SP4 "EgoDex-R dataset + collection loop" section — the closed loop (reconstruct→quality→accept/repairable/recapture), the App-F per-sequence schema fields, `egoaero-collect` usage, and an HONEST note (synthetic capture source, mock mini-dataset, not the paper's 4.3M frames / FastUMI-Ego; difficulty is a heuristic substitute for the MLLM). Update `egoaero/ASSUMPTIONS.md` with the SP4 entries (difficulty heuristic + weights, synthetic capture + tightness knob, task-description templating, dataset-scale substitution).

- [ ] **Step 4: Run tests**

Run: `python -m pytest egoaero/tests/dataset/test_cli_smoke.py -v` → PASS.
Run the full suite: `python -m pytest egoaero/tests/ -q` → all green (base + dataset; sim tests run since the stack is installed).
Manual: `cd egoaero && python -m egoaero.dataset.cli --out runs/egodexr_demo --n 3 --max-attempts 8` → prints a summary with decision counts (do NOT commit runs/ outputs).

- [ ] **Step 5: Commit**

```bash
git add egoaero/egoaero/dataset/cli.py egoaero/pyproject.toml egoaero/README.md egoaero/ASSUMPTIONS.md egoaero/tests/dataset/test_cli_smoke.py
git commit -m "egoaero: SP4 egoaero-collect CLI + README + ASSUMPTIONS"
```

---

## Self-Review

**Spec coverage (SP4 spec §2–§5 → tasks):**
- `mock_tightness` knob enabling a genuine quality spread (§2.3) → Task 1 ✓
- `dataset.yaml` config (§2) → Task 1 ✓
- `difficulty.py` heuristic 1–5 (§2.2) → Task 2 ✓
- `schema.py` write/validate/read, App-F fields (§2.1) → Task 3 ✓
- `capture.py` synthetic source (§2.3) → Task 4 ✓
- `collect.py` closed loop + summary + Table-1 flags (§2.4) → Task 5 ✓
- CLI + README + ASSUMPTIONS (§2.5, §6) → Task 6 ✓
- pure-numpy / no heavy deps / base suite green (§5, global constraints) → all dataset code is numpy + the numpy pipeline ✓
- faithfulness map (§3) → loop/schema/flags faithful; difficulty/capture/scale documented across Tasks 2,4,5,6 ✓

**Placeholder scan:** every step has complete code/commands. The one conditional ("if zero accepts, increase the press scale") in Task 5 is a concrete, bounded instruction with an exact knob to turn (the `0.05` in Task 1) — not a TBD.

**Type consistency:** `difficulty_score(quality_report, recon_summary, weights)` signature matches its callers (Task 2 test + collect.py Task 5). `synthetic_source(n, seed, num_frames, tightness_min, tightness_max)` and `clip_overrides(clip_cfg)` consistent across Tasks 4/5. `write_sequence(dataset_dir, seq_id, run_dir, metadata)` / `validate_sequence(dataset_dir, seq_id)` / `read_metadata` consistent across Tasks 3/5. `run_collection(out_dir, n_target, dataset_cfg, seed, work_root)` consistent across Tasks 5/6. Decision labels (`accept`/`repairable_accept`/`recapture`) match SP3's exact strings. `mock_tightness` config key consistent across config.py / stage0 / capture / mock_scene. quality.json keys (`decision`, `per_finger[*].Q_rec`, `R_after`, `U_unresolved`) match SP3's actual output.
