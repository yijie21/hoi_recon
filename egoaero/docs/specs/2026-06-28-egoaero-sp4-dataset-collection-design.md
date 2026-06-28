# EgoAERO reproduction — SP4: EgoDex-R dataset + collection loop (design)

Date: 2026-06-28
Status: approved (brainstorming) → implementation plan next
Method folder: `egoaero/` (self-contained method in the HOI-reconstruction workbench)

## 0. Context

Final sub-project of the full EgoAERO reproduction (paper at
`egoaero/egoaero.pdf_by_PaddleOCR-VL-1.6.md`). Build order **SP1 → SP3 → SP2 → SP4**.

- **SP1 — Asset-free reconstruction** (Sec 2.1 / App A–C): ✅ merged.
- **SP3 — Online quality assessment** (App E): ✅ merged.
- **SP2 — Two-stage residual RL policy** (App D): ✅ merged.
- **SP4 — EgoDex-R dataset + collection loop (Sec 3 / App F / Table 1)** ← *this spec*.

### Operating principle (unchanged)

> Faithful where the paper specifies; documented defaults/substitutions at gaps (logged in
> `egoaero/ASSUMPTIONS.md`); mock-runnable. No real captured data or hardware exists, so the
> dataset is built from synthetic clips — stated plainly.

## 1. Goal

Reproduce EgoAERO's **closed-loop data collection** (Sec 3) and the **EgoDex-R per-sequence
schema** (App F): for each captured demonstration, reconstruct (SP1), assess quality online
(SP3), and decide **accept / repairable_accept / recapture**; write accepted/repairable
sequences into a dataset with the full App-F field set + difficulty/metadata; emit a
Table-1-style dataset summary.

### Reuse

- **SP1 contract** (`<run>/contract/`): hand_mano, object_traj, object_mesh, contact + stage0
  raw-obs arrays (depth, masks, intrinsics, cam_traj) = the per-sequence geometric payload.
- **SP3 quality** (`<run>/quality.json` + stage8 `meta["quality"]`): the decision + per-finger
  diagnostics + the scalar quality terms.
- The full pipeline (`run_pipeline`, stages 0–8) is the per-clip reconstruction+assessment.

SP4 adds NO heavy deps — pure numpy, mock-runnable on the base box.

## 2. Architecture

```
egoaero/egoaero/dataset/
  __init__.py
  schema.py      EgoDex-R per-sequence record: writer + validator + reader
  difficulty.py  documented heuristic difficulty score (1..5) from quality diagnostics
  capture.py     synthetic capture source: yields mock ego clips with a contact-tightness knob
  collect.py     the Sec-3 closed loop: reconstruct -> quality -> decision -> write/skip
  cli.py         egoaero-collect entry point
egoaero/egoaero/configs/dataset.yaml   collection + difficulty + capture defaults (documented)
```

### 2.1 `schema.py` — EgoDex-R sequence record (App F)

`write_sequence(dataset_dir, seq_id, run_dir, metadata) -> dict`:
- Creates `<dataset_dir>/<seq_id>/` and assembles the App-F fields by copying/referencing the
  reconstruction run's outputs:
  - **raw observations:** `depth`, `obj_mask`, `hand_mask`, `intrinsics`, `cam_traj` (SLAM/camera
    poses in the table frame), `timestamps` — from the stage0 bundle.
  - **reconstructed hand-object states:** `hand_mano.npz` (MANO verts/joints; shape/pose params
    where available), `object_traj.npz` (6-DoF), `object_mesh.obj`, `contact.npz` (contact
    windows/maps) — from the SP1 contract.
  - **quality diagnostics:** the SP3 `quality.json` (per-frame/ per-finger diagnostics + scalar
    terms + decision).
  - **metadata.json:** `task_description`, `manipulated_object`, `relational_objects`,
    `difficulty` (1–5), `decision`, `frames`, `seq_id`.
  - a per-sequence `manifest.json` listing all written files.
- `validate_sequence(dataset_dir, seq_id) -> bool`: every required field/file present + metadata
  has the required keys.
- `read_metadata(dataset_dir, seq_id) -> dict`.

### 2.2 `difficulty.py`

`difficulty_score(quality_report, recon_summary) -> int` in `1..5` — documented heuristic
(paper uses an MLLM-based evaluator, App F). Combines, with documented weights: hand-object
occlusion level, object-motion magnitude, contact richness (fraction of active-contact frames),
and residual penetration / low recoverability. Higher = harder. Deterministic.

### 2.3 `capture.py`

`synthetic_source(n, seed, tightness_range) -> iterator of clip-configs`: yields mock ego-clip
configurations with a **contact-tightness knob** spanning easy↔hard, so the collection loop sees
a genuine spread of quality (and thus of decisions) — NOT by changing the SP3 thresholds, but by
generating clips whose hand-object contact really differs. Each config drives the mock pipeline
(`config.load_config(overrides=...)`) for one clip. The tightness knob maps to the synthetic
scene's hand-object clearance.

### 2.4 `collect.py` — the closed loop (Sec 3)

`run_collection(out_dir, n_target, cfg, seed=0) -> summary`:
- Iterate the synthetic capture source. For each clip:
  1. `run_pipeline` (stages 0–8) into a temp/working run dir; `contract.write`.
  2. Read the SP3 decision from `quality.json`.
  3. Compute `difficulty_score`; build metadata (task description from the clip config; relational
     objects e.g. `["table"]`).
  4. **accept** or **repairable_accept** → `schema.write_sequence(...)` into `out_dir`; increment
     accepted count. **recapture** → skip (count it), continue.
  5. Stop when `n_target` sequences are written or `max_attempts` reached.
- Write `<out_dir>/summary.json`: Table-1-style capability flags (`obj_state, asset_free, depth,
  slam, contact_eval = True`), per-decision counts, difficulty histogram, total frames, n_accepted.
- Returns the summary dict.

### 2.5 CLI

`egoaero-collect --out <dataset_dir> --n <K> [--max-attempts M] [--seed S]` → runs `run_collection`,
prints the summary.

## 3. Faithfulness map

- **Faithful:** the closed-loop accept/repairable_accept/recapture flow (Sec 3); the App-F schema
  field set; the Table-1 capability flags for EgoDex-R; "repairable" = keep after the
  already-applied stage6 contact repair; "recapture" = discard.
- **Documented defaults/substitutions:** difficulty heuristic (vs MLLM); synthetic capture source
  (vs FastUMI-Ego); task-description templating (vs human/MLLM authoring); MANO shape/pose params
  exposed only to the extent the mock provides; dataset scale (a few synthetic sequences vs 4.3M
  frames / 5,600 sequences). All in `ASSUMPTIONS.md`.

## 4. Honesty

SP4 builds a **mock mini EgoDex-R** from synthetic clips and reports the **real** decision counts.
Given the mock's contact geometry, the decision mix may skew toward recapture; the capture
tightness knob yields genuine accepts for a meaningful demonstration. No real captured data, no
4.3M frames — stated in the README and ASSUMPTIONS.

## 5. Testing (pure-numpy, base box; no heavy deps)

- `difficulty_score`: bounded to 1..5; monotonic in each input factor (harder occlusion/motion/
  penetration → higher; richer recoverable contact → not higher); deterministic.
- `schema.write_sequence`/`validate_sequence`: round-trip a sequence dir built from a real mock
  reconstruction run; validate True; missing-field → validate False; `read_metadata` returns the
  written metadata.
- `run_collection`: over a few synthetic clips, produces `<out_dir>/summary.json` with
  per-decision counts summing to attempts, `n_accepted ≤ attempts`, every written sequence
  validates, and capability flags all True.
- CLI smoke: `egoaero-collect --out tmp --n 1` builds a dataset dir + summary.
- Tests CWD-safe; no weights/data; no heavy deps (the pipeline + quality are pure numpy).

## 6. Deliverables

1. `egoaero/egoaero/dataset/` (schema, difficulty, capture, collect, cli) + `configs/dataset.yaml`.
2. `egoaero-collect` console script in `egoaero/pyproject.toml`.
3. `ASSUMPTIONS.md` entries for every default/substitution; `README.md` SP4 section (schema,
   closed loop, honest scale, a sample collection summary).
4. Pure-numpy unit + smoke tests; existing suite still green.

## 7. Completion

SP4 finishes the four-sub-project reproduction (SP1 reconstruction → SP3 quality → SP2 policy →
SP4 dataset/collection), an end-to-end faithful-where-specified EgoAERO reproduction runnable in
mock mode.
