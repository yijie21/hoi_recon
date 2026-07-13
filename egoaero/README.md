# egoaero — EgoAERO Part A reconstruction

A self-contained reimplementation of Part A (Section 2.1) of the **EgoAERO** paper: given a
monocular egocentric RGB-D video of a hand using an object, reconstruct the per-frame **MANO**
hand pose, the object's 6-DoF trajectory, and the hand-object contact — with no pre-scanned
object model ("asset-free"). It runs fully in **mock mode** today (synthetic data, no camera or
weights needed); real-backend hooks exist but raise `NotImplementedError` where the paper's
real-world components aren't wired up yet. The 8-stage pipeline below follows the paper's own
section structure and Appendices A–C.

This is a separate package from the rest of the repo — see the top-level
[`README.md`](../README.md) and [`GLOSSARY.md`](../GLOSSARY.md) for the main HOT3D pipeline and
benchmark; those terms don't apply here.

## The 9 stages

| # | Name | Matches the paper | Notes |
|---|------|--------------------|-------|
| 0 | `stage0_ego_io` | yes | Loads/generates ego frames, depth, GT scene (mock) |
| 1 | `stage1_semantic` | yes, documented default | MLLM prompt + seed-frame selection (§2.1.1); mock uses GT masks + an area–occlusion score |
| 2 | `stage2_track` | yes, documented default | RANSAC coarse init + memory-pool pose-graph optimization (App A); weights default to BundleSDF/FoundationPose settings |
| 3 | `stage3_mesh` | yes, documented default | Coarse-to-fine neural field (App B); mock uses tracked geometry + Umeyama alignment |
| 4 | `stage4_hand` | yes, documented default | HaWoR MANO fit + depth-residual global-translation correction (§2.1.3) |
| 5 | `stage5_ego_comp` | yes, documented default | Ego-motion compensation to table frame + temporal smoothing (§2.1.4) |
| 6 | `stage6_contact` | yes, faithful | Adaptive contact optimization (App C): constants taken verbatim from the paper, three-region iterative push-back |
| 7 | `stage7_eval` | yes | Before/after penetration, contact-gap, hand-jitter report |
| 8 | `stage8_quality` | yes, documented default | Online quality assessment (App E): accept / repairable_accept / recapture verdict |

"Documented default" means the paper specifies the algorithm's structure but not its exact
numeric constants; the values used here are logged in [`ASSUMPTIONS.md`](ASSUMPTIONS.md).

## Quickstart

```bash
# from egoaero/ directory (no install required)
python -m egoaero.cli --out runs/demo --mock

# run only specific stages (e.g. stages 0-3)
python -m egoaero.cli --out runs/demo2 --mock --stages 0-3

# override config
python -m egoaero.cli --out runs/demo3 --mock --set num_frames=32 seed=42
```

Results land in `runs/<name>/`:
- `stage0_ego_io/`, `stage1_semantic/`, … `stage8_quality/` — per-stage output
- `quality.json` — the online quality-assessment verdict (see below)
- `contract/` — the workbench contract output (see below)
- `config.yaml` — the frozen run config

## Online quality assessment

Stage 8 (`stage8_quality`) scores the reconstructed clip per **Appendix E** of the paper and
emits one of three verdicts:

| Decision | Meaning |
|----------|---------|
| `accept` | Q ≥ 0.6 — reconstruction quality meets the bar |
| `repairable_accept` | 0.3 ≤ Q < 0.6 — marginal, but recoverable |
| `recapture` | Q < 0.3, or a global failure — quality too low; recommend re-capturing |

The overall score `Q = exp(-α R_after - β B_repair - γ U_unresolved)` combines residual
penetration/gap after contact optimization (`R_after`), how much of the finger-correction budget
was used (`B_repair`), and the fraction of unresolved frames (`U_unresolved`).

Written to `quality.json` in the run directory, alongside `contract/`. This stage is
supplementary — the 4D-HOI contract output doesn't change based on the verdict.

**Note:** on the synthetic mock clip, the high procedural penetration produces a lower Q score
(`repairable_accept` or `recapture` is expected, not `accept`) — that's an honest reflection of
the mock scene, not a bug.

All threshold defaults (`eps_g_m`, `eps_delta_m`, `q_accept`, `q_repairable`,
`obj_move_thresh_m_per_frame`, etc.) are documented in [`ASSUMPTIONS.md`](ASSUMPTIONS.md) under
"SP3 — Online quality assessment".

## Contract output layout

After a full pipeline run, `contract/` holds:

```
contract/
  manifest.json        # lists all files + frame count
  hand_mano.npz        # verts (T, 778, 3) + joints (T, 21, 3) in table frame
  object_traj.npz      # obj_poses_t (T, 4, 4) SE3 trajectory in table frame
  object_mesh.obj      # reconstructed object mesh (OBJ, 1-indexed)
  contact.npz          # contact_mask (T, 778) binary per-vertex contact map
```

Validated by `egoaero.contract.validate(run_dir)` → `True` when all five files are present.

## Interactive 4D-HOI viewer (viser)

Play a run's reconstructed hand-object interaction in the browser — object + hand over a
timeline, per-frame contact highlighting, contact lines, live contact/gap readouts:

```bash
scripts/view_demo.sh                       # mock run (auto-generated)
scripts/view_demo.sh runs/egodexr/seq_0000 # a collected dataset sequence
# or: python -m egoaero.viz.viser_app --run <run_dir> [--stage stage6_contact]   (egoaero-view)
```

Needs `viser` (`pip install -e .[viz]`; already present in the `forehoi` env). Serves on port
8080 — SSH-forward it to view locally. It can also load a self-contained scene `.npz`
(canonical-mesh or per-frame point-cloud object); this is how the real `wild6.mp4`
reconstruction (WiLoR MANO hand + bottle proxy) is viewed. See `viz_output/README.md` for the
visual gallery and the wild6 demo.

## Shared evaluation metrics

The workbench contract requires reporting:

- hand MPJPE (mean per-joint position error, mm) and jitter
- object translation error (mm)
- penetration depth (mm)
- contact F1 and contact-frame gap (mm)

Stage 7 prints a before/after table for penetration, contact gap, and hand jitter. Full MPJPE
and contact-F1 need real ground-truth annotations (planned for SP3 — see the roadmap below).

## Further reading

- [`ASSUMPTIONS.md`](ASSUMPTIONS.md) — every unspecified value: what was assumed, why, and where
- [`docs/specs/2026-06-27-egoaero-sp1-reconstruction-design.md`](docs/specs/2026-06-27-egoaero-sp1-reconstruction-design.md) — SP1 design spec
- [`docs/plans/2026-06-27-egoaero-sp1-reconstruction-plan.md`](docs/plans/2026-06-27-egoaero-sp1-reconstruction-plan.md) — SP1 implementation plan

---

## SP2 — Two-stage residual RL policy

SP2 implements the **two-stage residual policy** from **Appendix D** of the EgoAERO paper, using
**MuJoCo 3** + **Stable-Baselines3 PPO** (reinforcement learning) with a **Shadow Hand** (right)
as a stand-in for the dexterous robot hand.

### Design

Two stages, both trained with PPO (`egoaero/egoaero/policy/`):

| Stage | Env | Task | Reward |
|-------|-----|------|--------|
| I — `pi_I` | `StageIEnv` | Wrist tracking + finger keypoint matching | App D: wrist-position, finger-keypoint, action-smoothness, power terms |
| II — `pi_R` (residual) | `StageIIEnv` | Contact + object tracking on top of frozen `pi_I` | App D: Stage-I terms + object-pose, contact-distance, contact-force, residual-regularization |

**Mocap-driven wrist.** The Shadow Hand's forearm is kinematically welded to a MuJoCo `mocap`
body (`wrist_target`). At rollout time the wrist trajectory from the SP1 contract drives
`data.mocap_pos` / `data.mocap_quat`; the 18 finger actuators are the only policy outputs. This
avoids adding a floating 6-DoF wrist base while still letting the hand follow the reconstructed
trajectory.

**App-D rewards** (`policy/rewards.py`): Stage I rewards wrist position/orientation, fingertip
keypoints, action smoothness, and actuator power. Stage II adds object pose (rotation +
translation), contact distance, contact-force saturation, and residual regularization. All term
weights are documented defaults — App D specifies the structure but not the numbers (see
[`ASSUMPTIONS.md`](ASSUMPTIONS.md)).

**App-H metrics** (`policy/metrics.py`): object rotation error Er (geodesic, deg), object
translation error Et (cm), fingertip error Ej / Eft (cm), success rate SR. Reported by
`egoaero-eval` after rollout.

### Usage

**Install the RL extras first:**
```bash
pip install -e "egoaero[rl]"   # mujoco>=3, stable-baselines3>=2, gymnasium>=1, torch
```

**Quick smoke run** (512 steps/stage, ~10 s on CPU):
```bash
# build a mock reconstruction run
python -m egoaero.cli --out runs/demo --mock

# train (smoke budget)
egoaero-train --run runs/demo --out runs/demo/policy --budget smoke

# evaluate
egoaero-eval --run runs/demo --policy runs/demo/policy
```

**Production GPU run** (1.5M steps/stage):
```bash
egoaero-train --run runs/demo --out runs/demo/policy --budget real
egoaero-eval  --run runs/demo --policy runs/demo/policy
```

`egoaero-train` and `egoaero-eval` are console scripts that call `egoaero.policy.cli:main` with
the `train` / `eval` subcommand. You can also invoke directly without installing:
```bash
python -m egoaero.policy.cli train  --run <run_dir> --out <pol_dir> --budget smoke
python -m egoaero.policy.cli eval   --run <run_dir> --policy <pol_dir>
```

### Budgets

| Budget | Steps per stage | Typical wall time | Purpose |
|--------|----------------|-------------------|---------|
| `smoke` | 512 | ~10 s CPU | CI / sanity check |
| `real`  | 1,500,000 | ~hours GPU | Feasibility demo |

### Honest scope

This is a single-clip, single-hand-substitute feasibility implementation, not a trained
manipulation result:
- The Shadow Hand is kinematically different from the human hand in the video. Fingertip
  keypoints (`Ej = Eft`) stand in for the paper's App-H joint error since there's no full
  robot↔MANO joint correspondence (see [`ASSUMPTIONS.md`](ASSUMPTIONS.md)).
- One clip won't match the paper's multi-subject, real-camera dataset numbers; the real-run
  metrics below are a feasibility demonstration only.

**Real-run metrics** (single mock clip, 64 frames; Shadow Hand substitute; mocap-driven wrist;
two-stage PPO, 40k steps/stage, CPU, ~107 s total):

| Setting | Er (deg) | Et (cm) | Ej (cm) | Eft (cm) | SR |
|---|---|---|---|---|---|
| Baseline (zero policy) | 59.2 | 5.9 | 39.5 | 39.5 | 0.0 |
| Trained (π_I + π_R) | 59.2 | 5.9 | 39.8 | 39.8 | 0.0 |

The pipeline runs end-to-end and trains in ~107 s, but at this demonstration budget — one
synthetic clip, substitute hand — the learned policy doesn't improve over the zero-action
baseline: object-tracking error is unchanged because the hand never establishes effective
contact, and success rate stays 0. That's consistent with the scope here (one clip, Shadow-Hand
substitute, mocap wrist, mock-scale rewards, a tiny budget vs. the paper's full Isaac-Gym
training over many sequences). The deliverable is a faithful, runnable reproduction of the
App-D/App-H machinery, not a trained manipulation result.

---

## SP4 — EgoDex-R dataset + collection loop

SP4 implements the **closed-loop dataset collection** from **Section 3 / Appendix F** of the
EgoAERO paper: a synthetic capture source drives the reconstruction pipeline, the online quality
assessor (SP3, stage 8) issues a decision, and accepted sequences are written into a mock
EgoDex-R dataset directory.

### Closed loop

```
synthetic_source (clip) -> run_pipeline -> stage8_quality ->
    accept          => write_sequence + increment n_accepted
    repairable_accept => write_sequence + increment n_accepted
    recapture       => discard, next attempt
```

The loop stops when `n_accepted >= n_target` or `max_attempts` is used up.

### App-F per-sequence schema

Each accepted sequence is written to `<out>/<seq_id>/`:

| File | Contents |
|------|----------|
| `metadata.json` | `seq_id`, `task_description`, `manipulated_object`, `relational_objects`, `difficulty` (1–5), `decision`, `frames` |
| `quality.json` | Full SP3 quality report: `decision`, `Q`, `R_after`, `B_repair`, `U_unresolved`, `per_finger` |
| `hand_mano.npz` | MANO verts `(T,778,3)` + joints `(T,21,3)` in table frame |
| `object_traj.npz` | Object SE3 trajectory `(T,4,4)` in table frame |
| `object_mesh.obj` | Reconstructed object mesh |
| `contact.npz` | Contact mask `(T,778)` |

A dataset-level `summary.json` is written at `<out>/summary.json`:
`n_accepted`, `n_attempts`, `decisions` (count per label), `difficulty_hist`, `total_frames`, `capabilities`.

### Usage

```bash
# collect 5 accepted sequences (default), up to 40 attempts
egoaero-collect --out runs/egodexr

# override n and max-attempts
egoaero-collect --out runs/egodexr --n 10 --max-attempts 80 --seed 7

# or invoke without install
python -m egoaero.dataset.cli --out runs/egodexr --n 3 --max-attempts 8
```

### Sample run (`--n 2 --max-attempts 6`)

Over 6 attempts the loop yielded 2 accepted sequences (2 `repairable_accept`, 4 `recapture`):

```json
{
  "n_accepted": 2,
  "n_attempts": 6,
  "decisions": {"accept": 0, "repairable_accept": 2, "recapture": 4},
  "difficulty_hist": {"1": 2, "2": 0, "3": 0, "4": 0, "5": 0},
  "total_frames": 24,
  "capabilities": {"obj_state": true, "asset_free": true, "depth": true, "slam": true, "contact_eval": true}
}
```

### Honest scope

This is a **mock reproduction** of the EgoDex-R collection loop:

- **Capture source**: fully synthetic NumPy clips, not FastUMI-Ego hardware or any real
  egocentric camera. The `tightness` parameter sweeps a `mock_tightness` knob in
  `core/mock_scene.py` (scale 0.08) that shifts procedural hand-object penetration, producing a
  genuine spread of quality decisions across the acceptance threshold.
- **Dataset scale**: each run collects a handful of mock sequences. The paper's real EgoDex-R
  dataset is 4.3M frames / 5,600 sequences captured with real FastUMI-Ego hardware — that scale
  and hardware are not reproduced here.
- **Difficulty rating**: a 5-term heuristic combining occlusion, object motion (normalized by
  0.5 m), residual quality `R_after` (normalized by 3.0), unresolved contact `U_unresolved`, and
  contact richness (inverse): `D = round(1 + 4 * frac)` where
  `frac = clip((w_occ·occ + w_mot·motion + 0.5·w_res·(residual + U_unresolved) - w_con·contact) / (w_occ + w_mot + w_res), 0, 1)`.
  The paper uses an MLLM judge for difficulty annotation; this is a documented heuristic
  substitute.
- **Task descriptions**: templated natural-language strings from a fixed vocabulary, not the
  paper's annotation pipeline.
- The `mock_tightness` scale of 0.08 was tuned to produce genuine `repairable_accept` outputs
  across the tightness range (lower tightness typically yields `recapture`). SP3's quality
  thresholds are unchanged (`q_accept=0.6`, `q_repairable=0.3`).

---

## SP2 / SP3 / SP4 roadmap

| Sprint | Topic | Status |
|--------|-------|--------|
| SP2 | RL contact policy (two-stage PPO, Shadow Hand, App-D rewards/App-H metrics) | Done |
| SP3 | Online quality assessment (App E accept/repairable/recapture) | Done |
| SP4 | EgoDex-R dataset collection loop (`egoaero-collect` CLI, mock mini-dataset) | Done |

---

## Environment

This method shares the repository workbench conda/pip environment. All runtime dependencies
(`numpy`, `scipy`, `pyyaml`, `trimesh`) are listed in [`pyproject.toml`](pyproject.toml) and are
available in the standard workbench env. No separate `environment.yml` is needed — install with:

```bash
pip install -e .   # from egoaero/ directory
```
