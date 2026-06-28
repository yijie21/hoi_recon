# egoaero — EgoAERO Part A (Sec 2.1) Reconstruction

Self-contained implementation of the **EgoAERO egocentric hand-object reconstruction
pipeline** (Part A, Section 2.1 of the EgoAERO paper).  Runs fully in **mock mode**
today (no weights, no real cameras); real-backend stubs are wired and raise
`NotImplementedError` where documented.

---

## What it reproduces

**EgoAERO Part A, Sec 2.1** — asset-free, egocentric 4D hand-object reconstruction:

> Given a monocular egocentric RGB-D stream, reconstruct the per-frame MANO hand,
> object 6-DoF trajectory, and contact maps — without any per-object asset or template.

The 8-stage pipeline follows the paper's section structure and Appendices A–C.

---

## The 9 stages

| # | Name | Faithful to paper | Notes |
|---|------|-------------------|-------|
| 0 | `stage0_ego_io` | ✅ | Loads/generates ego frames, depth, GT scene (mock) |
| 1 | `stage1_semantic` | ✅ documented-default | MLLM prompt + seed-frame selection (§2.1.1); mock uses GT masks + area–occlusion score |
| 2 | `stage2_track` | ✅ documented-default | RANSAC coarse init + memory-pool pose-graph opt (App A); weights from BundleSDF/FoundationPose defaults |
| 3 | `stage3_mesh` | ✅ documented-default | Coarse-to-fine neural field (App B); SP1 mock uses tracked geometry + Umeyama alignment |
| 4 | `stage4_hand` | ✅ documented-default | HaWoR MANO + depth-residual global-translation correction (§2.1.3) |
| 5 | `stage5_ego_comp` | ✅ documented-default | Ego-motion compensation to table frame + temporal smoothing (§2.1.4) |
| 6 | `stage6_contact` | ✅ **faithful** | Adaptive contact optimisation (App C): App C constants verbatim, three-region iterative push-back |
| 7 | `stage7_eval` | ✅ | Before/after penetration, contact-gap, hand-jitter report |
| 8 | `stage8_quality` | ✅ documented-default | Online quality assessment (App E): accept / repairable_accept / recapture verdict |

"Documented-default" means the paper specifies the algorithm structure but omits
numeric constants or hyperparameters; values used are logged in [`ASSUMPTIONS.md`](ASSUMPTIONS.md).

---

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
- `stage0_ego_io/`, `stage1_semantic/`, … `stage8_quality/` — per-stage bundles
- `quality.json` — online quality assessment verdict (see below)
- `contract/` — workbench contract output (see below)
- `config.yaml` — frozen run config

---

## Online quality assessment

Stage 8 (`stage8_quality`) scores the reconstructed clip according to **App E** of the EgoAERO paper and emits one of three verdicts:

| Decision | Meaning |
|----------|---------|
| `accept` | Q ≥ 0.6 — reconstruction quality meets the bar |
| `repairable_accept` | 0.3 ≤ Q < 0.6 — quality marginal; recoverability suggests it can be improved |
| `recapture` | Q < 0.3 or global failure — quality too low; re-capture recommended |

The overall score `Q = exp(-α R_after - β B_repair - γ U_unresolved)` combines residual
penetration/gap (`R_after`), finger correction budget used (`B_repair`), and fraction of
unresolved frames (`U_unresolved`).

Results are written to `quality.json` in the run directory (alongside `contract/`).
The stage is supplementary — the 4D-HOI contract output is unchanged regardless of verdict.

**Note:** on the synthetic mock clip, the high procedural penetration yields a lower Q score
(`repairable_accept` or `recapture` is expected — not `accept`). This is an honest reflection
of the mock scene, not a defect.

All threshold defaults (`eps_g_m`, `eps_delta_m`, `q_accept`, `q_repairable`,
`obj_move_thresh_m_per_frame`, etc.) are documented in [`ASSUMPTIONS.md`](ASSUMPTIONS.md) under
"SP3 — Online quality assessment".

---

## Contract output layout

After a successful full-pipeline run the `contract/` folder holds:

```
contract/
  manifest.json        # lists all files + frame count
  hand_mano.npz        # verts (T, 778, 3) + joints (T, 21, 3) in table frame
  object_traj.npz      # obj_poses_t (T, 4, 4) SE3 trajectory in table frame
  object_mesh.obj      # reconstructed object mesh (OBJ, 1-indexed)
  contact.npz          # contact_mask (T, 778) binary per-vertex contact map
```

Validated by `egoaero.contract.validate(run_dir)` → `True` when all five files are present.

---

## Shared evaluation metrics

The workbench contract requires reporting:

- hand MPJPE (mm) and jitter
- object translation error (mm)
- penetration depth (mm)
- contact F1 and contact-frame gap (mm)

Stage 7 prints a before/after table for penetration, contact gap, and hand jitter.
Full MPJPE and contact-F1 require real GT annotations (planned for SP3 — see roadmap).

---

## Further reading

- [`ASSUMPTIONS.md`](ASSUMPTIONS.md) — every unspecified value: what was assumed, why, and where
- [`docs/specs/2026-06-27-egoaero-sp1-reconstruction-design.md`](docs/specs/2026-06-27-egoaero-sp1-reconstruction-design.md) — SP1 design spec
- [`docs/plans/2026-06-27-egoaero-sp1-reconstruction-plan.md`](docs/plans/2026-06-27-egoaero-sp1-reconstruction-plan.md) — SP1 implementation plan

---

## SP2 — Two-stage residual RL policy

SP2 implements the **two-stage residual policy** described in Appendix D of the
EgoAERO paper, using **MuJoCo 3 + Stable-Baselines3 PPO** and the
**Shadow Hand** (right) as the dexterous hand substitute.

### Design

The policy stack has two stages, both trained with PPO (`egoaero/egoaero/policy/`):

| Stage | Env | Task | Reward |
|-------|-----|------|--------|
| I — `pi_I` | `StageIEnv` | Wrist tracking + finger keypoint matching | App D: wrist-position, finger-keypoint, action-smoothness, power terms |
| II — `pi_R` (residual) | `StageIIEnv` | Contact + object tracking on top of frozen `pi_I` | App D: Stage-I terms + object-pose, contact-distance, contact-force, residual-regularization |

**Mocap-driven wrist** — the Shadow Hand forearm is kinematically welded to a
MuJoCo `mocap` body (`wrist_target`).  At rollout time the wrist trajectory from
the SP1 contract drives `data.mocap_pos / data.mocap_quat`; the 18 finger
actuators are the only policy outputs.  This avoids adding a floating 6-DoF wrist
base while still letting the hand follow the reconstructed trajectory.

**App-D rewards** (`policy/rewards.py`) — Stage-I reward: wrist position/orientation,
fingertip keypoints, action smoothness, actuator power.  Stage-II adds: object pose
(rotation + translation), contact distance, contact force saturation, residual
regularization.  All term weights are documented defaults (App D gives structure but
no numbers; see [`ASSUMPTIONS.md`](ASSUMPTIONS.md)).

**App-H metrics** (`policy/metrics.py`) — object rotation error Er (geodesic, deg),
object translation error Et (cm), fingertip error Ej / Eft (cm), success rate SR.
Reported by `egoaero-eval` after rollout.

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

Both `egoaero-train` and `egoaero-eval` are console scripts that call
`egoaero.policy.cli:main` with the `train` / `eval` subcommand.  You can also
invoke directly without install:
```bash
python -m egoaero.policy.cli train  --run <run_dir> --out <pol_dir> --budget smoke
python -m egoaero.policy.cli eval   --run <run_dir> --policy <pol_dir>
```

### Budgets

| Budget | Steps per stage | Typical wall time | Purpose |
|--------|----------------|-------------------|---------|
| `smoke` | 512 | ~10 s CPU | CI / sanity check |
| `real`  | 1 500 000 | ~hours GPU | Feasibility demo |

### Honest scope

This is a single-clip, single-hand-substitute feasibility implementation:
- The Shadow Hand is a kinematically different device from the human hand in the
  video.  Fingertip keypoints (`Ej = Eft`) are used as a proxy for App-H joint
  error because there is no full robot ↔ MANO joint correspondence (see
  [`ASSUMPTIONS.md`](ASSUMPTIONS.md)).
- One-clip results will not match the paper's multi-subject, real-camera dataset
  numbers.  The real-run metrics below are a feasibility demonstration only.

**Real-run metrics: (filled in by a real `--budget real` run — see ASSUMPTIONS)**

---

## SP2 / SP3 / SP4 roadmap

| Sprint | Topic | Status |
|--------|-------|--------|
| SP2 | RL contact policy (two-stage PPO, Shadow Hand, App-D rewards/App-H metrics) | Done |
| SP3 | Online quality assessment (App E accept/repairable/recapture) | Done |
| SP4 | Dataset integration (Ego4D, EPIC-Kitchens egocentric clips) | Planned |

---

## Environment

This method shares the repository workbench conda/pip environment.  All runtime
dependencies (`numpy`, `scipy`, `pyyaml`, `trimesh`) are listed in
[`pyproject.toml`](pyproject.toml) and are available in the standard workbench env.
No separate `environment.yml` is needed — install with:

```bash
pip install -e .   # from egoaero/ directory
```
