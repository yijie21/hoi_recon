# EgoAERO reproduction — SP2: Two-stage residual RL policy (design)

Date: 2026-06-28
Status: approved (brainstorming) → implementation plan next
Method folder: `egoaero/` (self-contained method in the HOI-reconstruction workbench)

## 0. Context

Third sub-project of the full EgoAERO reproduction (paper at
`egoaero/egoaero.pdf_by_PaddleOCR-VL-1.6.md`). Build order **SP1 → SP3 → SP2 → SP4**.

- **SP1 — Asset-free reconstruction** (Sec 2.1 / App A–C): ✅ merged.
- **SP3 — Online quality assessment** (App E): ✅ merged.
- **SP2 — Two-stage residual RL policy (Sec 2.2 / App D, eval App G/H)** ← *this spec*. Depends on SP1.
- SP4 — EgoDex-R dataset + collection loop (App F).

### Operating principle (unchanged)

> Faithful where the paper specifies; principled documented defaults at every gap (logged in
> `egoaero/ASSUMPTIONS.md`). Scope chosen by the user: **real RL in an open sim** (not a mock).

## 1. Goal

Convert a single reconstructed hand-object trajectory (SP1 contract output) into an executable
dexterous manipulation policy via the EgoAERO **two-stage residual learning** scheme
(Sec 2.2): Stage I learns hand-trajectory tracking; Stage II learns a residual policy that
adds object + contact objectives. Evaluate with the App-H metrics (`Er, Et, Ej, Eft, SR`).

This is a **real** reproduction: genuine physics (MuJoCo), genuine RL (PPO), genuine training
on the GPU — using documented open substitutes for the paper's unavailable/unspecified pieces.

## 2. Chosen stack (feasibility-verified on this box: 2× RTX 4090, 48 CPU, 251 GB RAM)

- **Physics:** MuJoCo 3.x (`mujoco`, headless, contact forces) — documented substitute for Isaac Gym.
- **RL:** stable-baselines3 PPO + Gymnasium (battle-tested PPO; less hyperparameter guesswork).
- **Hand:** the **Shadow Hand** MJCF vendored from MuJoCo Menagerie (5 fingers ↔ MANO's 5-finger
  keypoint imitation) — documented substitute for the Inspire Hand. (Allegro = lighter 4-finger
  fallback, noted only.)
- **Object:** the reconstruction's `object_mesh.obj` as a MuJoCo free-body mesh geom; reference
  trajectory from the reconstruction's `object_traj`.

## 3. Architecture

A new self-contained `policy/` subsystem; pure-numpy reward/metric cores (unit-testable on the
base box) and heavy-dep sim/training modules (lazy-imported, gated tests).

```
egoaero/egoaero/policy/
  __init__.py
  rewards.py     App-D reward terms — PURE numpy, no heavy deps, unit-tested
  metrics.py     App-H metrics + SR — PURE numpy, unit-tested
  retarget.py    MANO wrist+fingertip reference -> Shadow-Hand joint targets (per-frame IK)
  hand_model.py  load vendored Shadow Hand MJCF; finger->actuator maps; fingertip site names
  task.py        build the MuJoCo scene (hand + object free body) + reference loader from contract
  env.py         Gymnasium envs: StageIEnv (reward r^I), StageIIEnv (residual on frozen pi_I, reward r^R)
  train.py       two-stage SB3-PPO driver: train pi_I -> freeze -> train pi_R; save policies
  evaluate.py    rollout a policy -> App-H metrics + SR; ablation harness (3 settings) -> report
egoaero/assets/shadow_hand/   vendored Menagerie Shadow Hand MJCF + meshes
egoaero/egoaero/configs/policy.yaml   reward weights, PPO hyperparams, budget, thresholds (documented)
CLI entry points: egoaero-train, egoaero-eval
```

Lazy imports: `mujoco`/`torch`/`stable_baselines3`/`gymnasium` are imported inside `task.py`/
`env.py`/`train.py`/`evaluate.py` only. `rewards.py` and `metrics.py` import only numpy, so the
rest of `egoaero` (and the 41-test base suite) imports and runs without the sim stack installed.

### 3.1 `rewards.py` (faithful App-D; pure numpy)

- `r_wrist(p, R, pdot, p_h, R_h, pdot_h, lam_p, lam_R, lam_v) -> float` = `exp(−λ_p‖p−p^H‖² − λ_R d_R(R,R^H)² − λ_v‖ṗ−ṗ^H‖²)`.
- `r_finger(x_kpts, x_kpts_h, lam_k) -> float` = mean over keypoints of `exp(−λ_k‖x−x^H‖²)`.
- `r_smooth(a, a_prev, torque, qdot, lam_a, lam_tau) -> float` = `exp(−λ_a‖a−a_prev‖² − λ_τ‖τ⊙q̇‖₁)`.
- `r_stage1(...) -> float` = `w_w r_wrist + w_f r_finger + w_s r_smooth`.
- `r_obj(p_o, R_o, podot, p_ref, R_ref, podot_ref, mu_p, mu_R, mu_v) -> float`.
- `r_contact(dists, forces, active_set, mu_d, mu_F) -> float` = mean over reference-active fingers
  of `exp(−μ_d d²)(1−exp(−μ_F‖F‖))`; returns 0.0 (term skipped) when `active_set` is empty.
- `r_res(delta_a, mu_delta) -> float` = `exp(−μ_Δ‖Δa‖²)`.
- `r_stage2(r1, r_obj_v, r_contact_v, r_res_v, eta_I, eta_o, eta_c, eta_delta) -> float`.

### 3.2 `metrics.py` (faithful App-H; pure numpy)

- `object_rotation_error(R_seq, R_ref_seq) -> float` (mean geodesic, **degrees**).
- `object_translation_error(p_seq, p_ref_seq) -> float` (mean L2, **cm**).
- `mean_joint_error(xj_seq, xj_ref_seq) -> float` (cm). `mean_fingertip_error(xf_seq, xf_ref_seq) -> float` (cm, |F|=5).
- `success(Er, Et, Ej, Eft, tau_r=30, tau_t=3, tau_j=8, tau_ft=6) -> bool`; `success_rate(list_of_metric_tuples, taus) -> float`.

### 3.3 `retarget.py`

`retarget(hand_ref, hand_model) -> joint_target_seq`: per frame, solve a small damped-least-squares
IK so the Shadow-Hand fingertip sites match the reconstructed MANO fingertip keypoints and the wrist
matches the reconstructed wrist pose. Provides the **warm-start** init for Stage I (per App D, the
retargeted trajectory only initializes training; the reward target stays the human reference).
Correspondence (which MANO keypoints ↔ which Shadow sites) and IK damping are documented defaults.

### 3.4 `task.py` + `env.py`

`task.py` builds the MuJoCo `MjModel`: the vendored Shadow Hand (position actuators) + the object as
a free-body mesh geom on a table plane; loads the reference (`hand_mano`, `object_traj`,
`contact` active-finger sets) from the SP1 contract dir.

`env.py` — two Gymnasium envs:
- `StageIEnv`: observation = robot hand state (joint pos/vel, wrist pose) + the current hand reference
  target; action = joint/wrist position targets (warm-started from retarget); reward = `r_stage1`.
  The object is present but tracking it is not required. Episode = clip length; no object termination.
- `StageIIEnv(pi_I)`: wraps the frozen Stage-I policy; action = residual `Δa` added to `a^I = π_I(s^I)`;
  observation adds object pose/vel, an object-geometry encoding (e.g. bounding-box + a few mesh stats),
  hand-object distance, and contact forces (MuJoCo `cfrc`); reward = `r_stage2`; early termination when
  object pose error / hand tracking error / non-contact penetration exceeds documented thresholds.

### 3.5 `train.py` + `evaluate.py` + CLI

`train.py`: `train_two_stage(task, cfg, budget) -> (pi_I, pi_R)` — SB3 PPO on `StageIEnv` for the
Stage-I budget, freeze, then SB3 PPO on `StageIIEnv(pi_I)` for the Stage-II budget; save both.

`evaluate.py`: `rollout(task, pi_I, pi_R) -> sim_trajectories`; `evaluate(...) -> {Er,Et,Ej,Eft,SR}`
over multiple rollout seeds; `ablation(task, cfg) -> {only_hand, wo_contact_opt, full}` producing a
Table-2-style report (Only-Hand = Stage-I policy alone; w/o-contact-opt = train on the
pre-contact-optimization reconstruction; full = the contact-optimized reconstruction).

CLI: `egoaero-train --run <recon_run_dir> --out <policy_dir> [--budget smoke|real]`,
`egoaero-eval --run <recon_run_dir> --policy <policy_dir>`.

## 4. Scope & success bar (honest)

- **One reconstructed clip → one MuJoCo task → real two-stage PPO.** Not the paper's 100-task
  benchmark; one hand substitute. Results will not match the paper's SR.
- **Dual budget:** `smoke` (a few hundred steps/stage — proves the loop end-to-end, used by gated
  tests, no convergence) and `real` (~1–3M steps on the GPU — produces genuine metrics). During
  implementation, run ONE real short training on the 4090 to confirm it learns; record the numbers
  in the SP2 README.
- **Success = the pipeline runs end-to-end and emits App-H metrics**, gated smoke test green, and one
  real run demonstrates non-trivial learning (Stage-I tracking error decreases over training).

## 5. Dependencies & test gating

- New deps in `egoaero/environment.yml` / `pyproject.toml` optional-extra `rl`:
  `mujoco>=3`, `torch` (CUDA build), `stable-baselines3>=2`, `gymnasium>=1`.
- `rewards.py`, `metrics.py`, `retarget.py` (numpy-only IK) → unit-tested on the base box.
- `task.py`, `env.py`, `train.py`, `evaluate.py` tests use `pytest.importorskip("mujoco")` /
  `("stable_baselines3")` so they are skipped (not failed) when the stack is absent — keeping the
  existing 41-test base suite green. A documented `scripts/setup_rl.sh` installs the stack.

## 6. Faithfulness map

- **Faithful:** all App-D reward terms (§3.1), App-H metrics + SR + thresholds (§3.2), the residual
  composition `a = a^I + Δa^R`, the two-stage structure, early termination, the ablation settings.
- **Documented defaults:** PPO + SB3 choice, network architecture, every weight (`w/λ/η/μ`), PPO
  hyperparameters, training budget, retargeting correspondence + IK damping, object-geometry encoding,
  early-termination thresholds, MuJoCo↔Isaac-Gym and Shadow↔Inspire substitutions. All in
  `ASSUMPTIONS.md`.

## 7. Testing

- **Pure (base box):** each reward term (zero-error→1.0; strictly decreasing in each error; `r_stage1`/
  `r_stage2` weighted sums; `r_contact` skipped on empty active set); each metric (formula + units;
  `success` true when all under thresholds, false when any over; `success_rate` fraction); retarget IK
  reduces fingertip error on a constructed target.
- **Gated (sim stack):** `StageIEnv`/`StageIIEnv` pass SB3 `check_env`; reset/step return valid
  obs/reward/termination; `train_two_stage` at `smoke` budget completes and saves two policies;
  `evaluate` returns `SR∈[0,1]` and the four errors; `ablation` emits the 3-setting report.

## 8. Deliverables

1. `egoaero/egoaero/policy/` (8 modules) + vendored Shadow Hand asset + `configs/policy.yaml`.
2. `egoaero-train` / `egoaero-eval` CLIs; `scripts/setup_rl.sh`.
3. `ASSUMPTIONS.md` entries for every default/substitution; `README.md` SP2 section incl. the real-run
   numbers.
4. Pure unit tests (green on base box) + gated sim tests; existing 41-test suite still green.

## 9. Deferred

SP4 (EgoDex-R dataset schema + the accept/repair/recapture collection loop, App F), which will reuse
SP3's quality decision and SP1's reconstruction.
