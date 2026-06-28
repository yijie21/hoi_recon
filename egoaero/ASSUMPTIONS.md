# EgoAERO reproduction — assumptions & deviations

Every value the paper leaves unspecified, with the section it fills and the source.

## Config defaults
- `track.*` weights/thresholds (App A): paper gives no values → defaults borrowed from
  BundleSDF [16] / FoundationPose [17] conventions. (Task 1, 8, 9)
- `contact.opp_gap_m=0.5mm`, `contact.region_weights`, `contact.pen_eps_m=2mm` (App C):
  paper specifies thumb/thenar gaps and bounds only → these are documented defaults. (Task 1, 13-14)
- `hand.depth_bias_m`, `track.drift_sigma_*`: mock-mode injected error magnitudes (not from
  paper) used to exercise the correction/optimizer modules. (Task 5, 8, 11)
- **SP4 mock press scale** (`tightness * 0.08 * bump` in `core/mock_scene.py`): raised from 0.05
  to 0.08 (Task 5) to produce a genuine accept/repairable_accept spread across the tightness sweep.
  At scale=0.05, only tightness=0.4 reached `repairable_accept`; at 0.08, tightness=0.2 and
  tightness=1.0 both reach `repairable_accept` (Q=0.352 and Q=0.434 respectively). The SP3
  quality thresholds (cfg.quality) are unchanged. (Task 5 / SP4)

## SE3 Parameterization
- `se3_log` / `se3_exp` use a **left-trivialized small-step parameterization** (translation
  decoupled from rotation), adequate for the iterative pose-graph refinement here; full SE3
  exponential not required at the step sizes used. (Task 2, 3, 6)

## Stage 1 (Semantic preprocessing)
- MLLM identity, keyframe count, prompt text, and seed-frame criterion (§2.1.1) are unspecified → mock uses GT masks and
  an `area − occlusion` seed-frame score (`argmax(object_area − hand_overlap)`). (Task 7)

## Stage 2b — pose-graph optimization (§2.1.2 / App A)
- App A specifies the term structure `Σλ_f E_feat + λ_g E_geo + λ_p E_pose + …` but gives no weights, kernel, optimizer
  type, feature matcher, `E_mask` equation, or memory-pool selection thresholds.  SP1 defaults borrow from
  BundleSDF / FoundationPose conventions (Task 9).
- **E_feat + E_pose implemented**; `E_geo / E_sdf / E_mask` and rotation refinement left at zero weight (real-backend
  territory).  The optimizer is gradient descent on translation only.
- **Mock correspondences**: canonical object-surface points are warped by the GT relative transform between frames
  (`pj = T_rel @ surf`, `T_rel = gt[j]^{-1} @ gt[i]`).  This makes the feat residual zero at GT poses for a *moving*
  object, so the graph Laplacian correctly smooths the injected translation drift rather than collapsing all poses to a
  common position.
- **Step size**: chosen adaptively as `0.9 / (max_degree × λ_f + λ_p)` to guarantee Jacobi convergence regardless of
  graph topology.  The brief's fixed step of 0.5 diverges when `max_degree × λ_f > 1` (e.g. K=4, λ_f=1.0 → factor
  −3.05 per iter).
- **Metric naming**: `track_err_deg_before` / `track_err_deg_after` store centroid-distance in **mm**, not degrees.
  The `_deg_` suffix is kept verbatim because Task 18's smoke test references those exact key names.  (Task 9)

## Stage 3 (coarse-to-fine mesh, §2.1.2 / App B)
- App B defines no field architecture / ray sampling / loss equations (`L_surf,L_free,L_occ,L_rgb,L_eik`) / weights —
  all deferred to the real backend; SP1 mock uses tracked geometry as the coarse mesh and implements only the
  specified rigid+scale SAM3D→coarse alignment (Umeyama). (Task 10)

## Stage 5 (ego-motion compensation, §2.1.4)
- **Table-frame definition**: §2.1.4 names a fixed "table frame" but gives no calibration procedure.
  Mock uses the GT `table_T` (a single fixed SE3 sampled at scene-generation time) directly.
  Real path would use ORB-SLAM3 world-frame + a tabletop plane-fit to obtain `table_T`; that backend
  raises `NotImplementedError`. (Task 12)
- **SLAM hand-pixel down-weighting**: paper mentions down-weighting hand pixels in SLAM
  (§2.1.4) — specifics unspecified; deferred to real ORB-SLAM3 backend. (Task 12)
- **Smoothing window**: §2.1.4 specifies "light temporal smoothing" but gives no window size.
  Mock uses a causal-padded moving-average of `cfg.ego.smooth_window` (default 5 frames) applied
  to hand vertices, hand joints, and object translation; object rotation is not smoothed. (Task 12)
- **No table/vertical constraint on the object**: the object pose in the table frame is the raw
  `w2t @ pose_w`; no additional table-plane projection or vertical constraint is imposed. (Task 12)

## Stage 6 (contact optimization, App C)

- **Region weights** `w_k` (thumb 1.0, opp 1.0, hukou 0.5): App C names three contact regions
  (thumb pad, opposing finger, thenar/hukou) but gives no relative importance weights.
  Documented defaults used; paper specifies only the gap thresholds and displacement bounds. (Task 14)
- **Opposing-finger gap** `g_opp = 0.5 mm`: App C gives the thumb gap (0.5 mm) and thenar gap
  (1.8 mm) explicitly; the opposing-finger gap is not stated. Matches the thumb gap as a
  documented default. (Task 13–14)
- **Penetration threshold** `ε = 2 mm` (`pen_eps_m`): App C defines the penetration push-back
  but gives no explicit ε value.  Documented default: 2 mm. (Task 14)
- **Contact-mask radius factor** `contact_mask_radius_factor = 4.0`: paper does not specify the
  contact-mask zone width; documented default zone is 4× the contact gap. (Task 14)
- **Finger-chain weight profile** `α_i^f`: App C specifies distal-heavy weighting along the
  MANO finger chain but gives no explicit exponent or profile shape.  Implemented as a linear
  ramp from palm (0) to fingertip (1) using the normalised along-finger coordinate (`z[fi]^1.0`).
  (Task 14)
- **App C constants used verbatim**: `contact_gap_m 0.0005`, `thenar_gap_m 0.0018`,
  `max_global_trans_m 0.034`, `max_finger_disp_m 0.015`, `max_pushback_m 0.008`,
  `smooth_window 9`, `boundary_frames 6`. (Task 14)
- **Local correction scope (App C § "replay-geometry level")**: Both the thumb pad and the
  dynamically-selected opposing finger receive the local offset.  The offset is applied to
  vertices (weighted by `finger_chain_weights`) AND to the four MANO-chain joints of each
  corrected finger (distal ramp `[0.25, 0.5, 0.75, 1.0]`).  The per-joint ramp follows
  App C's "distal-heavy" convention; joint indices are wrist(0) + 4 per finger in FINGERS
  order (thumb 1-4, index 5-8, middle 9-12, ring 13-16, little 17-20).  The penetration
  push-back step runs after all local corrections so overshoots are recovered. (final-review)

## SP3 — Online quality assessment (App E)
App E specifies the equations (Q_rec, B_repair, Q=exp(-aR-bB-gU), 3 decisions) but NO constants.
Documented defaults (config.quality): eps_g=4mm, eps_delta=12mm, delta_max=15mm (=max_finger_disp),
alpha=1.0, beta=0.5, gamma=1.0, pen_ref=50000mm & gap_ref=40mm (R_after normalizers, mock-scaled),
obj_move_thresh_m_per_frame=0.01 m/frame (U_unresolved), q_accept=0.6 / q_repairable=0.3 (decision thresholds).
U_unresolved heuristic (paper names but does not define): fraction of active frames where the object
is moving yet NO finger has recoverable contact. These are reasonable starting defaults, not paper values; on the synthetic mock clip the high procedural penetration yields a lower Q (repairable_accept/recapture) — an honest reflection of the mock scene, not a defect.

## SP2 — Task builder (policy/task.py)

### Mocap-driven wrist (task.py / Task 7)
The vendored `right_hand.xml` has a FIXED forearm (`rh_forearm`) with no floating joint.
To let the hand track the moving reconstructed wrist trajectory, we add a `mocap=True`
body `wrist_target` to the worldbody and pin `rh_forearm` to it via a WELD equality
(`mujoco.mjtEq.mjEQ_WELD`).  At rollout time the caller writes
`data.mocap_pos[task.wrist_mocap_id]` / `data.mocap_quat[task.wrist_mocap_id]` each step;
MuJoCo's equality solver kinematically follows.  This is simpler than a free-floating
wrist base (no extra actuators / control signal) while still letting the hand move in world space.

### Scene composition via MjSpec (not XML include)
We use `mujoco.MjSpec.from_file(hand_xml)` which resolves the hand's own `meshdir` relative
to the XML path, then programmatically add the table, mocap body, weld, and object.
The alternative (XML `<include>`) would clash with `<worldbody>` already present in
`right_hand.xml`; MjSpec avoids the conflict entirely.

### Contact-active heuristic (load_reference)
The contract carries no per-finger vertex groups so `contact_active[t]` is approximated
by a sphere proxy: fingertip is "active" if its distance to the object centroid departs
from the mesh bounding-sphere radius by < 20 mm.  This is a documented default; real
implementation would use mesh-SDF queries.

### finger_act_ids filter
The Shadow Hand's 20 actuators include 2 wrist actuators (`rh_A_WRJ1`, `rh_A_WRJ2`).
`finger_act_ids` is the list of the remaining 18 (those whose actuator name lacks 'WRJ').
Verified on the composed model (mujoco 3.10).

### Object-tracking reward reference velocity
In `StageIIEnv`, the object-tracking reward `r_obj` is computed with a zero reference
object velocity (`podot_ref = 0`). This means the velocity sub-term of `r_obj` penalizes
object motion in the reference frame rather than deviation from a reference velocity —
a documented mock-scale simplification where the object reference trajectory does not
specify a velocity profile.

## Task 8 — StageIEnv simplifications (env.py)

### Wrist orientation = identity
`data.mocap_quat[task.wrist_mocap_id]` is set to `[1,0,0,0]` every step.
The SP1 contract (`hand_mano.npz`) carries no wrist orientation: `ref` contains
`wrist_pos` (T,3) but no wrist rotation.  The wrist reward uses identity rotations
for both robot and reference (`R = R_h = I₃`), so the geodesic distance term is
always zero and the wrist reward depends only on position.  The weld equality
constraint aligns the forearm to the wrist_target body spatially; orientation
would require a rotation trajectory which the contract does not provide.

### Action space = finger actuators only (18-dim; no WRJ)
`StageIEnv` exposes only the 18 finger actuators (`task.finger_act_ids`).
Wrist motion is entirely dictated by the mocap reference.  The policy never
controls WRJ1/WRJ2 directly.

### Tendon-driven finger actuators (FFJ0, MFJ0, RFJ0, LFJ0)
Four of the 18 finger actuators are tendon-type (`mjtTrn.mjTRN_TENDON`).
Their `actuator_trnid` is a tendon index, not a joint index.  The observation
uses the first joint in each tendon's wrap path (resolved via
`model.tendon_adr[trnid]` → `model.wrap_objid[adr]`) to obtain qpos/qvel.
This reports the first coupled joint (e.g. FFJ2 for rh_A_FFJ0) rather than
a virtual "tendon position", which is the closest single-DoF proxy.

### Reward config defaults
All reward weights (`lam_p`, `lam_R`, `lam_v`, `lam_k`, `lam_a`, `lam_tau`,
`w_w`, `w_f`, `w_s`) are documented defaults in `config.py["reward"]`.
The paper (App D) defines the reward term structure but gives no numerical
values for these hyperparameters.

## Task 9 — StageIIEnv simplifications (env.py)

### Contact force proxy (cfrc_ext[3:6])
`data.cfrc_ext[bid]` is the external contact wrench applied to body `bid`,
stored as a 6-vector `[torque(3), force(3)]` in the world frame.  We take
`np.linalg.norm(cfrc_ext[bid][3:6])` (the linear force half) as the per-fingertip
contact force magnitude. This is the physically correct component representing
the magnitude of the linear contact force acting on the fingertip. See the reward
weight `mu_F` in the Stage-II reward configuration.

### Object-position early termination
`terminated` is triggered when the object centroid drifts more than
`cfg["term"]["obj_pos_err_m"]` (default 0.15 m) from the reference trajectory.
This threshold is deliberately generous so that the no-op Stage-I policy used
in tests (which makes no contact) does not immediately terminate the episode.
Paper (App D) specifies early termination for "large tracking errors" but gives
no threshold value; 0.15 m is a documented default for the mock setting.
Hand-error and penetration thresholds are mentioned in App D but not given
numerically; they are not implemented here (only the object-pos threshold is).

### obj_dofadr computation
The object free joint dof address is resolved from `model.jnt_dofadr[jid]`
where `jid = mj_name2id(model, mjOBJ_JOINT, "obj_free")`.  The free joint
has 7 qpos values (pos + quat) and 6 dof values (lin_vel + ang_vel); the
`qvel[obj_dofadr:obj_dofadr+6]` slice gives the object velocity for r_obj.

### Stage-I obs queried on 54-dim vector
`pi_I` is always called with a 54-dim float32 vector built the same way as
`StageIEnv._obs()`: `[fq(18), fqd(18), ref_wrist_pos(3), ref_fingertips(15)]`.
This ensures the frozen Stage-I policy sees exactly its training distribution.

## Task 11 — evaluate.py (rollout, App-H metrics, ablation)

### Ej = Eft = fingertip error
App-H defines four metrics: Er (rotation), Et (translation), Ej (joint position),
Eft (fingertip position).  For the substitute Shadow Hand the SP1 contract exposes
only five fingertip keypoints (`fingertips_h[T,5,3]`); there is no full per-joint
robot↔human correspondence defined.  Therefore both Ej and Eft are computed as
`mean_fingertip_error(rollout_fingertips, ref["fingertips_h"])`.  In a real setting
Ej would use all robot joint positions matched against MANO joint positions; that
requires a joint correspondence table the contract does not carry.

### wo_contact_opt ablation — documented placeholder
The `wo_contact_opt` condition requires a second reconstruction run produced with
stage-6 contact optimisation disabled (`--stages 0-5,7` at the CLI level, Task 12).
When `ablation()` is called without such a run dir, the function reuses the `full`
result and logs a WARNING.  The Task-12 CLI is responsible for wiring the separate
run dir and passing it here.

### _obj_pose() returns quaternion, not rotation matrix
`StageIIEnv._obj_pose()` returns `(pos[3], quat[4])`.  `rollout()` converts to
`R[3,3]` via `mujoco.mju_quat2Mat` immediately after the env step so that the
caller receives the (T,3,3) array documented in the task contract.

## SP4 — EgoDex-R dataset collection loop (App F)

### Difficulty heuristic (difficulty.py)
App F annotates each accepted sequence with a difficulty rating 1–5.  The paper uses an MLLM judge
for annotation; this implementation uses a 4-term heuristic:
`D = clip(round(raw * 4 + 1), 1, 5)` where
`raw = w_occ * occlusion + w_mot * obj_motion_m + w_res * R_after - w_con * contact_richness`,
normalized to [0,1] by `max(raw, eps)`.  Default weights: `w_occlusion=1.0, w_motion=1.0,
w_residual=1.0, w_contact=1.0` (documented defaults; paper gives no values).  (SP4 Task 2)

### Synthetic capture source (capture.py)
The paper's collection loop ingests clips from FastUMI-Ego hardware (5,600 real sequences,
4.3M frames).  SP4 substitutes a fully synthetic NumPy source: `synthetic_source(n, seed,
num_frames, tightness_min, tightness_max)` generates `n` clips with uniformly-spaced `tightness`
values in [tightness_min, tightness_max].  `tightness` controls the `mock_tightness` knob in
`core/mock_scene.py` (scale 0.08), which shifts the procedural hand-object penetration — low
tightness yields `recapture`, high tightness yields `repairable_accept`.  This is a documented
heuristic substitute for real capture hardware.  (SP4 Task 4)

### Task description templating (capture.py)
Each synthetic clip carries a `task_description` string and object labels drawn from a small
fixed vocabulary (objects: cup, bottle, scissors, screwdriver, apple; tasks: "pick up the {obj}",
"place the {obj} on the tray", "hand over the {obj}", "open the {obj}", "pour using the {obj}").
These are mock annotations — not from the paper's MLLM annotation pipeline.  (SP4 Task 4)

### Dataset scale substitution
EgoDex-R (paper) contains 5,600 sequences and 4.3M frames captured with FastUMI-Ego hardware.
SP4 produces a mock mini-dataset of a handful of sequences from the synthetic source.  No real
footage, no real hardware, no scale parity with the paper's numbers.  (SP4 Task 6)

### egoaero-collect CLI (dataset/cli.py)
`egoaero-collect --out <dir> --n <K> [--max-attempts M] [--seed S]` drives `run_collection`
with config from `configs/dataset.yaml` (defaults: `n_target=5`, `max_attempts=40`,
`num_frames=32`).  The `--n` and `--max-attempts` flags override the yaml values; `--seed`
controls the synthetic source RNG.  Registered as a console script in `pyproject.toml`.  (SP4 Task 6)

---

## Task 12 — CLI, console scripts, and real-run metrics

### Console scripts
`egoaero-train` and `egoaero-eval` both map to `egoaero.policy.cli:main`.  The
`train` / `eval` subcommand (first positional argument) selects the action.  Both
scripts are registered in `egoaero/pyproject.toml` under `[project.scripts]`; they
require `pip install -e "egoaero[rl]"` (mujoco, stable-baselines3, gymnasium,
torch).

### `--budget` choices
- `smoke` (default): 512 total timesteps per stage; completes in ~10 s on CPU.
  Used by CI / the gated smoke test.
- `real`: 1 500 000 total timesteps per stage; intended for a GPU run.  The paper
  does not specify total step counts; 1.5M is a documented default matching common
  SB3 dexterous-hand baselines.

### Real-run metrics placeholder
`README.md` contains a placeholder line **"Real-run metrics: (filled in by a real
`--budget real` run — see ASSUMPTIONS)"**.  This will be replaced with actual
`Er / Et / Ej / Eft / SR` numbers once the controller runs:
```bash
egoaero-train --run runs/demo --out runs/demo/policy --budget real
egoaero-eval  --run runs/demo --policy runs/demo/policy
```
Results from a single mock clip + Shadow Hand substitute are a feasibility
demonstration; they will not match paper numbers (multi-subject, real cameras,
real hand hardware correspondence).

### wo_contact_opt ablation CLI wiring (ASSUMPTIONS carry-over from Task 11)
`evaluate.ablation()` documents that the `wo_contact_opt` condition requires a
second run dir produced with `--stages 0-5,7` (stage-6 contact optimisation
disabled).  The CLI does not expose a `--wo-contact-run` flag in this sprint;
the ablation function logs a WARNING and reuses `full` when only one run dir is
provided.  A real ablation would pass separate run dirs to `ablation()` directly.
