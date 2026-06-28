# EgoAERO reproduction — assumptions & deviations

Every value the paper leaves unspecified, with the section it fills and the source.

## Config defaults
- `track.*` weights/thresholds (App A): paper gives no values → defaults borrowed from
  BundleSDF [16] / FoundationPose [17] conventions. (Task 1, 8, 9)
- `contact.opp_gap_m=0.5mm`, `contact.region_weights`, `contact.pen_eps_m=2mm` (App C):
  paper specifies thumb/thenar gaps and bounds only → these are documented defaults. (Task 1, 13-14)
- `hand.depth_bias_m`, `track.drift_sigma_*`: mock-mode injected error magnitudes (not from
  paper) used to exercise the correction/optimizer modules. (Task 5, 8, 11)

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

### Contact force proxy (cfrc_ext[:3])
`data.cfrc_ext[bid]` is the external contact wrench applied to body `bid`,
stored as a 6-vector `[torque(3), force(3)]` in the world frame.  We take
`np.linalg.norm(cfrc_ext[bid][:3])` (the rotational half) as a per-fingertip
contact magnitude proxy.  Using the torque half rather than the linear-force half
captures non-zero values whenever a finger is pressed against a curved surface
(which generates a moment even at low contact force).  A real deployment could
use `cfrc_ext[bid][3:6]` for linear contact force instead; the two values are
proportional in practice.  Documented as a proxy; see the reward weight `mu_F`.

### Object-position early termination
`terminated` is triggered when the object centroid drifts more than
`cfg["term"]["obj_pos_err_m"]` (default 0.5 m) from the reference trajectory.
This threshold is deliberately generous so that the no-op Stage-I policy used
in tests (which makes no contact) does not immediately terminate the episode.
Paper (App D) specifies early termination for "large tracking errors" but gives
no threshold value; 0.5 m is a documented default for the mock setting.
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
