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
