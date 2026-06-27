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
