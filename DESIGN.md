# DESIGN — Compositional HOI-from-Video pipeline

Goal: from a monocular RGB video recover **4D HOI** = per-frame hand motion (MANO),
object 3D shape + 6D pose trajectory, and contact (when/where). This compositional
pipeline is the **teacher / error-characterization rig** whose cleaned outputs become
pseudo-ground-truth + distillation targets for a future single feed-forward model.

Spine follows **CHOIR** (arXiv:2605.20992): coarse contact-agnostic init → generative
spatial rectification → contact-aware joint optimization. Expanded into swappable
modules so every stage's error is measurable.

Target setting (per project decision): **model-free / unknown object** is the primary
branch (SAM-3D-Objects / BundleSDF); capture setting kept general (third-person via
HaMeR+Dyn-HaMR, egocentric via HaWoR) behind one hand interface.

---

## Stage 0 — Preprocess & camera
- **In:** raw RGB video.
- **Out:** frames; intrinsics `K[3,3]`; extrinsics `[T,4,4]` (world→cam); metric depth.
- **Models:** VIPE (extrinsics/intrinsics); MoGe-v2 / Depth-Anything-V2 / Metric3D-v2 (metric depth); DROID-SLAM fallback.
- **Errors:** monocular scale ambiguity; camera drift on low parallax; rolling shutter / blur.
- **Log:** depth reprojection error & confidence; VIPE reprojection residual; parallax.

## Stage 1 — Detect, sides & segmentation (2D cues)
- **In:** frames.
- **Out:** hand boxes + L/R; interacting-object box; object masks (modal + amodal); object point tracks.
- **Models:** WiLoR det-head (+ interacting-object box head) or 100DOH; SAM 2; amodal video seg (Chen 2025); CoTracker3.
- **Errors:** mask leakage/loss under occlusion; L/R swaps & ID switches; track drift on specular/rotating objects.
- **Log:** mask IoU stability; hand-object mask overlap; track confidence.

## Stage 2 — Hand reconstruction (per-frame → world)
- **In:** frames + hand boxes/sides + camera.
- **Out:** per-frame MANO (θ,β,orient,transl); world-space stabilized trajectory; joints3d.
- **Models:** HaMeR (per-frame) + Dyn-HaMR (temporal/world); HaWoR (egocentric).
- **Errors (dominant):** root-depth/translation ambiguity; jitter; β drift across frames.
- **Log:** 2D keypoint reprojection vs wrist depth; acceleration (jitter); β variance.

## Stage 3 — Object shape + 6D pose (model-free primary)
- **In:** frames + amodal masks + metric depth + camera + point tracks.
- **Out:** object mesh (canonical) + scale; per-frame 6D pose `[T,4,4]`.
- **Models:** SAM-3D-Objects (anchor mesh + guarded follow-track) **and/or** BundleSDF (RGB-D neural SDF over whole clip). CAD branch (FoundationPose/MegaPose) kept as a calibration control.
- **Errors (highest in model-free):** anchor-frame shape/scale ambiguity; 6D drift in occluded contact phase; symmetry flips.
- **Log:** multi-view silhouette IoU of mesh; per-frame mask-reprojection IoU; rotational jumps.

## Stage 4 — Spatial alignment
- **In:** hand motion + object trajectory + depth + camera.
- **Out:** hand & object in ONE metric world frame; resolved global scale gauge.
- **Method:** express both via camera extrinsics; solve one global similarity (Umeyama) to metric depth.
- **Errors:** residual hand↔object scale mismatch — quantify the contact-frame surface gap. This is the misalignment CHOIR's later stages exist to fix.

## Stage 5 — Contact-agnostic 4D fit (coarse)  ← initial watchable result
- **In:** aligned scene + masks + 2D keypoints.
- **Out:** temporally smooth hand motion + object 6D trajectory; constant β; **no contact reasoning**.
- **Method:** joint per-clip smoothing + (silhouette/keypoint reprojection in real mode).

## Stage 6 — Generative rectification + contact correspondences
- **In:** coarse 4D HOI + object geometry.
- **Out:** rectified relative placement + per-frame **barycentric contact correspondences**.
- **Model:** flow-matching grasp prior trained on GraspPair (≈500k DexGraspNet grasps); predicts ray-depth corrections. Mock/fallback: heuristic snap-to-contact.
- **Correspondences:** KNN (k≈50) on object surface, valid if distance < 2 cm and surface-normal angle < 60°.

## Stage 7 — Contact-aware joint optimization (final)
- **In:** rectified frames + correspondences + stage-1/5 evidence.
- **Out:** refined hand motion, object shape, 6D trajectory, per-frame contact maps.
- **Losses:** `L_contact` (pull active hand verts to barycentric anchors) + `L_pen` (one-sided non-penetration) + `L_silhouette` + `L_anchor` (prior) + `L_temporal`, with a periodically rebuilt soft contact cache.

## Stage 8 — Evaluation & error attribution (research payload)
- Per-stage residual + confidence; ablate one module at a time.
- **Metrics:** hand MPJPE/PA-MPJPE + accel; object ADD(-S), mask IoU, traj smoothness; contact F1/IoU, penetration depth/volume; contact-frame surface gap.
- **Export:** stage-7 outputs as pseudo-GT; intermediate signals (masks, depth, stage-6 grasp corrections, contact maps) as distillation targets for the single feed-forward network.

---

### Error-budget intuition (what the feed-forward model must internalize hardest)
1. Monocular **scale/depth at the wrist** (stage 2/4).
2. **Occluded-contact relative placement** (stage 6) — image evidence underdetermines it; needs a grasp prior.
3. Model-free **object shape/scale** (stage 3).
