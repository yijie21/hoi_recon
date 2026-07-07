# SAM-3D mesh → GT-depth cloud registration (kettle_N15, 2026-07-06)

Question: in the GT-depth regime (user-stipulated: ignore depth-gauge error),
does "semantically localize the object points, then align the extracted
SAM-3D object to them" produce a stable HOI object trajectory?

**Verdict: YES — but only with the scale locked.** Per-frame trimmed rigid
ICP of the canonical SAM-3D mesh onto mask-selected GT-depth points beats the
full RC pipeline on every honest metric. Freeing per-frame scale destroys it,
even on GT depth — direct confirmation that scale is unobservable from
partial-view geometric fit (the pivot test's conclusion, reproduced in the
best possible regime).

## Numbers (75 frames; fair_metrics.json)

| trajectory | cloud→surface fit med/p90 | visible-depth MAE | wiggle | rot speed |
|---|---|---|---|---|
| RC pipeline (archived kettle_gt) | 10.0 / 22.4 mm | 2.97 cm | 1.58 cm | 3.00°/f |
| **ICP rigid (scale locked = 1)** | **4.7 / 5.7 mm** | **1.12 cm** | **0.79 cm** | 1.84°/f |
| ICP free per-frame scale | 4.9 / 5.9 mm | 1.55 cm | 1.26 cm | 1.36°/f |

- Metric-artifact warning: the eval_rc_matrix convention (median z of ALL
  verts − median GT of the VISIBLE front) reads ~+4 cm for a geometrically
  perfect fit of this ~20 cm kettle. On that metric the pipeline scores 0.74
  and rigid ICP 4.42 — i.e. the old metric *rewarded* the pipeline for
  sitting ~3 cm too close (its true visible-front bias: −2.95 cm).
- **Free scale is degenerate**: from a correct scale-1.0 init, the very first
  frame's similarity fit jumps to s=1.56 (a 53% oversized kettle) with the
  point residual *unchanged* (~5 mm); over the clip s swings ±14% in slow
  10–25-frame oscillations — the exact low-frequency gauge signature. The
  visible front of an inflated, pushed-back object is indistinguishable from
  the true one at sensor noise level. Trim-80% correspondence rejection
  makes it worse (drops the constraining periphery).
- Rigid ICP's residual is flat ~5 mm all clip (one 11 mm blip at f67, a
  mask/occlusion glitch); the pipeline's is 8–23 mm and anti-correlates with
  visible area (corr −0.47: occlusion hurts it), rigid ICP barely (+0.22).

## Reading

1. Semantic localization was never the missing piece (masks were injected
   everywhere), but the user's instinct — one canonical rigid object,
   re-registered per frame — is validated in the GT-depth regime and is
   *better than the current pipeline stages 4–7* at object placement.
2. The one thing registration cannot supply is the global scale: it must
   come from the canonical mesh being metric (here: SAM-3D scaled by the
   pipeline) or an external anchor (hand size / contact). One number per
   clip, not per frame.
3. Ordering with prior results: GT depth + locked-scale registration → 0.79
   cm wiggle. The remaining program (b5) is exactly this component chain
   with VGGT-Omega instead of GT depth + a photometric term to survive the
   depth-gauge error that VGGT-Omega leaves behind (R_ko 0.35).

## Protocol

Archived run runs/kettle_gt is the data source (the raw /workspace/hoi4d
clip tree no longer exists on disk): stage0 GT-injected depth (f16 m),
stage1 SAM2 object masks (IoU 0.98 vs kill-test ref), stage3/4 SAM-3D
canonical mesh (metric, det(pose)=1 verified), stage8 pseudo_gt poses as the
pipeline baseline. Per frame: 5px-eroded mask, backproject, ≤3000 pts (seed
0); trimmed (80%) Umeyama ICP against 20k mesh surface samples, ≤60 iters;
frame 0 init = pipeline pose, frame t init = frame t−1 result. One clip,
one category — kettle; not yet swept over the other 11 clips (SAM-3D meshes
absent: its env is pre-Blackwell and the matrix ran depth-lift fallback).

Files: sam3d_icp_test.py (registration + eval_rc_matrix-convention scores),
fair_metrics.py (visibility-aware z-buffer metric + surface-fit residual),
make_figure.py → curves.png, trajectories.npz, results.json,
fair_metrics.json.

## Pipeline integration (2026-07-07): `object_icp` flag in render_and_compare

Wired as a stage-4 hook: `hoi_recon/object_icp.py` (locked-scale trimmed rigid
ICP, same protocol as above) + `configs/real_forehoi_icp.yaml`
(`object_icp.enable: true`). Two companion stage-7 weight changes in that
config, needed because the grasp optimizer was tuned to distrust the old
depth-lift track: `optim.joint.w_prior_obj 25 → 200`, `w_seat 3 → 1` (the
hand closes the grasp; the ICP object track is trusted). Without them stage7
dragged the object a median 1.6 cm (max 6.6) off the ICP placement
(pipeline_ab_v1_default_stage7.json).

End-to-end A/B, stages 4–8 rerun from the identical archived stage0–3 caches
(runs/kettle_gt_base vs runs/kettle_gt_icp; eval_pipeline_ab.py →
pipeline_ab.json):

| stage-8 output | fit med/p90 | vis MAE | bias | wiggle | grasp verts <1cm |
|---|---|---|---|---|---|
| baseline (no flag) | 9.9 / 22.4 mm | 2.95 cm | −2.95 | 1.59 cm | 257/778 |
| **object_icp + trust weights** | **4.8 / 6.5 mm** | **1.03 cm** | −0.99 | **0.65 cm** | **316/778** |

The raw stage-4 ICP quality (4.7 mm) now survives to stage 8, and the grasp
*improves* (hand moves onto the trusted object). Hand centroid accel rises
26 → 40 mm (the hand is doing the closing); hand placement verified visually.
Videos: rc_ab_object_reproj.mp4 / rc_ab_hoi_reproj.mp4 (top = baseline,
bottom = ICP; e.g. f35–65 the baseline kettle visibly floats off toward the
bowls, the ICP kettle stays on the kettle). Note the archived kettle_gt run
is not bit-reproduced by today's code (≤5 mm object) — hence the fresh
same-code baseline.

## Mesh-quality arm (2026-07-07): regenerated SAM-3D mesh (sam3d5090 env)

SAM-3D rebuilt for Blackwell (third_party/sam-3d-objects/BLACKWELL_ENV.md);
stage 3 regenerated the kettle mesh in-pipeline (base2/icp2 share it; the
archived mesh turned out to be a flat tray-like blob, the new one is a real
kettle — lid, spout, handle). Four-way (eval_pipeline_ab.py base icp base2
icp2 → pipeline_ab_base_icp_base2_icp2.json):

| run | mesh | placement | fit med/p90 | vis MAE | bias | wiggle |
|---|---|---|---|---|---|---|
| base | flat archived | depth-lift | 9.9/22.4 mm | 2.95 cm | −2.95 | 1.59 cm |
| icp | flat archived | ICP | 4.8/6.5 mm | 1.03 cm | −0.99 | 0.65 cm |
| base2 | new kettle | depth-lift | 17.6/20.3 mm | 4.54 cm | −4.54 | 1.88 cm |
| icp2 | new kettle | ICP | **4.2/6.9 mm** | **0.69 cm** | **−0.69** | **0.37 cm** |

Two findings: (1) better mesh + registration is the best cell on every
metric — bias −0.99 → −0.69 cm, wiggle 0.65 → 0.37 cm; (2) better mesh
WITHOUT registration is *worse* than the flat mesh (4.54 vs 2.95 cm MAE) —
the depth-lift placement anchors the mesh centroid on the visible-front
depth, so a mesh with true 3D depth extent lands deeper/wronger. Shape
quality only pays through registration. Remaining Route-B/A headroom on this
clip: ~4 mm fit residual + ≤0.7 cm front bias.

## Route B (2026-07-07): fused-cloud shape refinement — verdict: it's a SCALE
## problem, not a shape problem

route_b_deform.py (sam3d5090 env, pytorch3d): fuse all 75 registered GT
clouds into the canonical frame (stage-4 ICP poses of icp2), deform the mesh
under one-sided point→face + displacement-Laplacian + anchor, plus ONE global
log-scale. Result: **per-vertex deformation converged to ~0.1 mm (nothing) —
the SLAT-generated shape is already at the data's noise floor — but the
global scale corrected to 1.128**: stage 3's projected-bbox metric-scale
heuristic had the kettle 13% undersized (occlusion + erosion bias), and no
per-frame/front-view metric could see it (icp2 bias was only −0.69 cm). The
fused multi-frame cloud is what makes global scale observable. Cloud→mesh
p90 45.8 → 17.9 mm.

Fifth pipeline arm (icp3 = refined mesh injected into stage 3, stages 4-8):

| run | fit med/p90 | vis MAE | bias | wiggle | grasp <1cm |
|---|---|---|---|---|---|
| icp2 (raw new mesh) | 4.2/6.9 mm | 0.69 cm | −0.69 | 0.37 cm | 262 |
| icp3 (Route-B refined) | **4.0/4.8 mm** | 0.93 cm | −0.93 | 0.58 cm | **293** |

Per-frame ICP residual dropped 6.3 → 3.7 mm med (8.1 → 5.1 p90) — the
correctly-scaled mesh registers substantially better, strong evidence the
+12.8% is real and not registration blur. The vis-front numbers read
slightly worse, but that metric's z-buffer min-z bias grows with surface
area per sample (bigger mesh, fixed 20k samples) and front-view metrics
systematically favor undersized meshes — the same unobservability that hid
the 13% in the first place. Trust the 3D fit + grasp columns.

**Program consequence.** Route A (guided-diffusion shape generation) is NOT
justified: generated shape is already noise-floor-good. What IS worth
keeping from the user's depth-guidance idea is its scale half: add a
**fused-cloud global-scale refit** (one number per clip; alternate
per-frame rigid ICP ↔ shared-scale Umeyama, 1-2 rounds) as a standard step
in the object_icp module. That converts the b5 "absolute anchor" requirement
for the OBJECT into a solved sub-problem whenever a depth substrate exists.

**Baked in (icp4).** `object_icp.py` now has `global_scale_refit` (on in
real_forehoi_icp.yaml): register pass → freeze poses → solve ONE scale about
the canonical origin on the fused cloud (NO per-frame centering, mild 0.95
trim — per-frame centering + tight trims delete exactly the radial-extent
evidence; the first implementation did that and found s=1.005) → re-register.
On the raw new mesh it recovers **s=1.1455** and lands end-to-end at fit
3.9/5.7 mm, vis MAE 0.73, wiggle 0.36 cm (pipeline_ab_icp2_icp3_icp4.json) —
Route B's gain with zero offline steps.

## Silhouette check (2026-07-07): the depth metrics have a blind spot

The 3-row progression videos (rc_ab3_object/hoi_reproj.mp4: baseline / +ICP /
final) show the final arm's mesh visually hanging off the kettle in 2D while
its 3D fit is 3.9 mm. silhouette_check.py quantifies what the depth metrics
can't see (they only score mask pixels the mesh covers; overhang outside the
mask is invisible to them):

| arm | footprint IoU | area ratio (mesh/mask) | centroid offset |
|---|---|---|---|
| base | 0.51 | — | — |
| icp (flat mesh) | **0.65** | 1.11 | 31 px |
| icp2 (new mesh) | 0.58 | 1.68 | 26 px |
| icp4 (+scale 1.146) | 0.46 | 1.94 | 50 px |

Reading: the regenerated SAM-3D kettle's **lateral proportions are too wide**
(1.68× mask area at scale 1.0 — the depth-only ICP fits the top dome and
never sees width), the global scale refit fits the cloud's radial relief and
**amplifies** the lateral excess (1.94×), and the widened depth-flat basin
lets the pose slide in-plane (50 px). The old flat mesh matched the
silhouette best precisely because stage 3 scales meshes to the mask bbox.
So after all modifications: best-ever depth placement/stability, but the
projected silhouette REGRESSED — depth-only registration cannot arbitrate
image-space fidelity. This is the sharpest motivation yet for the b5
photometric/silhouette term: add mask-IoU (or rendered-silhouette) evidence
to the registration so width and in-plane translation become observable;
per-axis (anisotropic) scale from silhouette + fused cloud jointly is the
concrete next experiment.
