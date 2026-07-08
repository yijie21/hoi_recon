# Current best strategy for stable, aligned HOI reconstruction

*(status 2026-07-07, after the kill tests, Gate-2, and the object-registration
experiments; all numbers are kettle_N15 / HOI4D unless stated. Every claim
below has a RESULTS.md with the harness that produced it.)*

## The strategy in one paragraph

Take the best available **metric depth substrate** (GT sensor depth when it
exists; VGGT-Omega otherwise — never per-frame monocular), segment the object
with **validated video masks**, generate ONE canonical textured object mesh
with **SAM-3D** (single anchor frame), and then place that mesh per frame by
**registration against the depth cloud** — per-frame rigid ICP with the
per-frame scale locked (unobservable), plus ONE global metric scale solved
from the fused all-frames cloud (observable). Let the stage-7 grasp optimizer
trust the registered object track and close the grasp by moving the hand.
What this does NOT yet solve: object orientation and lateral extent are
weakly observable from depth alone — they need the image-space (silhouette /
photometric) term, which is the next build.

## The runnable recipe

```bash
# env: rc5090 (torch 2.11+cu128); SAM-3D subprocess env: sam3d5090
cd render_and_compare
export RC_OBJECT_MASK_PATTERN="<clip>/masks/frame_{idx:06d}_masks/object.png"
export RC_OBJECT_MASK_ERODE=5
export RC_GT_DEPTH_DIR=<clip>/depth RC_GT_INTRINSICS=<clip>/intrin.npy   # or vggt_depth_mm
/workspace/miniconda3/envs/rc5090/bin/python -m hoi_recon.cli \
    --video <clip.mp4> --out runs/<name> --real \
    --config configs/real_forehoi_icp.yaml --depth gt \
    --object-prompt <x> <y>
```

## Components, with the functional code

### 1. Depth substrate — the single biggest lever

- **GT injection**: `--depth gt` backend reads `RC_GT_DEPTH_DIR` (16-bit mm
  PNGs, sorted-index matched) + `RC_GT_INTRINSICS` —
  `render_and_compare/hoi_recon/backends/real_perception.py` (`_gt_geometry`).
- **VGGT-Omega injection** (best GT-free substrate): densify once per clip
  with `compare/hoi4d/gate2/densify_vggt_depth.py` (single forward over all
  frames, uint16 mm PNGs), then feed through the same `gt` backend — see
  `compare/hoi4d/gate2/run_rc_matrix.sh` for the exact wiring.
- Evidence: 12-clip matrix, `compare/hoi4d/gate2/RC_MATRIX_RESULTS.md` —
  MoGe→VGGT-Omega closes **53% of object-depth / 61% of object-jitter** of
  the oracle (GT) gap with zero method changes. Per-frame mono depth (MoGe)
  is what causes the depth breathing; the kill tests
  (`compare/hoi4d/{kill_test,pivot_test}/RESULTS.md`) proved background
  substrates and geometric anchors cannot repair it after the fact.

### 2. Object segmentation — inject validated masks

- `RC_OBJECT_MASK_PATTERN` + `RC_OBJECT_MASK_ERODE` seam in
  `real_perception.py` (`segment_object` → `_external_object_masks`): use
  SAM2/SAM3-tracked masks you have verified, identically across conditions.
  An unlucky auto point-prompt once segmented a 1.2k-px fragment (IoU 0.04)
  and silently poisoned everything downstream.

### 3. Canonical object mesh — SAM-3D, one anchor frame

- Stage 3 (`hoi_recon/stages/stage3_object.py` → `run_object_sam3d` in
  `real_perception.py`) picks the largest-mask frame and runs
  `third_party/sam-3d-objects/sam3d_infer.py` (`--no-texture`) as a
  subprocess in the **sam3d5090** env (`backend.sam3d_env`). Build recipe for
  Blackwell/RTX-5090: `render_and_compare/scripts/subprocess_entries/
  sam-3d-objects/BLACKWELL_ENV.md` (weights: gated `facebook/sam-3d-objects`,
  download with `HF_HUB_DISABLE_XET=1`).
- Route-B finding (`compare/hoi4d/gate2/sam3d_icp/route_b_deform.py`): the
  generated *shape* is already at the depth noise floor (per-vertex
  refinement moved ~0.1 mm) — do NOT invest in guided-diffusion shape
  generation. The generated *proportions/scale* are what's wrong (see §5).

### 4. Object placement — registration, not depth-lift (the core module)

- **`render_and_compare/hoi_recon/object_icp.py`** — the heart of the
  strategy. Hooked into stage 4 behind `cfg.object_icp`
  (`hoi_recon/stages/stage4_align.py`), configured in
  `configs/real_forehoi_icp.yaml`:
  - `refine_object_poses()`: per-frame trimmed (80%) rigid ICP of 20k mesh
    surface samples onto the masked, eroded, backprojected depth cloud;
    sequential init (frame 0 from the stage-3 track).
  - **Per-frame scale locked by design**: a free per-frame similarity fit is
    degenerate — a 53% oversized kettle fits the visible front at unchanged
    residual (`sam3d_icp_test.py`, `RESULTS.md`).
  - `global_scale_refit: true` → `_solve_shared_scale()`: ONE metric scale
    about the canonical origin from the fused all-frames cloud, poses frozen
    during the solve, mild 0.95 trim, then re-register. (Per-frame centering
    or tight trims delete the scale evidence — documented in the docstring.)
  - `fg_band_m` (default 0.15): rejects mask pixels whose depth is >15 cm
    off the per-frame median — real sensor depth bleeds background values
    ~8% deep into even the eroded mask and inflates the scale solve.
  - `rotation: free | init` — see §5, this is an open trade-off.
- Why registration at all: depth-lift placement (the old default) cannot use
  mesh quality — a *better* SAM-3D mesh made depth-lift *worse* (4.54 vs
  2.95 cm MAE) while registration turned it into the best result.
- Stage-7 companion (`configs/real_forehoi_icp.yaml → optim.joint`):
  `w_prior_obj: 200, w_seat: 1` — the grasp optimizer
  (`hoi_recon/joint_grasp.py`) must trust the registered object and move the
  HAND to close the grasp; with default weights it dragged the object a
  median 1.6 cm off the ICP track.

### 5. Known limits (measured, not hypothetical)

| limit | evidence | mitigation in code |
|---|---|---|
| Rotation weakly observable from top-down depth; trimming discards the disambiguating spout/handle points → ICP walks 33–97° from the image-based init | `sam3d_icp/RESULTS.md` (orientation section), `rc_ab_rotation_*.mp4` | joint refinement (`_joint_refine`): rotation starts at the image-based init, silhouette DT term supplies the image-space gradient, w_rot_prior=10 keeps it out of the symmetric basin |
| SAM-3D mesh lateral proportions too wide (1.68× mask footprint) → no rigid pose satisfies depth AND silhouette; global isotropic scale refit amplifies it | `sam3d_icp/silhouette_check.py`, IoU table in RESULTS.md | per-axis scale in `_joint_refine` — found [1.088, 0.944, 1.109]: lateral axis shrunk, footprint 1.94×→1.36×, IoU 0.46→0.69 |
| "GT" sensor depth is not pixel-perfect: boundary halo, ±20 px frame-to-frame depth↔RGB wobble, ~8% background contamination inside the eroded mask | measured in-session (RESULTS.md) | `fg_band_m` filter in `object_icp.py`; erosion; treat scale estimates from pre-filter runs (1.13–1.18) as upper bounds |
| Depth metrics have a silhouette blind spot (only score covered mask pixels; overhang invisible) | `silhouette_check.py` | always render the reprojection overlays (`hoi_recon/viz/reproject.py`, auto-generated `*_reproj.mp4`) and run `silhouette_check.py` alongside `eval_pipeline_ab.py` |

### 6. Scoreboard (kettle_N15, GT depth, stages 4–8 from identical caches)

| arm | object placement | 3D fit med/p90 | vis MAE | bias | wiggle | silh. IoU | footprint |
|---|---|---|---|---|---|---|---|
| base | depth-lift, flat archived mesh | 9.9/22.4 mm | 2.95 cm | — | 1.59 cm | 0.51 | — |
| icp | + rigid ICP | 4.8/6.5 mm | 1.03 cm | −1.03 | 0.65 cm | 0.65 | 1.11× |
| icp2 | + regenerated SAM-3D mesh | 4.2/6.9 mm | 0.69 cm | −0.69 | 0.37 cm | 0.58 | 1.68× |
| icp4 | + global scale refit (rot free) | **3.9/5.7 mm** | 0.73 cm | −0.73 | **0.36 cm** | 0.46 | 1.94× |
| icp5 | rot locked to silhouette tracker | 10.1/10.8 mm | 1.18 cm | −1.17 | 0.75 cm | 0.59 | 1.67× |
| **icpj3** | **joint depth+silhouette, per-axis scale** | 11.3/12.1 mm | **0.67 cm** | **−0.42** | 0.66 cm | **0.69** | **1.36×** |

**GT-pose verdict (2026-07-08, HOI4D annotations + CAD now on disk at
`/workspace/datasets/hoi4d`, evaluator `sam3d_icp/gt_pose_eval.py`)**: all
arms carry a large CONSTANT attitude error (abs rotation 55–81°; after
removing one constant offset the per-frame tracking error is only 13–19° —
icpj3 best at 13.3°). The anchor-frame attitude initialization is the
dominant unfixed error; depth-metric differences below ~1 cm are beneath the
GT annotation's own noise floor (6–16 mm) and must not be used to rank arms.
Chamfer vs GT (after the CAD bbox-centering convention fix — HOI4D poses
the box center): icp2 21.3 / icp4 23.5 / icp5 23.9 / **icpj3 20.8 mm**;
centroid 5.3–7.0 cm, icpj3 best on every GT metric. Full table + caveats in
`sam3d_icp/RESULTS.md`.

**icpj3 (config `real_forehoi_icp_joint.yaml`) is the current default arm**:
best visible-surface accuracy, best (smallest) depth bias, best silhouette
IoU and footprint, orientation kept from the image-based track. Read the two
lagging columns correctly: icp4's 3.9 mm `fit` and 0.36 cm wiggle are earned
with a 33–97°-wrong rotation exploiting the dome symmetry — `fit_mm` rewards
that cheat, and its rigid-basin convergence is artificially stable. icpj3's
remaining wiggle is data-driven (the ±20 px depth↔RGB wobble): w_temp 8
over-smooths and gets worse; w_temp 2.5 is the optimum found.
Per-axis scale found [1.088, 0.944, 1.109] — the silhouette SHRANK the wide
lateral axis while depth grew the observed axes; the isotropic refit (1.146)
could not express this and inflated the footprint instead.
(icp6/icp7 — the fg-band reruns — were stopped mid-flight and removed;
icpj/icpj2/icpj4 were tuning variants, also removed.)
Eval code: `compare/hoi4d/gate2/sam3d_icp/{eval_pipeline_ab.py,
silhouette_check.py}`; videos: `rc_ab*.mp4`, joint arm:
`rc_ab_joint_{object,hoi}_reproj.mp4` in the same folder.

## The joint registration (built 2026-07-08 — was "next build")

`object_icp.py::_joint_refine`, enabled by `object_icp.joint_silhouette` in
`configs/real_forehoi_icp_joint.yaml`. Torch (GPU), ~15 s for 75 frames on
top of the ICP init. Optimizes per-frame rotation deltas + translations and
ONE global per-axis log-scale against: (a) trimmed depth correspondences in
mm, refreshed each round; (b) a distance-transform out-of-silhouette penalty,
bilinearly sampled so it is differentiable, with allowed region = object
mask ∪ valid hand boxes (the object legitimately projects behind the
occluding hand); (c) a coverage term (visible mask pixels must be near a
projected mesh point); (d) a small prior toward the image-based init
rotations plus second-difference smoothness on the ABSOLUTE trajectory
(the init rotations themselves jitter, so smoothing deltas alone fails —
measured: icpj vs icpj3). Pixels and mm are naturally comparable at this
camera (~1.1 mm/px), so unit-ish weights balance (w_sil 1, w_cov 0.3,
w_rot_prior 10, w_temp 2.5).

## Next build

**Anchor-frame attitude search** — the GT eval says per-frame tracking is
already decent (13° med) but the whole trajectory is rotated wrongly as a
block by the stage-3 anchor init that `rotation: init` and the joint prior
faithfully preserve. Fix at the source: N rotation hypotheses at the anchor
frame (icosahedral × azimuth grid, or 4-azimuth minimum), each scored by the
held-out photometric term (`sam3d_icp/photometric_check.py` machinery) plus
depth+silhouette, winner initializes the pipeline. After that, in
expected-value order: (2) hand side — HaWoR/HaMeR anchors (hand-depth
closure was only 0.16 in the matrix); (3) photometric refinement on top of
the joint track (b5 slice 2); (4) multi-clip validation — the RGB release
and annotations for ALL clips are now in `/workspace/datasets/hoi4d`
(depth-folder download still pending a downloader fix; kettle_N15 GT depth
lives in the archived runs).

## Dataset options beyond HOI4D (surveyed 2026-07-08)

Requirement: video + sensor depth + CAD + object pose trajectory + hand pose.
HOI4D's annotation quality is the current bottleneck for evaluation
(per-frame human box fits: ±1.5 cm scatter, 6.4 mm GT-vs-depth floor,
depth↔RGB wobble — measured in `sam3d_icp/RESULTS.md`).

| dataset | depth | CAD | obj pose traj | hand | pose quality vs HOI4D |
|---|---|---|---|---|---|
| **HOT3D** (Meta 2024) | ✗ no depth camera (Aria/Quest3) — but mocap-grade poses + scanned CADs let you RENDER pixel-perfect GT depth | ✓ 33 scanned + PBR | ✓ mocap-grade | ✓ MANO/UmeTrack | ≫ |
| **DexYCB** (NVIDIA 2021) | ✓ 8× RealSense | ✓ YCB | ✓ | ✓ MANO | > (8-cam joint fits) |
| **HO-Cap** (2024) | ✓ 8× RealSense | ✓ scanned | ✓ | ✓ MANO | > newest tooling |
| H2O (2021) | ✓ | ✓ | ✓ | ✓ two hands | > |
| ARCTIC (2023) | ✗ | ✓ articulated | ✓ mocap-grade | ✓ | ≫ |
| Aria Digital Twin (2023) | ~ rendered from twin | ✓ | ✓ | limited (no MANO) | ≫ |

Notes: HOT3D is egocentric with headset motion (camera:identity does not
hold — use its mocap camera trajectories); Aria RGB is fisheye624 (use
hand_tracking_toolkit's camera model, not pinhole K). DexYCB/HO-Cap are the
drop-in matches for the current harness (real sensor depth, static cams).
**Verified in-session (2026-07-08)**: HOT3D BOP-clips release downloaded from
HF `bop-benchmark/hot3d` to `/workspace/datasets/hot3d` (2 Aria clips + all
33 GLBs); GT overlays are pixel-tight even for in-hand moving objects, and the GT
hands render finger-perfect via the toolkit-native UmeTrack model —
`compare/hot3d/gt_overlay_hot3d.py`, videos `gt_overlay_hot3d_clip-*.mp4`.
WARNING for any MANO use on this box: the only MANO_LEFT.pkl here (HaWoR's,
copied to /workspace/datasets/hot3d/mano) is FABRICATED — a right-hand model
with mirrored template but the right hand's PCA basis/means — decoding
official left-hand thetas with it reverses the palm. Get the official MPI
MANO_LEFT.pkl before trusting any left-hand MANO decode.
Pose the object_models_eval GLBs (meters) — objects.json poses are in the
BOP-eval canonical; the display object_models (mm) are NOT all canonically
consistent (uid 31 mouse is axis-swapped + offset ~2 cm). Poses/cameras are
quaternion-wxyz world transforms.

**HOT3D-HIT** (arXiv 2512.07394 "Reconstructing Objects along Hand
Interaction Timelines") downloaded to
`/workspace/datasets/hot3d/hot3d-hit/ROHIT-Paper-data/hot3d_hit.json`:
113 per-object interaction timelines over 20 HOT3D sequences (22 categories,
all names map to model uids via object_models_models_info.json); segments
labeled scene_static / scene_dynamic / inhand (1,239 in-hand grasps, 58,789
frames), frame indices into the FULL sequence. 19/20 sequences are Aria and
contain 302 BOP clips — the bridge to single-interaction clip selection
(BOP clips are timestamp-indexed; full-sequence frame↔timestamp mapping via
the ROHIT repo's parsing, github.com/zhifanzhu/objects-along-hit).

### Pipeline run on HOT3D (2026-07-08): the strategy generalizes, and so does its failure mode

The full icpj recipe ran end-to-end on HOT3D clip-002500 (vase, 150 frames)
via a new adapter, `compare/hot3d/make_rc_input.py`: Aria fisheye rectified
onto a virtual upright 90°-FOV pinhole camera (1024², upright by rotating the
camera about its optical axis — no post-rotation), and GT depth ray-cast
(open3d) from ALL posed eval GLBs + UmeTrack hand meshes into that camera as
16-bit mm PNGs (`--depth gt` / RC_GT_DEPTH_DIR convention; HOT3D has no depth
sensor). SAM2 prompt = projected GT centroid at frame 0. Run:
`runs/hot3d_vase_icpj`; side-by-side vs GT:
`compare/hot3d/rc_vs_gt_hot3d_vase.mp4`; metrics
`compare/hot3d/gt_pose_eval_hot3d.py` → `gt_pose_hot3d.json`.

Joint refine converged to depth 3.9 mm / sil 0.60 px (tighter than kettle's
5.2 mm / 2.4 px — cleaner rendered depth, mocap GT). Against mocap GT:
chamfer 17.5 mm med, centroid 2.48 cm med, rot_traj 19.4° med — all better
than kettle_N15 icpj3 (20.8 mm / 5.29 cm), and this time with NO annotation
noise floor to hide behind. Phase breakdown is the real finding:

| phase | rot_traj | chamfer | note |
|---|---|---|---|
| f0–95 vase static on table | 14–21° | 17–22 mm | placement visually on-object throughout |
| f100–120 picked up, mouth-on views | 54–74° | **6–8 mm** | symmetry cheat: chamfer at its BEST while rotation is at its worst |
| f140–150 fast in-hand rotation | 122° | 21 mm | attitude visibly wrong in overlay; centroid drifts to 6 cm |

Same two known limits reproduced on a second dataset: (a) constant anchor
attitude error (abs rot 77° via shape-G) — the anchor-frame attitude search
stays the next build; (b) in-hand rotation tracking fails on near-revolution
shapes exactly when depth ICP has no azimuth signal. Per-axis scale hit its
usefulness limit too: it shrank SAM-3D's over-wide x-axis 22% ([0.78, 0.97,
1.06]). HOT3D is adopted as the primary eval bed going forward — mocap GT
makes sub-cm differences decidable.

### 6-clip HOT3D batch (2026-07-08): failure taxonomy

Selection: HIT names the interaction object per sequence; each clip's own
mocap GT confirms in-clip motion + FOV (probe of 18 train-Aria clips →
`compare/hot3d/probe_clips.py`, batch driver `run_batch.py`, per-clip
side-by-sides `rc_vs_gt_<cat>_<clip>.mp4`, aggregate `batch_summary.json`).
Medians vs mocap GT (`gt_pose_hot3d_*.json`):

| clip (target) | chamfer | centroid | rot_traj | rot_abs | verdict |
|---|---|---|---|---|---|
| vase 002500 | 17.5 mm | 2.5 cm | 19.4° | 77° | good track; azimuth cheat in-hand |
| potato_masher 002349 | 18.9 mm | 5.7 cm | 24.1° | **37°** | best attitude — asymmetric shape pins rotation |
| bottle_bbq 002034 | 19.9 mm | 3.7 cm | 39.3° | 91° | revolution shape → azimuth unobservable; z-scale hit 1.4 clamp |
| puzzle_toy 001964 | 18.6 mm | 3.4 cm | 82.7° | 154° | cube = 24-fold geometric symmetry: depth+sil has ZERO azimuth signal (only 37/150 frames registered) |
| mug_white 001970 | 60.7 mm | 15.4 cm | 18.0° | 169° | SAM2 merged mug+FOREARM during propagation → SAM-3D generated a 25 cm mug-with-arm blob (caught visually) |
| spatula_red 001990 | 158.8 mm | 13.9 cm | 32.8° | 46° | SAM2 anchor mask leaked into table/plate → 80 cm mesh; thin-object segmentation failure |

Taxonomy: (1) registration itself is consistently ~18-20 mm chamfer whenever
stage 1 gives a clean mask — across 4 categories; (2) rotation quality is a
direct function of shape asymmetry (masher 37° ≫ bottle/cube), confirming
azimuth must come from PHOTOMETRIC evidence, not geometry; (3) the two
catastrophic failures are both stage-1 SEGMENTATION on hand-held/thin
objects (mug+arm merge, spatula→table leak), upstream of everything this
strategy tunes. Priority update: hand-aware segmentation (subtract hand
masks / negative clicks from SAM2) is now tied with the anchor attitude
search; both are cheaper than any further registration work.

## Blocked / caveats

- The raw HOI4D tree (`/workspace/hoi4d`: 12 clips, GT depth, masks) is
  **gone from disk**; all current evidence is single-clip (kettle_N15) from
  the archived `render_and_compare/runs/kettle_gt*`. Restore the dataset
  before any multi-clip claim.
- Hand side is untouched by all of this (hand-depth closure was only 0.16 in
  the matrix): HaWoR/HaMeR hand + its anchors is the other half of b5.
- Dead pre-Blackwell envs on this box: forehoi, daid, hort, easyhoi, hold.
  Working: rc5090, sam3d5090, gate2.
