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
| Rotation weakly observable from top-down depth; trimming discards the disambiguating spout/handle points → ICP walks 33–97° from the image-based init | `sam3d_icp/RESULTS.md` (orientation section), `rc_ab_rotation_*.mp4` | `object_icp.rotation: init` keeps the stage-3 silhouette-tracker rotation, ICP solves translation only (`configs/real_forehoi_icp_rotinit.yaml`). Trade-off: direction right, depth fit 3.9→10.1 mm |
| SAM-3D mesh lateral proportions too wide (1.68× mask footprint) → no rigid pose satisfies depth AND silhouette; global scale refit amplifies it | `sam3d_icp/silhouette_check.py`, IoU table in RESULTS.md | none yet — this is the next build (below) |
| "GT" sensor depth is not pixel-perfect: boundary halo, ±20 px frame-to-frame depth↔RGB wobble, ~8% background contamination inside the eroded mask | measured in-session (RESULTS.md) | `fg_band_m` filter in `object_icp.py`; erosion; treat scale estimates from pre-filter runs (1.13–1.18) as upper bounds |
| Depth metrics have a silhouette blind spot (only score covered mask pixels; overhang invisible) | `silhouette_check.py` | always render the reprojection overlays (`hoi_recon/viz/reproject.py`, auto-generated `*_reproj.mp4`) and run `silhouette_check.py` alongside `eval_pipeline_ab.py` |

### 6. Scoreboard (kettle_N15, GT depth, stages 4–8 from identical caches)

| arm | object placement | 3D fit med/p90 | vis MAE | wiggle | silh. IoU |
|---|---|---|---|---|---|
| base | depth-lift, flat archived mesh | 9.9/22.4 mm | 2.95 cm | 1.59 cm | 0.51 |
| icp | + rigid ICP | 4.8/6.5 mm | 1.03 cm | 0.65 cm | **0.65** |
| icp2 | + regenerated SAM-3D mesh | 4.2/6.9 mm | **0.69 cm** | 0.37 cm | 0.58 |
| icp4 | + global scale refit (rot free) | **3.9/5.7 mm** | 0.73 cm | **0.36 cm** | 0.46 |
| icp5 | rot locked to silhouette tracker | 10.1/10.8 mm | 1.18 cm | 0.75 cm | 0.59 |

No single arm wins everything — icp4 is the depth/stability champion, icp5
has the right orientation, icp has the best silhouette. That split is the
precise definition of the next build. (icp6/icp7 — the fg-band reruns —
were stopped mid-flight; ignore those run dirs.)
Eval code: `compare/hoi4d/gate2/sam3d_icp/{eval_pipeline_ab.py,
silhouette_check.py}`; videos: `rc_ab*.mp4` in the same folder.

## Next build (the one experiment that resolves the split)

**Joint depth + silhouette registration with per-axis scale**: extend
`object_icp.py`'s per-frame objective with a rendered-silhouette / mask-IoU
term (and optionally a small rotation prior toward the stage-3 track), and
promote the global scale to a per-axis (anisotropic) similarity solved from
fused cloud + silhouette jointly. Expected: icp4's seating + icp5's
orientation + ~1.1× footprint in one run. This is also exactly slice 1 of
the b5 photometric refiner (`idea-loop-reports/2026-07-03-4dgs-jitter-free-
hoi/branches/b5-gs-universal-refiner.md`) — the surviving research
direction; its remaining budget after the substrate work: object 9.6→1.4 cm,
hand 8.2→3.1 cm, jitter 1.7→0.65 cm (RC_MATRIX_RESULTS.md).

## Blocked / caveats

- The raw HOI4D tree (`/workspace/hoi4d`: 12 clips, GT depth, masks) is
  **gone from disk**; all current evidence is single-clip (kettle_N15) from
  the archived `render_and_compare/runs/kettle_gt*`. Restore the dataset
  before any multi-clip claim.
- Hand side is untouched by all of this (hand-depth closure was only 0.16 in
  the matrix): HaWoR/HaMeR hand + its anchors is the other half of b5.
- Dead pre-Blackwell envs on this box: forehoi, daid, hort, easyhoi, hold.
  Working: rc5090, sam3d5090, gate2.
