# HOI methods workbench — collect & compare

Every hand-object-interaction (HOI) reconstruction method in this repo lives in its own
sibling folder under `hoi_recon/` and can run standalone. This `compare/` folder adds a
**method-agnostic viewer** plus a small **adapter** per method, so any of them can be viewed on
one shared timeline — in the same viewer style as `render_and_compare`.

> Method codes used elsewhere in this repo (`icpjgr`, `fpauto`, `any6dp`, …) are decoded in
> [`../GLOSSARY.md`](../GLOSSARY.md). This page mostly covers a different set of third-party
> methods (HORT, ForeHOI, do-as-i-do, EasyHOI, HOLD) that are compared here, not in the main
> HOT3D benchmark.

## Methods (subfolders)

| Method | Subfolder | Status | Env | Real perception? |
|---|---|---|---|---|
| render_and_compare (CHOIR) | `render_and_compare/` | mock runs work; real perception needs its two heavy backend envs | `forehoi` (numpy mock) | optimization + backends wired |
| egoaero (EgoAERO) | `egoaero/` | mock only (real perception stubbed) | base / `forehoi` | no (GT-driven mock) |
| **ForeHOI** | `forehoi/` | fully runnable (feed-forward on video) | `forehoi` | yes |
| **do-as-i-do** | `do-as-i-do/` | fully runnable (SAM3→SAM-3D→MoGe→HaWoR→TAPIR→guided-diffusion) | `daid` | yes |
| **HORT** | `hort/` | fully runnable (feed-forward, single image) | `hort` | yes |
| HOLD | `hold/` | env ready; real reconstruction blocked (data is account-walled) | `hold` | yes, once data unblocked |
| EasyHOI | `easyhoi/` | hand + object shape run; fusion/alignment blocked (LISA + afford-diffusion envs) | `easyhoi` (+`instantmesh`) | yes (hand + shape only) |

Each subfolder has a `RUN.md` with the exact, verified run command and output format.

## The shared scene format

Every adapter converts a method's native output into a self-contained `compare/scenes/<method>.npz`:

```
hand_verts   [T,Nh,3]   required, metres, OpenCV camera frame
hand_faces   [F,3]      optional -> MANO mesh; absent -> point cloud
obj_verts/obj_faces/obj_poses   canonical mesh + per-frame 4x4 pose   (mesh objects)
  OR obj_points [T,M,3] (+ obj_point_colors)                          (point-cloud objects)
contact_mask [T,Nh]     optional -> in-contact verts turn red
source       str        label
```

## View it (interactive, like render_and_compare)

```bash
PY=/workspace/miniconda3/envs/forehoi/bin/python    # has viser
# all methods side by side on one timeline:
$PY compare/viser_workbench.py compare/scenes/*.npz --port 8080
# pick a subset:
$PY compare/viser_workbench.py compare/scenes/hort.npz compare/scenes/forehoi.npz
```
Reach it from your laptop: `ssh -p $VAST_TCP_PORT_22 -L 8080:127.0.0.1:8080 root@$PUBLIC_IPADDR` → http://localhost:8080

Static preview (no server): `$PY compare/snapshot.py compare/out.png compare/scenes/*.npz`
→ see `compare/compare_all.png`.

ForeHOI also keeps its own native viewer: `forehoi/viser_4d_demo.py output_4d/wild1`.

## Rebuild the scenes (adapters)

```bash
PY=/workspace/miniconda3/envs/forehoi/bin/python
$PY compare/adapters/hort_to_scene.py     hort/out_demo/f0030                    compare/scenes/hort.npz
$PY compare/adapters/forehoi_to_scene.py  forehoi/output_4d/wild1                compare/scenes/forehoi.npz
$PY compare/adapters/rc_to_scene.py       render_and_compare/runs/demo/stage8_eval/pseudo_gt.npz compare/scenes/render_and_compare.npz
$PY compare/adapters/egoaero_to_scene.py  /tmp/egoaero_viz/run                   compare/scenes/egoaero.npz
$PY compare/adapters/easyhoi_to_scene.py  easyhoi/data_run                       compare/scenes/easyhoi.npz
```

## Same-clip comparison (wild6.mp4) — 5 methods, real perception

Five methods reconstruct the **same** clip (`egoaero/assets/wild6.mp4`, a left hand grasping a
~15 cm bottle) with real perception, so they're directly comparable:

| Scene | Method on wild6 | Object output | Object world size |
|---|---|---|---|
| `render_and_compare_wild6.npz` | render_and_compare (MoGe+WiLoR+SAM2+HaMeR+SAM-3D, 74 frames) | textured mesh (scale-fixed, see below) | 0.188 m |
| `do_as_i_do_wild6.npz` | do-as-i-do (SAM3+SAM-3D+MoGe+HaWoR+guided-diffusion, 293 frames) | coherent bottle mesh, 6-DoF tracked | 0.121 m |
| `hort_wild6.npz` | HORT, per-frame on 33 frames | sparse point cloud, no temporal consistency (hand-centred, see below) | 0.154 m |
| `easyhoi.npz` | EasyHOI, single snapshot (T=1, see below) | shape mesh, pose not aligned | 0.120 m |
| `forehoi_wild6.npz` | ForeHOI, 12 frames | dense coherent bottle mesh (186K verts) | 0.251 m |

```bash
PY=/workspace/miniconda3/envs/forehoi/bin/python
# 5 methods side by side on one shared timeline (interactive viser):
$PY compare/viser_workbench.py \
    compare/scenes/render_and_compare_wild6.npz \
    compare/scenes/do_as_i_do_wild6.npz \
    compare/scenes/forehoi_wild6.npz \
    compare/scenes/hort_wild6.npz \
    compare/scenes/easyhoi.npz --port 8080
# static 5-panel snapshot (no server):
$PY compare/snapshot.py compare/compare_wild6_5way.png \
    compare/scenes/render_and_compare_wild6.npz \
    compare/scenes/do_as_i_do_wild6.npz \
    compare/scenes/forehoi_wild6.npz \
    compare/scenes/hort_wild6.npz \
    compare/scenes/easyhoi.npz
```
→ `compare/compare_wild6_5way.png`. Reach the server from your laptop the same way as above.
Each method is normalized to the same box size and laid out in a row, so shapes line up
regardless of each method's own metric scale (see the depth-jitter note below for what that
normalization hides).

**Takeaways** (real bottle ≈ 0.15 m), after the fixes below:
- **render_and_compare** — bottle-sized object (0.188 m) held firmly in the grasp; hand good,
  contact tight. Competitive/best.
- **ForeHOI** — cleanest, most complete object mesh (0.251 m); strong all-round.
- **HORT / EasyHOI** — object size close to true, but HORT's is a noisy per-frame cloud and
  EasyHOI's isn't pose-aligned.
- **render_and_compare's differentiable variant** — better hand keypoint reprojection (4.9 px)
  but its object pose drifts off the hand on some frames; not strictly better than the robust
  path here.

How wild6 was produced for HORT and ForeHOI (shared-GPU box, ForeHOI needs a ~19 GB window):
```bash
# HORT: per-frame on wild6 frames -> temporal scene
conda activate hort && CUDA_VISIBLE_DEVICES=0 python hort/demo.py --img_folder <wild6 frames> --out_folder hort/out_wild6
$PY compare/adapters/hort_seq_to_scene.py hort/out_wild6 compare/scenes/hort_wild6.npz
# ForeHOI: on the video (obj/hand pixel prompts on frame 0)
cd forehoi && CUDA_VISIBLE_DEVICES=<free gpu> python export_4d_hoi.py --video .../wild6.mp4 \
    --obj-points 502,483 502,387 --hand-points 90,607 --frames 12 --name wild6
$PY compare/adapters/forehoi_to_scene.py forehoi/output_4d/wild6 compare/scenes/forehoi_wild6.npz
```
render_and_compare's own real run: `render_and_compare/runs/wild6_real/` (robust path) and
`runs/wild6_combined/` (full differentiable path); setup in `render_and_compare/REPRODUCE.md`.
Rebuild its scene: `$PY compare/adapters/rc_to_scene.py render_and_compare/runs/wild6_real/stage7_contact_optim/arrays.npz compare/scenes/render_and_compare_wild6.npz`.
Scenes: `render_and_compare_wild6.npz` (robust, fixed), `render_and_compare_wild6_combined.npz`
(differentiable), `render_and_compare_wild6_before.npz` (pre-fix, see below).

## Bugs found and fixed along the way

These were each caught by looking at the reconstruction, not by a metric — kept here because the
root causes are easy to hit again.

- **HORT: depth jitter inflates the hand's apparent size.** HORT is per-frame feed-forward with no
  temporal consistency; on wild6 its absolute depth swings ~1.15 m frame-to-frame even though x/y
  barely move, so the whole scene appears to teleport and HORT's hand shrinks to ~15% of its
  bounding box in the shared viewer (other methods sit at ~80%). Fix: `hort_seq_to_scene.py`
  recenters every frame on the hand centroid (same shift applied to hand and object, so the grasp
  is preserved). Result: centroid range 1.154 m → 0, hand box fraction 0.15 → 0.82. Raw frames
  still available via `--no-recenter`.

- **HORT: mirrored hand on a left-handed clip.** HORT (like WiLoR/HaMeR) only models right hands.
  Its demo detects a left hand, flips the image horizontally, reconstructs a right hand, and
  exports it in that mirrored frame without flipping back (no flip flag is stored either). So
  wild6 (a left hand) came out as a mirror-image right hand with the object on the wrong side.
  Fix: `hort_seq_to_scene.py --mirror` negates x on both hand and object (grasp preserved) and
  reverses MANO face winding (a reflection inverts triangle orientation). The wild6 scene is now
  built with `--mirror`.

- **EasyHOI is a single snapshot, not a 4D trajectory.** It's a single-image method — `easyhoi.npz`
  has T=1. It reconstructs one frame's hand+object shape and can't be scrubbed over time like the
  others (render_and_compare=74 f, ForeHOI=12 f, HORT=33 f); it shows as a static pose.

- **render_and_compare: object came out 5× too big.** Root cause was stage 1, not the scaling math:
  `_object_prompt` returned the hand-box centre as SAM2's object prompt, so for a hand holding an
  object it segmented the hand instead (masks covered 55–73% of the frame), and SAM-3D reconstructed
  a hand-blob (0.813 m). Fix: a real `--object-prompt X Y` override wired CLI→config→stage 1. With
  `--object-prompt 502 482` (the bottle, from YOLO on frame 0), the object mask dropped to 8.5% of
  the frame and size came out at 0.188 m. Code changed: `hoi_recon/{cli.py,
  stages/stage1_detect_track.py, backends/real_perception.py}`; re-run recipe in
  `render_and_compare/REPRODUCE.md`. `render_and_compare_wild6_before.npz` keeps the broken result
  for comparison.

- **render_and_compare: hand too small.** The MANO hand's longest span came out 0.144 m vs a real
  adult hand's ~0.19 m (~0.77×) — set at the WiLoR hand stage, the same on every downstream stage,
  so not a late bug. Monocular hand scale is inherently ambiguous, and wild6 is a hard clip (hand
  heavily occluded by the grasped bottle, portrait phone video). The object is unaffected (it's
  anchored on MoGe depth directly). Fix: `compare/fit_hand_scale.py` applies a per-clip global
  scale + translation. Neither obvious 2D target is reliable on wild6 (WiLoR's `kp2d` is scattered,
  15% out-of-frame; the YOLO hand box is ~2.25× the hand, including forearm), so instead it scales
  to the canonical hand-size prior (0.19 m → scale 1.32) and translates to match the projected hand
  centroid to the box centre. Result: span 0.144→0.190 m, centroid-vs-box error 164→35 px. Writes
  `arrays_handfit.npz` (both `backproject.py` and `rc_to_scene.py` prefer it if present).
  Before/after: `compare/backproj/rc_handfit_before_after.png`.

- **do-as-i-do: object appears to change size in the overlay (it's depth, not scale).** In
  `compare/backproj/daid/overlay.mp4` the bottle sometimes projects larger than real and wobbles
  frame-to-frame, but its actual 3D size is constant (`local_to_scene.scale` = 0.6054 on every one
  of 293 frames, std 0.0000, and the mesh is one canonical mesh × a single `mesh_scale` of 0.0977
  — the tracker runs with `--fix_scale_to_init_frame`, so the object is rigid). What actually moves
  is depth: `translation_scale_optimized` swings 0.52→1.01 (~2×) and camera-frame z goes
  0.154→0.268 m (std 0.024), while in-plane x/y barely move (std 0.006 m, ~4× tighter). Since
  apparent size under a pinhole camera is ∝ 1/z, that depth swing *is* the apparent-size swing —
  the object never resizes in the 3D (viser) view, only in the 2D reprojection.
  - **Root cause:** the object's depth is anchored to the HaWoR hand's depth
    (`obj_target = h_real + k·(o_pm − h_pm)`, correlation 0.775), and HaWoR's own per-frame depth
    wobbles about 8× more than the hand truly moves (z path-length 0.53 m vs net displacement
    0.067 m → wiggle ratio 7.9).
  - **A better depth model does not fix it.** Swapping in DA3-metric depth (`DA3METRIC-LARGE`,
    `depth_models/da3.py`, verified metric: ref-frame z 0.52 m vs MoGe's affine 1.81 m) in place of
    MoGe left the apparent-size swing unchanged (1.75×→1.76×), because the anchor is the hand, not
    the depth map (DA3 also isn't scale-consistent with the reused MoGe track without a full
    ~8 h `track_object` re-run). The DA3 wrapper is available but `pipeline.yaml` still defaults to
    MoGe.
  - **Temporal smoothing does fix it.** A median(9)+moving-avg(7) filter on the object's translation
    (applied identically to hand and object, so the grasp is preserved) cuts flicker 0.018→0.005
    (3.6× lower) and the size swing 1.75×→1.59×. `compare/adapters/do_as_i_do_to_scene.py` applies
    this by default (`--raw` disables it). Before/after: `compare/backproj/daid_depth_smoothing.png`;
    smoothed overlay: `compare/backproj/daid_smoothed/`.

## do-as-i-do — setup notes (reproducible)

do-as-i-do's reconstruction (`do-as-i-do/reconstruction/`) is a 7-stage pipeline: SAM3 (text-
prompted segmentation) → SAM-3D mesh → MoGe pointmaps → HaWoR hands → GeoCalib (gravity/horizon)
→ TAPIR (point tracking, for velocity) → guided-diffusion object-pose tracking → hand-anchored
translation/scale optimization. It needs ≥32 GB VRAM and the gated `facebook/sam3` weights. Built
as its own `daid` conda env by copying `forehoi` (reuses its compiled pytorch3d/kaolin/nvdiffrast/
moge/spconv for sm_89) and layering the rest on top. Version pins that mattered, all matching the
authors' `env/sam3d.yml`:

- **utils3d** — the SAM-3D fork needs the old `utils3d==0.0.2` API (`utils3d.torch.*`, flat
  `depth_edge`); PyPI's `utils3d` is an unrelated package, so install git commit `d790d33`
  (pre `maps`-refactor).
- **moge** — must be `moge==1.0.0` (git commit `a8c3734`), which uses `utils3d.torch` (moge 2.x
  uses `utils3d.pt` and conflicts). Reuse the `moge-vitl` v1 weights.
- **jax/chex** — TAPIR needs `jax==0.4.30 jaxlib==0.4.30 chex==0.1.86 ml-dtypes==0.4.1` (numpy-1.26
  safe).
- **weights** — SAM-3D reused from `/workspace/code/ForeHOI/checkpoints` (symlinked, saves 12 GB);
  HaWoR (`hawor.ckpt`, `infiller.pt`, `detector.pt`) and TAPIR (`bootstapir_v2`) downloaded
  (ungated); MANO_RIGHT reused; `_DATA/data/mano_mean_params.npz` copied from a HaMeR data dir.
- **MANO_LEFT** (needed since wild6 is left-handed, and not present on this box) — generated by
  x-mirroring MANO_RIGHT (negate x on `v_template`/`J`, negate x on `posedirs` output + conjugate
  the pose feature, flip faces; `shapedirs` left "buggy" to match the official file the way
  HaWoR's `fix_shapedirs` expects it). Verified exact: 0.0000 mm vs. mirroring the right hand
  through HaWoR's own `run_mano_left` on random poses/shapes.
- **SAM3's "left hand" text prompt returns an empty mask** on this clip. Worked around by (a)
  patching `generate_mesh_sam3d.py` to skip empty masks (the hand is meshed by HaWoR instead), and
  (b) generating the stage-4 scale-optimization hand mask from the HaWoR hand-mesh projection
  (verified to land on the real hand).
- **Fast-SAM3D tracker** — drop `--enable_ss_cache`/`--torch_compile`; both force distilled
  `ss_generator_faster` weights we don't have. Result is identical, just slower without them.

Run recipe: `scratchpad/run_daid_wild6.sh` (full run, VRAM-guarded) or `resume_daid_wild6.sh`
(resumes from TAPIR). Output: `wild6/obj_tracking_out/white_bottle/combined_visualization/
layout_camera_frame_optimized.json` (293-frame object 6-DoF, OpenCV camera frame) +
`white_bottle.obj` + `wild6/wild6/all_hand_meshes.npz`. Adapter:
`compare/adapters/do_as_i_do_to_scene.py do-as-i-do/reconstruction/wild6 compare/scenes/do_as_i_do_wild6.npz`.

**Result (wild6):** bottle object 0.121 m (real ≈0.15 m), hand span 0.174 m, hand–object depth gap
2 cm (co-located). In the backprojection the hand is on-target and the object mesh tracks the
bottle 6-DoF through the grasp and tilt — competitive with render_and_compare and ForeHOI, and
(unlike HORT) a coherent tracked mesh rather than a per-frame cloud.
`compare/backproj/daid/`, 5-way view: `compare/compare_wild6_5way.png`.

## Backprojection / reprojection check (overlay on the real frames)

`compare/backproject.py` overlays each method's 3D reconstruction back onto the frames it
consumed, using that method's own camera (no cross-method re-warp), so what you see is its true
pixel reprojection error. Hand mesh = green, object = blue.

```bash
PY=/workspace/miniconda3/envs/forehoi/bin/python
$PY compare/backproject.py rc      compare/backproj/rc       # full-frame 720x1280 originals, 74f
$PY compare/backproject.py forehoi compare/backproj/forehoi  # its 518x518 processed frames, 12f
$PY compare/backproject.py hort    compare/backproj/hort     # HORT's native 224 crop (un-flipped), 33f
$PY compare/backproject.py daid    compare/backproj/daid     # do-as-i-do full-frame 720x1280, 293f
```
Each writes `overlay.mp4` plus a 6-frame `contact_sheet.png`.

| Method | Target | Hand alignment | Object alignment |
|---|---|---|---|
| render_and_compare | full frame 720x1280 (original) | on the hand; slightly small/offset a few frames | tracks the bottle through the grasp — solid |
| do-as-i-do | full frame 720x1280 (original) | on the hand (HaWoR) | 6-DoF tracks the bottle through grasp + tilt — solid |
| ForeHOI | its 518² processed frames | very tight | on the bottle in grasp frames (letterboxed square view) |
| HORT | its 224 crop only | excellent (WiLoR) | sparse noisy point cloud, floats around the grasp |

Notes: HORT runs on a crop and never saves the crop box, so its reprojection can only be shown on
that crop (un-flipped to the true left hand) — a full-frame overlay would need re-deriving the
detection box. ForeHOI's frames are letterboxed because it squares the portrait video. The mesh
overlay is a painter's-algorithm rasterizer in cv2 (depth-sorted alpha triangles) — no GL, runs
headless.

## Caveats for a fair comparison

The scenes above (outside the wild6 5-way section) are each method's own native demo on a
*different* clip (HORT→wild6 bottle, ForeHOI→wild1, the two mock methods→a synthetic sphere) — not
a controlled comparison. For a true head-to-head, run the real-perception methods on the same
clip and use render_and_compare's real backends instead of mock, the way the wild6 5-way run
above does. That's the natural next step if more methods need comparing; the adapters already
support it — just point them at same-clip outputs.
