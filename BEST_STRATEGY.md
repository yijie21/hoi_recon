# Current best strategy — HOI object reconstruction

*Status 2026-07-10. This is the concise, current strategy doc: the whole arc in
brief, the **full end-to-end workflow** of today's best strategy, the live numbers,
and what's left to improve. The long-form journey (every tier, bug, and dead end)
lives in [`compare/hot3d/docs/REFLECTION.md`](compare/hot3d/docs/REFLECTION.md);
the head-to-head numbers in
[`compare/hot3d/scores/LEADERBOARD.md`](compare/hot3d/scores/LEADERBOARD.md).*

---

## TL;DR

**Goal:** recover an object's 6DoF trajectory + shape from an egocentric hand-object
video, measured against mocap-grade GT.

**Benchmark:** frozen 6-clip HOT3D bench, mocap-grade GT, *calibrated RGB-D* (HOT3D
has no depth sensor, so depth is ray-cast from the posed CADs + hand meshes — this
calibration turned out to be the decisive property of the whole project).

**Two things carry the title "best" today, and they answer different questions:**

| | **icpjgr** (best *integrated* pipeline) | **combined** (best *accuracy*) |
|---|---|---|
| what | one runnable command; hand-built depth+silhouette registration + grasp | Any6D per-frame RGB-D pose + icpjgr's temporal layer as a post-processor |
| worst-clip chamfer | **21.2 mm** (never catastrophic) | 21.1 mm, but sustained rotation flips on masher/cube |
| accuracy | 3.2× over baseline | lower chamfer on **4/6** clips (bottle 5.6, mug 3.4, vase 6.5, spatula 10.1) |
| robustness | ✅ bounded worst case, no drift, no flips | ⚠️ only fixes *isolated* symmetry flips |
| status | ✅ accepted best arm, fully integrated | 🔬 prototype (post-processor), points the way |

**The finding that reframes the project:** on calibrated RGB-D, a learned per-frame
estimator that *consumes* the depth (Any6D, FoundationPose) is **more accurate** than
the hand-built registration; the pipeline's real contribution is **temporal
robustness** (it never drifts, never symmetry-flips). The winning recipe is the
*combination* — a learned pose core inside the temporal optimizer.

---

## The full workflow (current best strategy, end to end)

Input: egocentric/monocular RGB video (+ camera intrinsics; on HOT3D also the
ray-cast GT depth). Output: per-frame object mesh + 6DoF pose trajectory
(`stage8_eval/pseudo_gt.npz` = `{obj_verts, obj_faces, obj_poses[T,4,4]}`) + hand
meshes. The pipeline runs as 9 cached stages (`render_and_compare/hoi_recon/stages/`),
resumable per stage.

**Stage 0 — Preprocess / camera** (`stage0_preprocess`).
Rectify to a pinhole camera and extract frames. On HOT3D the adapter
(`compare/hot3d/make_rc_input.py`) rectifies the Aria **fisheye624** stream onto a
*virtual upright pinhole* (90° FOV, 1024²) via `cv2.remap`, and **ray-casts GT depth**
(open3d `RaycastingScene`) from the posed eval GLBs + UmeTrack hand meshes into that
camera as 16-bit mm PNGs → the calibrated RGB-D substrate (`--depth gt`,
`RC_GT_DEPTH_DIR`). *Depth substrate is the single biggest lever: GT sensor depth
when it exists, VGGT-Omega otherwise — never per-frame monocular (it breathes).*

**Stage 1 — Hand-aware object segmentation** (`stage1_detect_track`, **T1 — the
biggest win**). Track **both hands and the object** as separate SAM2 objects. Prompt
the object with K≤5 candidate clicks (original + offsets), **veto** any that land on
the frame-0 hand mask, **score** each resulting track by
`temporal-IoU − hand-overlap − area-jump − border-fraction`, and pick the best.
Two guardrails: *minimal-intervention* (if the original click's track is already
healthy, leave the plain mask untouched — never hurt a clean clip) and *vanilla
fallback* (object enclosed by both hands). Code: `hoi_recon/mask_qa.py` +
`real_perception.py::_run_sam2_multi_hypothesis`, gated by
`backend.hand_aware_seg`. *This upstream mask decides everything downstream — both
of the original catastrophic failures were stage-1 mask errors (mug absorbed the
forearm → 25 cm blob; spatula leaked onto the table → 80 cm mesh), because the frozen
prompt pixel can land on the occluding hand.* → **mug 60.7→7.0, spatula 158.8→20.5 mm.**

**Stage 2 — Hand mesh** (`stage2_hand`). HaMeR/HaWoR → per-frame MANO hands + wrist
global rotation (`mano_global`). Feeds grasp contact and the T3 rigidity term.

**Stage 3 — Canonical object mesh** (`stage3_object`). Pick the largest-mask anchor
frame; run **SAM-3D** on the masked crop (subprocess in the `sam3d5090` env) → one
canonical textured mesh, reused for every frame. The generated *shape* is already at
the depth noise floor — do **not** invest in guided-diffusion shape refinement; the
generated *proportions* are what need fixing (handled by per-axis scale in stage 4).
*Every comparison is **mesh-controlled**: reuse the incumbent's stage 0–3 so SAM-3D's
GPU nondeterminism can't masquerade as method signal.*

**Stage 4 — Per-frame object pose (the core; two modes).**
- **(a) Registration core — `icpjgr`, the integrated best.** `object_icp.py`:
  per-frame trimmed (80%) rigid **ICP** of 20k mesh samples onto the masked, eroded,
  backprojected depth cloud (sequential init; per-frame scale *locked* — a free
  per-frame similarity is degenerate), then **ONE global metric + per-axis log-scale**
  solved from the fused all-frames cloud, then a differentiable **joint depth +
  silhouette** refinement (`_joint_refine`): trimmed depth correspondences (mm) +
  distance-transform out-of-silhouette penalty (allowed region = object mask ∪ hand
  boxes) + coverage + rotation prior + second-difference *trajectory* smoothness.
  Plus two targeted priors: **T2 chroma attitude search** at the anchor (score N
  azimuth hypotheses on LAB-chroma, **spread-gated at 18 LAB** so it only acts when
  texture discriminates — fixes the bottle 19.9→9.9, self-disables elsewhere) and
  **T3 speed-gated grasp-rigidity** (`‖dR_obj − dR_wrist‖²` over contact-detected
  stable-grasp pairs, gated to ≥4°/frame wrist rotation — recovers azimuth during
  fast in-hand rotation where depth is blind; broad p90 tail gain).
  Config: `configs/real_forehoi_icp_joint_grasp.yaml`.
- **(b) Learned core — `combined`, the accuracy frontier.** Replace the registration
  with a **learned per-frame RGB-D estimator** (Any6D, render-and-compare, CVPR'25)
  fed the *same* mesh + depth + mask — more accurate per frame, but per-frame
  independent → snaps a minority of frames to a ~180° symmetry-equivalent pose. Then
  apply the temporal layer as post-processing (`compare/hot3d/combined_refine.py`):
  **symmetry-flip resolution** (discover the mesh's near-symmetry group; per frame
  pick the symmetry-equivalent rotation temporally closest to its neighbours — fixes
  the flip *without moving the surface*) + **data-anchored translation jitter
  smoothing** (`argmin ‖y−x‖² + λ‖D²y‖²`, closed form, λ_trans=3 — a universal free
  win: position jitter p90 66→7 mm at <1 mm chamfer cost). Rotation smoothing stays
  **off** — the scalar smoother averages across residual flips and *harms*; it needs a
  flip-aware SO(3) formulation.

**Stages 5–7 — Coarse fit → rectify → grasp closure** (`stage5_coarse_fit`,
`stage6_rectify`, `stage7_contact_optim`). The grasp optimizer (`joint_grasp.py`)
**trusts the object track and moves the HAND** to close the grasp (`w_prior_obj: 200`)
rather than dragging the registered object off its track.

**Stage 8 — Eval** (`stage8_eval`). Writes `pseudo_gt.npz`; scored by
`compare/hot3d/gt_pose_eval_hot3d.py` vs mocap GT (symmetric posed-surface **chamfer**
mm, **centroid** cm, **rot_traj** deg after trajectory-optimal constant alignment,
alignment-invariant **canonical-shape ICP**).

---

## Current numbers (mesh-controlled, vs mocap GT — chamfer mm / rot_traj p90 °)

| clip | icpj (baseline) | **icpjgr** (best integrated) | combined (Any6D + temporal) |
|---|---|---|---|
| bottle_bbq | 19.9 / 72.8 | 9.9 / 72.9 | **5.6 / 16.7** |
| mug_white | 60.7 / 49.9 | 7.0 / 76.5 | **3.4** / 73.0 |
| vase | 17.5 / 80.9 | 17.7 / 75.7 | **6.5** / 93.1 |
| spatula_red | 158.8 / 70.2 | 21.2 / 42.6 | **10.1** / 153.4 |
| potato_masher | 18.9 / 42.5 | **18.8 / 42.3** | 12.3 / 172.8 ⚠ sustained flip |
| puzzle_toy (cube) | 18.6 / 127.8 | 18.5 / 127.2 | 21.1 / 157.4 ⚠ 24-fold symmetric |

icpj → icpjgr: **worst-clip chamfer 158.8→21.2 mm, mean 49.1→15.5 mm (3.2×)**;
rotation median stays ~flat, floored by the cube (24-fold symmetric + SAM-3D generates
wrong stickers → no geometry *or* texture signal exists). combined wins chamfer on
4/6 but its rotation is only fixed where flips are *isolated* — masher (sustained
wrong basin) and cube (true symmetry) still defeat it. Full table + the T4 learned
bake-off (HORT / ForeHOI / FoundationPose / Any6D) in
[`scores/LEADERBOARD.md`](compare/hot3d/scores/LEADERBOARD.md) and
[`docs/T4_RESULTS.md`](compare/hot3d/docs/T4_RESULTS.md).

---

## What's still to improve (ranked by expected value)

1. ✅ **DONE — Integrate the learned core *inside* the pipeline.** Stage-4
   `pose_core: learned` runs Any6D as a subprocess + the temporal layer in one pipeline
   run, object frozen through grasp (arm `any6dp`, `configs/real_any6d.yaml`). Wins
   chamfer 4/6 (often 2–3×) → the **placement-optimal** arm; fails the gate on rotation
   (icpjgr stays `BEST_ARM`). Detail: [`docs/T5_NOTES.md`](compare/hot3d/docs/T5_NOTES.md).
2. ⛔ **NEGATIVE — cannot recover the learned core's rotation.** Four approaches tried
   (depth-anchored basin selection = redundant with the depth-consuming core;
   grasp-rigidity prior = hurts; surgical flip-fix = neutral; icpjgr-rot/Any6D-transl
   hybrid = chamfer collapses). Finding: `any6dp` and `icpjgr` are a genuine
   **placement-vs-rotation Pareto pair** — Any6D's chamfer is inseparable from its
   rotation. [`docs/T5_NOTES.md`](compare/hot3d/docs/T5_NOTES.md).
3. 🔍 **DIAGNOSED (gate-invisible) — geometry anchor-attitude search.** A 400-hypothesis
   SO(3) go/no-go: attitude errors are correctable in principle (mug `rot_abs` 139→12
   achievable) and multi-frame depth+silhouette *partially* recovers them where shape
   discriminates (mug handle 139→46), but depth is fooled on thin/symmetric shapes and
   the whole fix is **gate-invisible** (`rot_traj` removes the constant offset it fixes).
   Not built. [`docs/T5_NOTES.md`](compare/hot3d/docs/T5_NOTES.md).
4. ⛔ **NEGATIVE — texture re-projection.** Baking the real clip texture onto the mesh is
   smeared (needs the accurate rotations it's meant to help produce — chicken-egg),
   circular (self-consistent with the baking pose), and where it adds discrimination
   (cube) it's defeated by symmetry; where T2 already works (bottle) it *hurts*.
   [`docs/T5_NOTES.md`](compare/hot3d/docs/T5_NOTES.md).
5. ✅ **DONE (earlier, T1) — hand-aware segmentation** is the highest-leverage upstream
   fix; every method's ceiling is set by the stage-1 mask on hand-held objects.
6. ✅ **DONE — scaled the bench to 12 clips** (HOT3D-HIT). The placement win generalizes
   (any6dp chamfer wins **9/11**, new-clip median 7.1→2.8 mm) and the Pareto trade-off
   holds (rotation regresses 6/11). [`docs/T5_NOTES.md`](compare/hot3d/docs/T5_NOTES.md).

**Campaign verdict:** item 1 (learned placement core) is the durable win — validated at
2× scale. The rotation/attitude/texture axis (items 2–4) is a proven hard wall on this
benchmark: symmetric-object orientation is fundamentally under-constrained by depth, and
the corrective signals (wrist, generated texture, temporal priors) are noisier than the
learned per-frame estimate. `icpjgr` remains the rotation-robust `BEST_ARM`; `any6dp` is
the placement-optimal alternative.

---

## Runnable recipe

```bash
# envs: rc5090 (pipeline + eval), sam3d5090 (SAM-3D subprocess),
#       forehoi5090 (Any6D / ForeHOI / FoundationPose), hort5090 (HORT)
PY=/workspace/miniconda3/envs/rc5090/bin/python
cd compare/hot3d

# best integrated pipeline on the 6-clip bench (adapter -> pipeline -> overlay -> score)
$PY run_batch.py selection.json --arm icpjgr --config \
    configs/real_forehoi_icp_joint_grasp.yaml

# accuracy frontier: learned per-frame pose, then the temporal layer (per clip)
$PY run_any6d_hot3d.py <ABS rc_input> <ABS icpjgr_run> <ABS any6d_run>   # env forehoi5090
$PY combined_refine.py <any6d_run> <combined_run>                        # flip-fix + jitter smooth
$PY gt_pose_eval_hot3d.py <rc_input> <combined_run>                      # score vs GT
$PY leaderboard.py render                                                # -> scores/LEADERBOARD.md
```

Single arm end to end without the batch driver: `python -m hoi_recon.cli --video
rgb.mp4 --out <run> --real --config configs/real_forehoi_icp_joint_grasp.yaml
--depth gt --object-prompt <x> <y>` (with `RC_GT_DEPTH_DIR` / `RC_GT_INTRINSICS` set).

---

## Reference & load-bearing caveats

- **Benchmark inputs:** `/workspace/datasets/hot3d/rc_input_<num>_<clip>/`
  (rgb.mp4, frames/, ray-cast depth_png/, intrinsics.npy). 6 frozen clips: bottle_bbq
  002034, mug_white 001970, vase 002500, potato_masher 002349, spatula_red 001990,
  puzzle_toy 001964.
- **Acceptance gate** (`leaderboard.py`): lexicographic — no clip regresses >20% on
  chamfer/rot_traj (noise floors +2 mm / +5°), then worst-clip chamfer strictly
  improves, or ties within 1 mm and mean rot_traj improves. *The gate under-credits
  real wins* (T2 = chamfer, T3 = rotation tail, Any6D = placement) — always keep the
  raw per-clip numbers, not just the verdict.
- **Convention traps** (all four caught by *rendering and eyeballing*, never by a
  metric): HOI4D poses annotate the CAD **bbox centre**, not the origin; the only
  `MANO_LEFT.pkl` on this box is **fabricated** (mirrored right hand — get the official
  MPI file before trusting any left-hand decode); HOT3D ships two model sets whose
  canonicals disagree — **pose the `object_models_eval` GLBs (meters), not the display
  GLBs**; poses/cameras are quaternion-wxyz world transforms.
- **HOT3D-HIT** (the bench-expansion source): 113 per-object interaction timelines
  over 20 sequences at
  `/workspace/datasets/hot3d/hot3d-hit/ROHIT-Paper-data/hot3d_hit.json`; 302 Aria BOP
  clips. Selection driver `compare/hot3d/probe_clips.py`.
- **Envs:** dead pre-Blackwell envs (`forehoi`, `hort`, `hold`, `easyhoi`, `daid`,
  cu118/cu121) have no sm_120 kernels — do not use. Blackwell revival recipes:
  [`compare/hot3d/docs/T4_NOTES.md`](compare/hot3d/docs/T4_NOTES.md) ("one env for
  all" is infeasible; each learned method was rebuilt by cloning the `sam3d5090`
  stack).
- **The durable lessons:** calibration is a moat; accuracy and robustness are
  different axes (learned pose wins one, temporal optimization wins the other);
  mesh-control every comparison; render and eyeball — it caught every convention trap
  and load-bearing bug in the project.
