# HOT3D HOI-recovery improvement loop — design

Date: 2026-07-08. Approved approach: **A — greedy expected-value ladder**
(sequential tiers, verified commits, autonomous with checkpoints).

## Problem

The best strategy (`configs/real_forehoi_icp_joint.yaml`, "icpj") was
evaluated on 6 HOT3D single-object interaction clips against mocap GT
(`compare/hot3d/batch_summary.json`, BEST_STRATEGY.md taxonomy). Findings:
registration is ~18–20 mm chamfer wherever stage-1 masks are clean; rotation
error tracks shape symmetry (masher 37° abs vs cube 154°); two clips fail
catastrophically from stage-1 segmentation (mug+forearm merge 60.7 mm,
spatula→table leak 158.8 mm).

Goal of this campaign: systematically improve "recover HOI from video" on
this benchmark using any promising optimization-based or learned method.

## Test regime (fixed)

- Pipeline input: rectified RGB + rendered GT depth (`--depth gt`), object
  UNKNOWN at test time (model-free). GT poses/models are used for
  evaluation only.
- Objective (lexicographic): **T1** worst-clip chamfer into the ~20 mm band
  (fix mug, spatula); **T2** reduce mean rot_traj and absolute attitude
  error. Placement is already near the depth-noise floor.
- Mode: autonomous experiment→eval→commit loop; checkpoint reports at tier
  boundaries, plateaus, or decisions needing the user.

## Frozen benchmark

- The 6 clips and their adapter outputs `/workspace/datasets/hot3d/
  rc_input_*` are FROZEN for the campaign (vase 002500, potato_masher
  002349, bottle_bbq 002034, puzzle_toy 001964, mug_white 001970,
  spatula_red 001990). No adapter changes mid-campaign.
- Every experiment is a named **arm** (config + code state). Runs:
  `render_and_compare/runs/hot3d_<cat>_<clipnum>_<arm>`. Eval:
  `compare/hot3d/gt_pose_hot3d_<run>.json` via `gt_pose_eval_hot3d.py`.
- `compare/hot3d/LEADERBOARD.md`: one row per arm × clip (chamfer_mm med,
  centroid_cm med, rot_traj med/p90, rot_abs med) + per-arm aggregate
  (worst-clip chamfer, mean rot_traj). Baseline = icpj rows from
  batch_summary.json.

## Acceptance gate

An arm is committed iff no individual clip regresses >20% on chamfer or
rot_traj vs the best committed arm, AND (lexicographic):
1. worst-clip chamfer strictly improves, OR
2. worst-clip chamfer ties (within 1 mm) and mean rot_traj (6 clips)
   strictly improves.
Tie-breakers: rot_abs, centroid. Any run whose stage 3 fell back to
depth-lift (log line "falling back to depth-lift") is INVALID — rerun or
mark failed; never score a fallback as the arm's result.

**Render-and-eyeball rule**: before a full 6-clip eval, every candidate
change is screened on ONE clip and its overlay video visually inspected
(this session's three convention bugs were all caught by eyes, not
metrics).

## Tiers

Each tier opens with a short literature/web pass (paper_search + web
search) to adopt the best known technique instead of reinventing it.
Within a tier, iterate variants; after 2 consecutive gate failures, write
findings to BEST_STRATEGY.md and advance to the next tier.

### T1 — hand-aware segmentation (fixes mug + spatula)
- Negative SAM2 prompts from detected hand boxes; subtract projected
  HaMeR/WiLoR hand pixels from object masks.
- Mask-QA gate: hand-overlap fraction, temporal IoU stability, area-jump
  detection; on failure, re-prompt SAM2 at a cleaner frame (target visible,
  low hand overlap).
- Touches stage 1 (`real_perception.py::segment_object` and neighbors)
  only.

### T2 — photometric azimuth (attacks rotation everywhere)
- Differentiable LAB-chroma NCC of the splatted vertex-colored SAM-3D mesh
  added as a term in `object_icp.py::_joint_refine` (machinery adapted
  from `compare/hoi4d/gate2/sam3d_icp/photometric_check.py`).
- Anchor-frame attitude search: N rotation hypotheses at the anchor frame
  (azimuth grid × optional icosahedral), scored by photometric +
  depth+silhouette; winner initializes the trajectory.

### T3 — grasp-rigidity prior
- Detect stable-grasp segments (object–wrist relative velocity below
  threshold over a window; cross-check against HOT3D-HIT statistics).
- During those segments constrain object pose DELTAS to HaMeR wrist
  deltas (soft penalty in `_joint_refine`). The hand carries the azimuth
  signal that symmetric geometry hides.

### T4 — learned tracker bake-off
- Literature pass for 2025–26 model-free RGB-D object tracking/
  reconstruction (BundleSDF and successors, HOLD-style neural fits).
- Integrate the 1–2 most promising as alternative stage-4 backends (new
  conda envs; Blackwell sm_120 constraint — expect build friction).
- Adopt or ensemble per-phase if they beat the committed arm.

## Failure handling

- OOM: SAM-3D subprocess retries once after freeing GPU (second GPU
  available; a stray 23 GB process caused the one OOM so far).
- Per-clip crashes don't abort the batch (existing driver behavior).
- Every committed arm records its exact config diff + numbers in the
  commit message and LEADERBOARD.md.

## Reporting

Checkpoint report to the user at: each tier boundary, each committed win,
plateau (2 failed experiments), or any blocking decision (licenses, big
downloads, spending >~2 h on one integration).
