# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Orchestration workflow
You (Fable) are the orchestrator. Plan, decompose, synthesize.
Reasoning-heavy phases → deep-reasoner. Mechanical work → fast-worker.
High-stakes decisions: task deep-reasoner (Opus) with the problem, think it through thoroughly,
and synthesize a concise conclusion you can act on. Keep your own context lean.

## Where things stand (start here)
Object HOI reconstruction on HOT3D. **Three** best arms (a Pareto triple):
`icpjgr` (rotation-robust `BEST_ARM`, `render_and_compare/configs/real_forehoi_icp_joint_grasp.yaml`;
bounds worst-case rotation on in-hand-rotated symmetric objects), `any6dp` (Any6D learned core,
`configs/real_any6d.yaml`), and **`fpauto`** (FoundationPose auto — track+register_each with a
drift-gated selector; `compare/hot3d/run_fp_hot3d.py`, env forehoi5090). **`fpauto` is the newest
best learned core: it beats `any6dp` on BOTH placement (mean chamfer 8.2 vs 11.5mm) and rotation
(mean rot_traj-p90 88.6 vs 111.6°)** by using a uniform metric mesh + FP's native flip-free tracker —
see [`compare/hot3d/docs/T6_NOTES.md`](compare/hot3d/docs/T6_NOTES.md). NOTE this does NOT reopen the
**proven-dead-end** rotation/attitude/texture *prior* axis (temporal / grasp-rigidity / anchor-attitude
/ texture-baking — all tested negative; do not re-attempt); fpauto is a better learned estimator, not a
corrective prior. Read, in order: [`BEST_STRATEGY.md`](BEST_STRATEGY.md) (workflow + roadmap outcomes +
**Open directions**), [`compare/hot3d/docs/T5_NOTES.md`](compare/hot3d/docs/T5_NOTES.md) (full campaign),
[`compare/hot3d/docs/T6_NOTES.md`](compare/hot3d/docs/T6_NOTES.md) (fpauto), then
[`README.md`](README.md) (nav). Numbers: `compare/hot3d/scores/LEADERBOARD.md`. Envs + convention
traps are in the recalled `hoi-recon-*` memories.

## Repository layout (three peers)
- **`render_and_compare/`** — the main pipeline (installable package `hoi_recon`). 9 cached,
  resumable stages under `hoi_recon/stages/stage{0..8}_*.py`, orchestrated by `hoi_recon/pipeline.py`,
  entered via `hoi_recon/cli.py` (`python -m hoi_recon.cli`). Configs in `configs/`, run outputs in
  `runs/` (gitignored). See its [`README.md`](render_and_compare/README.md) + [`DESIGN.md`](render_and_compare/DESIGN.md).
- **`compare/`** — the benchmark study. `compare/hot3d/` is the live harness: `run_batch.py` (driver),
  `make_rc_input.py` (HOT3D→pipeline adapter: fisheye rectify + GT-depth raycast), `run_any6d_hot3d.py`
  + `combined_refine.py` (learned-core arm), `gt_pose_eval_hot3d.py` (scorer), `leaderboard.py`.
- **`egoaero/`** — separate EgoAERO Part A package (asset-free egocentric HOI, its own stages/tests).

## Architecture — the pipeline (render_and_compare)
Input: monocular/egocentric RGB (+ intrinsics; on HOT3D also ray-cast GT depth).
Output contract: `runs/<name>/stage8_eval/pseudo_gt.npz` = `{obj_verts, obj_faces, obj_poses[T,4,4]}`.
Stages are **cached** — each writes `stage<N>_*/` and is skipped if present (`--force` to recompute,
`--stages` to select a subset). Full per-stage detail (levers, the two stage-4 pose cores, the
grasp-moves-the-hand rule) is in [`BEST_STRATEGY.md`](BEST_STRATEGY.md#the-full-workflow) — don't
duplicate it, read it. Key modules: `object_icp.py` (registration core, arm icpjgr),
`object_any6d.py` (learned core, arm any6dp), `mask_qa.py` + `real_perception.py` (hand-aware
stage-1 segmentation, the highest-leverage win), `joint_grasp.py` (contact closure), `temporal_pose.py`.

**Every comparison is mesh-controlled:** reuse the incumbent's stage 0–3 (`run_batch.py` auto-promotes
identical-mask stage2/stage3 dirs) so SAM-3D GPU nondeterminism can't masquerade as method signal.

## Commands
Conda envs referenced by the docs/harness (`rc5090`, `sam3d5090`, `forehoi5090`, `hort5090`) are the
documented contract but may be **absent on a recycled box** — verify with `conda env list` before
running; storage under `/workspace` is not guaranteed persistent (see `/workspace/CLAUDE.md` §3).

```bash
# --- Pipeline (render_and_compare) ---
# Mock mode: full pipeline on a synthetic clip, no weights needed — the fast sanity check
python -m hoi_recon.cli --out runs/demo --mock --num-frames 48
# Single real arm end to end (needs RC_GT_DEPTH_DIR / RC_GT_INTRINSICS set for HOT3D)
python -m hoi_recon.cli --video rgb.mp4 --out runs/<name> --real \
    --config configs/real_forehoi_icp_joint_grasp.yaml --depth gt --object-prompt <x> <y>
# Re-run only some stages of an existing run
python -m hoi_recon.cli --out runs/<name> --real --stages 4- --force

# --- HOT3D benchmark harness (from compare/hot3d/, env rc5090) ---
python run_batch.py selection.json --arm icpjgr --config \
    ../../render_and_compare/configs/real_forehoi_icp_joint_grasp.yaml   # adapter→pipeline→overlay→score
python run_any6d_hot3d.py <ABS rc_input> <ABS icpjgr_run> <ABS any6d_run>  # learned core (env forehoi5090)
python combined_refine.py <any6d_run> <combined_run>                       # flip-fix + jitter smooth
python gt_pose_eval_hot3d.py <rc_input> <run>                              # score vs mocap GT
python leaderboard.py render                                               # → scores/LEADERBOARD.md

# --- Tests (pytest, mostly mock — no GPU/weights) ---
cd render_and_compare && python -m pytest tests/            # or a single file / -k <name>
cd egoaero && python -m pytest tests/
```
`selection.json` = `[{"clip": "clip-002034", "uid": "7", "cat": "bottle"}, ...]`. Benchmark inputs live
at `/workspace/datasets/hot3d/rc_input_<num>_<clip>/` (rgb.mp4, frames/, depth_png/, intrinsics.npy).

## Load-bearing caveats (rendering caught every one; a metric caught none)
- **Convention traps:** HOT3D ships two model sets — pose the `object_models_eval` GLBs (meters), not
  the display GLBs; the only `MANO_LEFT.pkl` on this box is **fabricated** (mirrored right hand); HOI4D
  poses annotate the CAD **bbox centre**, not the origin; poses/cameras are quaternion-**wxyz** world
  transforms.
- **Depth substrate is the biggest lever:** GT sensor/ray-cast depth when it exists, VGGT-Omega
  otherwise — never per-frame monocular (it breathes).
- Pre-Blackwell envs (`forehoi`, `hort`, `hold`, `easyhoi`, `daid`, cu118/cu121) have no sm_120 kernels
  — dead. Revival recipes: [`compare/hot3d/docs/T4_NOTES.md`](compare/hot3d/docs/T4_NOTES.md).
- The acceptance gate (`leaderboard.py`) **under-credits real wins** — always keep the raw per-clip
  chamfer/rot_traj numbers, not just the verdict.
