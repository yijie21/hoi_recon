# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Orchestration workflow
You (Fable) are the orchestrator. Plan, decompose, synthesize.
Reasoning-heavy phases → deep-reasoner. Mechanical work → fast-worker.
High-stakes decisions: task deep-reasoner (Opus) with the problem, think it through thoroughly,
and synthesize a concise conclusion you can act on. Keep your own context lean.

## What this branch is
This project reconstructs a hand using an object (a **4D hand-object interaction**) from
egocentric video, tested on **HOT3D** (a benchmark with motion-capture ground truth).

**`main` carries only the shipped best method.** The full research history — alternative
methods (Any6D/`any6dp`, HORT, ForeHOI, …), the HOI4D-era study, the egoaero package, the
hoi_flow refiner, campaign notes, and the multi-method leaderboard — lives on the **`dev`
branch**. Recover any historical file with `git checkout dev -- <path>`.

The shipped method (names decoded in [`GLOSSARY.md`](GLOSSARY.md)):
1. **Registration pipeline (`icpjgr`)** — fit the SAM-3D object mesh to depth+silhouette,
   close the grasp. Rotation-robust. Config `render_and_compare/configs/real_forehoi_icp_joint_grasp.yaml`.
2. **Learned object core (`fpauto`, FoundationPose)** — best placement (~8 mm) and rotation;
   ships on 5 of 6 clips (`compare/hot3d/run_fp_hot3d.py`, env `forehoi5090`). The
   **potato masher** (spinning, near-symmetric) keeps the registration track: symmetric-object
   rotation is genuinely under-constrained by depth — this was investigated exhaustively
   (see dev branch); do not reopen rotation priors here.
3. **Hand-reprojection optimizer** — aligns the MANO hand to the observed hand pixels,
   2–4 px (`compare/hot3d/run_hand_reproj.py`, env `sam3d5090`).

Read in order: [`GLOSSARY.md`](GLOSSARY.md), [`README.md`](README.md) (method + run recipe),
[`compare/hot3d/RESULTS.md`](compare/hot3d/RESULTS.md) (numbers),
[`render_and_compare/REPRODUCE.md`](render_and_compare/REPRODUCE.md) (fresh-machine setup).
Envs + convention traps are in the recalled `hoi-recon-*` memories.

## Repository layout
- **`render_and_compare/`** — the pipeline (installable package `hoi_recon`). 9 cached,
  resumable stages under `hoi_recon/stages/stage{0..8}_*.py`, orchestrated by `hoi_recon/pipeline.py`,
  entered via `hoi_recon/cli.py` (`python -m hoi_recon.cli`). Configs in `configs/`, run outputs in
  `runs/` (gitignored). See its [`README.md`](render_and_compare/README.md) + [`DESIGN.md`](render_and_compare/DESIGN.md).
- **`compare/hot3d/`** — the HOT3D benchmark harness: `run_batch.py` (driver),
  `make_rc_input.py` (HOT3D→pipeline adapter: fisheye rectify + GT-depth raycast),
  `run_fp_hot3d.py` (fpauto arm), `run_hand_reproj.py` (hand optimizer),
  `gt_pose_eval_hot3d.py` / `gt_hand_eval_hot3d.py` (scorers). Also hosts the separate
  clean-training-clips pipeline (`HOI_CLIPS.md` + `gen_clean_clips.py` etc.).

Pipeline output contract: `runs/<name>/stage8_eval/pseudo_gt.npz` =
`{obj_verts, obj_faces, obj_poses[T,4,4]}`. Stages are **cached** — each writes
`stage<N>_*/` and is skipped if present (`--force` to recompute, `--stages` to select).

**Every comparison is mesh-controlled:** reuse the incumbent's stage 0–3 (`run_batch.py`
auto-promotes identical-mask stage2/stage3 dirs) so SAM-3D GPU nondeterminism can't
masquerade as method signal.

## Commands
Conda envs (`rc5090`, `sam3d5090`, `forehoi5090`) are the documented contract but may be
**absent on a recycled box** — verify with `conda env list`; storage under `/workspace` is
not guaranteed persistent (see `/workspace/CLAUDE.md` §3). Build recipes:
`render_and_compare/REPRODUCE.md`.

```bash
# --- Pipeline (render_and_compare) ---
# Mock mode: full pipeline on a synthetic clip, no weights needed — the fast sanity check
python -m hoi_recon.cli --out runs/demo --mock --num-frames 48
# Single real arm end to end (needs RC_GT_DEPTH_DIR / RC_GT_INTRINSICS set for HOT3D)
python -m hoi_recon.cli --video rgb.mp4 --out runs/<name> --real \
    --config configs/real_forehoi_icp_joint_grasp.yaml --depth gt --object-prompt <x> <y>
# Re-run only some stages of an existing run
python -m hoi_recon.cli --out runs/<name> --real --stages 4- --force

# --- HOT3D benchmark harness (from compare/hot3d/; env rc5090 unless stated) ---
python run_batch.py selection.json --arm icpjgr --config \
    ../../render_and_compare/configs/real_forehoi_icp_joint_grasp.yaml   # adapter→pipeline→score
python run_fp_hot3d.py <ABS rc_input> <ABS icpjgr_run> <ABS fp_run> --mode auto  # fpauto (env forehoi5090)
python run_hand_reproj.py <ABS run_dir> <ABS rc_input> [<out_dir>]       # hand optimizer (env sam3d5090)
python gt_pose_eval_hot3d.py <rc_input> <run>                            # score object vs mocap GT
python make_hoi_best_overlay.py <fp_run> <icpjgr_run> overlays/hoi_best_<clip>.mp4

# --- Tests (pytest, mostly mock — no GPU/weights) ---
cd render_and_compare && python -m pytest tests/            # or a single file / -k <name>
cd compare/hot3d && python -m pytest tests/
```
`selection.json` = the 6 bench clips. Benchmark inputs live at
`/workspace/datasets/hot3d/rc_input_<num>_<clip>/` (rgb.mp4, frames/, depth_png/, intrinsics.npy).

## Load-bearing caveats (rendering caught every one; a metric caught none)
- **Convention traps:** HOT3D ships two model sets — pose the `object_models_eval` GLBs (meters),
  not the display GLBs; a `MANO_LEFT.pkl` fabricated by mirroring the right hand is NOT valid;
  poses/cameras are quaternion-**wxyz** world transforms.
- **Depth substrate is the biggest lever:** GT sensor/ray-cast depth when it exists, VGGT-Omega
  otherwise — never per-frame monocular (it breathes).
- Pre-Blackwell envs (cu118/cu121) have no sm_120 kernels on this box — dead.
- Aggregate gate metrics **under-credit real wins** — always keep the raw per-clip
  chamfer/rot_traj numbers, not just a verdict.
- Render and eyeball an overlay after any pose-affecting change.
