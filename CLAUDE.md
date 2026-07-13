# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Orchestration workflow
You (Fable) are the orchestrator. Plan, decompose, synthesize.
Reasoning-heavy phases → deep-reasoner. Mechanical work → fast-worker.
High-stakes decisions: task deep-reasoner (Opus) with the problem, think it through thoroughly,
and synthesize a concise conclusion you can act on. Keep your own context lean.

## Where things stand (start here)
This project reconstructs a hand using an object (a **4D hand-object interaction**) from
egocentric video, tested on **HOT3D** (a benchmark with motion-capture ground truth).

**Every method name and metric is decoded in [`GLOSSARY.md`](GLOSSARY.md) — read it first.**
The three object methods, in plain terms:
- **Registration pipeline (`icpjgr`)** — our hand-built method (fit object to depth+silhouette,
  then close the grasp). Rotation-robust, never fails badly. The default `BEST_ARM`.
  Config `render_and_compare/configs/real_forehoi_icp_joint_grasp.yaml`.
- **Learned object core (`fpauto`, FoundationPose)** — best object placement (~8 mm) and rotation;
  the object track we ship on 5 of 6 clips. `compare/hot3d/run_fp_hot3d.py`, env `forehoi5090`.
- **Earlier learned core (`any6dp`, Any6D)** — good placement, weaker rotation; superseded by `fpauto`.

The best result ships the **learned object core + the hand optimizer** (aligns the MANO hand to
the observed hand pixels, 2–4 px), except the **potato masher** (a spinning symmetric object),
which keeps the registration pipeline for rotation.

**Do NOT reopen the rotation-prior dead end** (temporal / grasp-rigidity / anchor-attitude /
texture-baking priors — all tested and failed). `fpauto` is a better *estimator*, not a corrective
prior. Symmetric-object rotation is genuinely under-constrained by depth.

Read in order: [`GLOSSARY.md`](GLOSSARY.md) (names), [`BEST_STRATEGY.md`](BEST_STRATEGY.md)
(strategy + what's been tried), the campaign notes under `compare/hot3d/docs/`
([`T5_NOTES.md`](compare/hot3d/docs/T5_NOTES.md), [`T6_NOTES.md`](compare/hot3d/docs/T6_NOTES.md)),
then [`README.md`](README.md). Numbers: [`compare/hot3d/scores/LEADERBOARD.md`](compare/hot3d/scores/LEADERBOARD.md).
Reproduce the best result from a fresh clone: [`render_and_compare/REPRODUCE.md`](render_and_compare/REPRODUCE.md).
Envs + convention traps are in the recalled `hoi-recon-*` memories.

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

# --- HOT3D benchmark harness (from compare/hot3d/; envs noted per-line — rc5090 unless stated) ---
python run_batch.py selection.json --arm icpjgr --config \
    ../../render_and_compare/configs/real_forehoi_icp_joint_grasp.yaml   # adapter→pipeline→overlay→score
python run_any6d_hot3d.py <ABS rc_input> <ABS icpjgr_run> <ABS any6d_run>  # any6dp learned core (env forehoi5090)
python combined_refine.py <any6d_run> <combined_run>                       # flip-fix + jitter smooth
python run_fp_hot3d.py <ABS rc_input> <ABS icpjgr_run> <ABS fp_run> --mode auto  # fpauto learned core (env forehoi5090)
./score_fp_modes.sh <ABS rc_input> <fp_run> <ABS icpjgr_run>              # score register_each/track/fuse (cache-instant)
python run_hand_reproj.py <ABS run_dir> <ABS rc_input> [<out_dir>]        # hand→kp2d 2D-reproj optimizer (env sam3d5090)
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
