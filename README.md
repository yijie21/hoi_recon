# hoi_recon — Hand-Object Interaction Reconstruction workbench

Recover a **4D hand-object interaction** (per-frame hand + object mesh + 6DoF object
trajectory) from egocentric/monocular video. This repo holds our own
reconstruction **pipeline**, a set of third-party **methods** revived for
comparison, and the **study** that benchmarks them all against mocap-grade ground
truth on HOT3D.

## ⭐ Best method (full 4D HOI) — how to run it

The best reconstruction pairs the **best object** track with the **best hand**, then renders
both backprojected together:

- **Object → `fpauto`** (FoundationPose auto: `track`+`register_each`, drift-gated). Best object
  backprojection — beats the learned `any6dp` on **both** placement (mean chamfer **8.2 vs 11.5 mm**)
  and rotation (mean rot_traj-p90 **88.6 vs 111.6°**). Exception: on sustained in-hand-rotated
  symmetric objects (the potato masher) `icpjgr` still owns rotation — use it there.
  Detail: [`compare/hot3d/docs/T6_NOTES.md`](compare/hot3d/docs/T6_NOTES.md).
- **Hand → the hand-reprojection optimizer** (`joint_opt.py --freeze_object`, kp2d-aligned). Best
  hand backprojection — the MANO hand lands on the observed hand at **1.9–3.8 px** 2D reprojection
  (from 5–108 px before), across all 6 clips.
  Detail: [`render_and_compare/docs/adr/0001-hand-reprojection-optimizer.md`](render_and_compare/docs/adr/0001-hand-reprojection-optimizer.md).

**Overlays** (`compare/hot3d/overlays/`): `hoi_best_<clip>.mp4` = [ original | object | object+hand ];
`hand_cmp_<clip>.mp4` = hand before/after; `rc_vs_gt_<clip>_*.mp4` = object vs mocap GT.

**Run it on a HOT3D clip** (envs: `rc5090` pipeline/eval/overlay, `forehoi5090` FoundationPose,
`sam3d5090` SAM-3D + hand optimizer). From `compare/hot3d/`, e.g. the bottle clip
(`cat=bottle_bbq num=002034`, rc_input `002034_bottle_bbq`, clip `clip-002034`):

```bash
RC5=/workspace/miniconda3/envs/rc5090/bin/python
FH5=/workspace/miniconda3/envs/forehoi5090/bin/python
RUN=../../render_and_compare/runs/hot3d_bottle_bbq_002034_icpjgr
FP=../../render_and_compare/runs/hot3d_bottle_bbq_002034_fpauto
RC=/workspace/datasets/hot3d/rc_input_002034_bottle_bbq

# 1. Full pipeline once (arm icpjgr): produces the hand stages + the SAM-3D mesh every arm shares.
$RC5 run_batch.py selection_fixed.json --arm icpjgr \
     --config ../../render_and_compare/configs/real_forehoi_icp_joint_grasp.yaml
# 2. Best OBJECT — FoundationPose auto on the icpjgr uniform-metric mesh (env forehoi5090).
$FH5 run_fp_hot3d.py $RC $RUN $FP --mode auto
# 3. Best HAND — freeze the object, align MANO to kp2d + hand-silhouette (spawns sam3d5090).
$RC5 run_hand_reproj.py $RUN $RC
# 4. Combined HOI overlay: best object (fpauto) + best hand, splatted together.
$RC5 make_hoi_best_overlay.py $FP $RUN overlays/hoi_best_bottle_bbq_002034.mp4
# 5. Score vs mocap GT — object (chamfer/rot) and hand (chamfer/2D-reproj).
$RC5 gt_pose_eval_hot3d.py $RC $FP
$RC5 gt_hand_eval_hot3d.py /workspace/datasets/hot3d/clips/clip-002034 $RC \
     before=$RUN after=$RUN/hand_reproj_opt/out.npz
```

For the potato masher, use `$RUN` (icpjgr) as the object source in steps 4–5 instead of `$FP`.

## Start here

| If you want… | Read |
|---|---|
| The whole story + all-method comparison + lessons | **[`compare/hot3d/README.md`](compare/hot3d/README.md)** (overall summary) and [`compare/hot3d/docs/REFLECTION.md`](compare/hot3d/docs/REFLECTION.md) (full journey) |
| The current best strategy: full workflow + what's left | **[`BEST_STRATEGY.md`](BEST_STRATEGY.md)** |
| The latest campaign: learned-core integration, the Pareto finding, what's *not* fixable and why | **[`compare/hot3d/docs/T5_NOTES.md`](compare/hot3d/docs/T5_NOTES.md)** |
| The head-to-head numbers | [`compare/hot3d/scores/LEADERBOARD.md`](compare/hot3d/scores/LEADERBOARD.md), [`compare/hot3d/docs/T4_RESULTS.md`](compare/hot3d/docs/T4_RESULTS.md) |
| To run the pipeline | [`render_and_compare/README.md`](render_and_compare/README.md) + [`RUN_REAL.md`](render_and_compare/RUN_REAL.md) |

**TL;DR of the findings** (superseded for *use* by the ⭐ best-method box above; kept as the
campaign narrative): our optimization pipeline (`render_and_compare`, best integrated arm
`icpjgr`) cut mean chamfer 3.2× on HOT3D and never fails catastrophically. A learned
RGB-D pose core (Any6D) is now integrated *inside* the pipeline (arm `any6dp`, one run) —
it wins placement decisively (chamfer better on **9/11 clips**, new-clip median 7.1→2.8
mm at 2× scale) but trades rotation. `any6dp` and `icpjgr` are a proven
**placement-vs-rotation Pareto pair**: four independent attempts to give the learned core
icpjgr's rotation all failed — the rotation/attitude/texture axis is a **hard wall**
(symmetric-object orientation is under-constrained by depth, and on hand-held objects the
one discriminating feature is exactly what the hand occludes). Learned methods that
*discard* depth (HORT, ForeHOI) get good shape but misplace. Latest campaign in full:
[`compare/hot3d/docs/T5_NOTES.md`](compare/hot3d/docs/T5_NOTES.md).

---

## Repository map

### Our methods (built here)

| Folder | What | Status |
|---|---|---|
| [`render_and_compare/`](render_and_compare/) | The main compositional pipeline: depth substrate → SAM-3D object mesh → joint depth+silhouette registration → contact optimization. Rotation-robust best arm **icpjgr** (`BEST_ARM`); stage-4 `pose_core: learned` also hosts the **`any6dp`** arm (Any6D pose core + temporal layer, `configs/real_any6d.yaml`) — placement-optimal. Docs: [DESIGN](render_and_compare/DESIGN.md), [REPRODUCE](render_and_compare/REPRODUCE.md), [RUN_REAL](render_and_compare/RUN_REAL.md), [RESEARCH_DIRECTIONS](render_and_compare/RESEARCH_DIRECTIONS.md). Run dirs in `runs/`, configs in `configs/`. | ✅ runnable (env `rc5090`) |
| [`egoaero/`](egoaero/) | EgoAERO Part A — asset-free egocentric HOI reconstruction (adaptive contact optimization). | 🟡 recon runnable in mock |

### The comparison study — [`compare/`](compare/)

| Path | What |
|---|---|
| **[`compare/hot3d/`](compare/hot3d/)** | **The main study.** HOT3D benchmark (6 frozen clips, scaled to 12 for validation), the pipeline arms (icpj→icpjgr), the learned-method bake-off (HORT/ForeHOI/FoundationPose/Any6D), the in-pipeline learned core (`any6dp`), and the rotation-fixing dead ends. Self-contained README + `docs/` (incl. `T5_NOTES.md`) + `scores/` + `overlays/`. |
| [`compare/hoi4d/`](compare/hoi4d/) | The earlier HOI4D-era comparison (kettle_N15): GT-pose eval, photometric checks, the bbox-centre convention discovery. Superseded by HOT3D but documents the traps. |
| [`compare/adapters/`](compare/adapters/) | Adapters that map each method's raw output into a common scene format for the viewer. |
| [`compare/scenes/`](compare/scenes/), `backproj/`, `daid_run/`, `method_notes/` | Common scene bundles, backprojection overlays, do-as-i-do run scripts, per-method notes. |

### Third-party methods (cloned for the bake-off)

Revived on this Blackwell (sm_120) box where feasible — see
[`compare/hot3d/docs/T4_NOTES.md`](compare/hot3d/docs/T4_NOTES.md) for the
feasibility matrix and Blackwell recipes ("one env for all" is infeasible).

| Folder | Method | Bake-off outcome |
|---|---|---|
| [`any6d/`](any6d/) | Any6D (CVPR'25) — model-free 6D pose from RGB-D anchor | ✅ revived (`forehoi5090`); **beats icpjgr chamfer 5/6** |
| [`forehoi/`](forehoi/) | ForeHOI — feed-forward object recon from HOI video (also bundles FoundationPose in `wheels/`) | ✅ revived (`forehoi5090`); good shape, misplaced; FP = strongest per-frame pose |
| [`hort/`](hort/) | HORT — monocular hand-held object recon | ✅ revived (`hort5090`); no metric scale, loses |
| [`do-as-i-do/`](do-as-i-do/) | Do-as-I-Do — 7-stage HOI pipeline | ⛔ infeasible in budget (6–8 CUDA extensions) |
| [`hold/`](hold/) | HOLD (CVPR'24) — per-video HOI optimization | ⛔ infeasible (kaolin wheel, walled weights, unbuilt preprocessing) |
| [`easyhoi/`](easyhoi/) | EasyHOI — in-the-wild single-image HOI | ⛔ infeasible (5 CUDA rebuilds, LISA 26 GB) |

### Reference docs & notes

| File | What |
|---|---|
| [`BEST_STRATEGY.md`](BEST_STRATEGY.md) | Concise strategy doc: the arc in brief, the full end-to-end workflow (icpjgr rotation-robust core + the `any6dp` learned placement core), the live numbers, and the roadmap with every item's outcome (1–6 done or definitively resolved). |
| [`compare/hot3d/docs/T5_NOTES.md`](compare/hot3d/docs/T5_NOTES.md) | The learned-core integration (item 1) + the four rotation-fixing negatives (items 2–4) + the 12-clip scale validation (item 6), with the evidence for each. |
| [`hoi.md`](hoi.md) | HOI paper reading list (the T4 candidate source). |
| [`allinone.md`](allinone.md) | Paper-search results: HOI datasets with RGB-D + CAD + pose trajectories. |
| [`docs/superpowers/`](docs/superpowers/) | The spec + implementation plan for the HOT3D improvement campaign. |
| [`idea-loop-reports/`](idea-loop-reports/) | Generated research-idea reports (e.g. jitter-free 4DGS-HOI). |
| [`CLAUDE.md`](CLAUDE.md) | Agent orchestration guide for this repo. |

## Environments (conda, on the RTX 5090 / Blackwell box)

| Env | For |
|---|---|
| `rc5090` | the pipeline (`render_and_compare`) + HOT3D eval/harness (torch 2.11+cu128) |
| `sam3d5090` | SAM-3D object generation (torch 2.8+cu128; the donor stack for the others) |
| `forehoi5090` | ForeHOI + FoundationPose + Any6D (cloned from sam3d5090) |
| `hort5090` | HORT (cloned from sam3d5090 + pointnet2_ops) |

Dead pre-Blackwell envs (`forehoi`, `hort`, `hold`, `easyhoi`, `daid`, cu118/cu121)
have no sm_120 kernels — do not use. See `compare/hot3d/docs/T4_NOTES.md`.

## The benchmark

Frozen 6-clip HOT3D bench (mocap-grade GT), scaled to **12 clips** (via HOT3D-HIT
motion/FOV probing → `compare/hot3d/selection_all.json`) for the item-1 validation.
Inputs at `/workspace/datasets/hot3d/rc_input_<num>_<clip>/` (rgb.mp4, frames/,
ray-cast depth_png/, intrinsics.npy). Each method writes a run to
`render_and_compare/runs/hot3d_<clip>_<method>/stage8_eval/pseudo_gt.npz`
(`{obj_verts, obj_faces, obj_poses[T,4,4]}`), scored by
`compare/hot3d/gt_pose_eval_hot3d.py` against the GT trajectory.

## Method contract (for adding a new method)

Each method folder is a self-contained peer: its own env/build, deps, tests, docs,
and gitignored runtime artifacts (`runs/`, `checkpoints/`, `third_party/`). It takes
the same input (monocular/egocentric RGB, optional intrinsics + depth), writes the
`stage8_eval/pseudo_gt.npz` output layout, and reports the shared metrics (chamfer,
centroid, rot_traj, canonical-shape ICP). To add one: clone into a sibling folder,
implement the contract, add a row to the tables above, and wire a runner under
`compare/hot3d/`.
