# hoi_recon — Hand-Object Interaction Reconstruction workbench

Recover a **4D hand-object interaction** (per-frame hand + object mesh + 6DoF object
trajectory) from egocentric/monocular video. This repo holds our own
reconstruction **pipeline**, a set of third-party **methods** revived for
comparison, and the **study** that benchmarks them all against mocap-grade ground
truth on HOT3D.

## Start here

| If you want… | Read |
|---|---|
| The whole story + all-method comparison + lessons | **[`compare/hot3d/README.md`](compare/hot3d/README.md)** (overall summary) and [`compare/hot3d/docs/REFLECTION.md`](compare/hot3d/docs/REFLECTION.md) (full journey) |
| The current best strategy: full workflow + what's left | **[`BEST_STRATEGY.md`](BEST_STRATEGY.md)** |
| The head-to-head numbers | [`compare/hot3d/scores/LEADERBOARD.md`](compare/hot3d/scores/LEADERBOARD.md), [`compare/hot3d/docs/T4_RESULTS.md`](compare/hot3d/docs/T4_RESULTS.md) |
| To run the pipeline | [`render_and_compare/README.md`](render_and_compare/README.md) + [`RUN_REAL.md`](render_and_compare/RUN_REAL.md) |

**TL;DR of the findings:** our optimization pipeline (`render_and_compare`, best arm
`icpjgr`) cut mean chamfer 3.2× on HOT3D and never fails catastrophically; learned
RGB-D pose estimators (Any6D, FoundationPose) that *consume the calibrated depth*
are more *accurate* per-frame but less *robust*; the **combined method** (learned
pose + our temporal layer) gets both. Learned methods that *discard* depth (HORT,
ForeHOI) reconstruct good shape but misplace it. Full detail in the summary above.

---

## Repository map

### Our methods (built here)

| Folder | What | Status |
|---|---|---|
| [`render_and_compare/`](render_and_compare/) | The main compositional pipeline: depth substrate → SAM-3D object mesh → joint depth+silhouette registration → contact optimization. Became the HOT3D best arm **icpjgr**. Docs: [DESIGN](render_and_compare/DESIGN.md), [REPRODUCE](render_and_compare/REPRODUCE.md), [RUN_REAL](render_and_compare/RUN_REAL.md), [RESEARCH_DIRECTIONS](render_and_compare/RESEARCH_DIRECTIONS.md). Run dirs in `runs/`, configs in `configs/`. | ✅ runnable (env `rc5090`) |
| [`egoaero/`](egoaero/) | EgoAERO Part A — asset-free egocentric HOI reconstruction (adaptive contact optimization). | 🟡 recon runnable in mock |

### The comparison study — [`compare/`](compare/)

| Path | What |
|---|---|
| **[`compare/hot3d/`](compare/hot3d/)** | **The main study.** Frozen 6-clip HOT3D benchmark, the pipeline arms (icpj→icpjgr), the learned-method bake-off (HORT/ForeHOI/FoundationPose/Any6D), and the combined method. Self-contained README + `docs/` + `scores/` + `overlays/`. |
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
| [`BEST_STRATEGY.md`](BEST_STRATEGY.md) | Concise strategy doc: the arc in brief, the full end-to-end workflow of today's best strategy (icpjgr + the combined-method frontier), the live numbers, and the ranked what's-left-to-improve. |
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

Frozen 6-clip HOT3D bench (mocap-grade GT), inputs at
`/workspace/datasets/hot3d/rc_input_<num>_<clip>/` (rgb.mp4, frames/, ray-cast
depth_png/, intrinsics.npy). Each method writes a run to
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
