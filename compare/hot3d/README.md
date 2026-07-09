# compare/hot3d — HOT3D benchmark, pipeline arms & learned-method bake-off

Everything for the HOT3D object-recovery study: the frozen 6-clip benchmark, the
optimization-pipeline arms (icpj → icpjgr), the learned-method comparison
(HORT / ForeHOI / FoundationPose / Any6D), and the combined method.

**Start here:** [`docs/REFLECTION.md`](docs/REFLECTION.md) — the full experience
log, method comparison, and lessons. [`docs/T4_RESULTS.md`](docs/T4_RESULTS.md) —
the head-to-head numbers. [`scores/LEADERBOARD.md`](scores/LEADERBOARD.md) — the
per-clip table for every arm/method.

## Directory skeleton

```
compare/hot3d/
├── README.md                  # this file
├── docs/                      # write-ups (read these)
│   ├── REFLECTION.md          #   full journey + method comparison + lessons
│   ├── T4_RESULTS.md          #   learned-method bake-off, head-to-head table
│   └── T4_NOTES.md            #   feasibility matrix (which methods revivable on Blackwell)
├── scores/                    # quantitative results
│   ├── LEADERBOARD.md         #   generated per-clip table (all arms + methods)
│   ├── BEST_ARM               #   current best pipeline arm (icpjgr)
│   └── batch_summary_<X>.json #   per-clip medians for arm/method X
├── overlays/                  # GT-vs-estimate videos (visual results)
│   ├── rc_vs_gt_<clip>_<X>.mp4 #   pipeline/method X on <clip> vs mocap GT
│   ├── hort_<clip>.mp4         #   HORT reprojection overlays
│   └── gt_overlay_hot3d_*.mp4  #   GT-only sanity overlays
│
├── ── data / harness ──
├── make_rc_input.py           # adapter: Aria fisheye -> pinhole RGB + ray-cast GT depth
├── run_batch.py               # run a pipeline arm on the 6-clip bench, score, overlay
├── gt_pose_eval_hot3d.py      # score a run vs mocap GT (chamfer/centroid/rot_traj/shape)
├── leaderboard.py             # gate + render scores/LEADERBOARD.md
├── probe_clips.py             # select single-object interaction clips from HOT3D
│
├── ── overlays / viz ──
├── make_rc_vs_gt_overlay.py   # GT-vs-estimate side-by-side video for any run
├── gt_overlay_hot3d.py        # GT objects+hands overlay (sanity check)
├── hort_hot3d_overlay.py      # HORT-specific reprojection overlay
│
└── ── methods ──
    ├── run_any6d_hot3d.py     # Any6D per-frame RGB-D pose on the bench (mesh-controlled)
    └── combined_refine.py     # combined method: learned pose + temporal flip-fix + jitter smooth
```

Run dirs live in `render_and_compare/runs/hot3d_<clip>_<X>/` (X = arm/method);
each has `stage8_eval/pseudo_gt.npz` = `{obj_verts, obj_faces, obj_poses[T,4,4]}`,
the input to `gt_pose_eval_hot3d.py`. Frozen bench inputs are at
`/workspace/datasets/hot3d/rc_input_<num>_<clip>/` (rgb.mp4, frames/, depth_png/,
intrinsics.npy) — never regenerate them mid-study.

## Arms & methods (best-known result per clip in `scores/LEADERBOARD.md`)

| id | what | consumes GT depth? | one-liner |
|---|---|---|---|
| icpj | baseline pipeline | ✓ | SAM-3D mesh + depth+silhouette joint ICP |
| icpjs | + hand-aware seg (T1) | ✓ | fixed the catastrophic mask failures |
| icpjp | + chroma attitude (T2) | ✓ | fixed textured-symmetric azimuth (bottle) |
| **icpjgr** | + grasp-rigidity (T3) | ✓ | **best pipeline arm** |
| forehoi | ForeHOI (learned) | ✗ | great shape, wrong placement (self-estimates depth) |
| fp | FoundationPose (learned) | ✓ | strongest per-frame pose; default tracker drifts |
| any6d | Any6D (learned) | ✓ | beats icpjgr chamfer 5/6; per-frame flip outliers |
| combined | Any6D + temporal layer | ✓ | best-of-both (bottle beats icpjgr on all metrics) |

## Reproduce

```bash
PY=/workspace/miniconda3/envs/rc5090/bin/python
# score any run vs GT:
$PY gt_pose_eval_hot3d.py /workspace/datasets/hot3d/rc_input_002034_bottle_bbq \
    ../../render_and_compare/runs/hot3d_bottle_bbq_002034_icpjgr
# re-render the leaderboard:
$PY leaderboard.py render        # -> scores/LEADERBOARD.md
# combined method on a learned run:
$PY combined_refine.py <any6d_run_dir> <out_run_dir>   # trans-smooth on, rot-smooth off
```

Envs: `rc5090` (pipeline + eval), `forehoi5090` (ForeHOI/FoundationPose/Any6D on
Blackwell), `hort5090` (HORT). See `docs/T4_NOTES.md` for the Blackwell revival
recipes; "one env for all" is infeasible (cu118/cu121/cu130 + custom CUDA).
