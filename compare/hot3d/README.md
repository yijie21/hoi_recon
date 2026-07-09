# HOI object recovery on HOT3D — overall summary

The goal throughout: **recover an object's 6DoF trajectory + shape from an
egocentric hand-object video**, measured against mocap-grade ground truth. This
folder holds the whole study — the benchmark, the optimization pipeline, the
learned-method bake-off, and the combined method. This page is the overall
summary; the detailed journey is in [`docs/REFLECTION.md`](docs/REFLECTION.md),
the head-to-head numbers in [`docs/T4_RESULTS.md`](docs/T4_RESULTS.md), the live
table in [`scores/LEADERBOARD.md`](scores/LEADERBOARD.md).

---

## The arc, start to finish

**1. HOI4D → HOT3D (why the benchmark changed).** Early work was on HOI4D
(kettle_N15), but its annotations were the bottleneck — ±1.5 cm / ±20 px scatter,
a 6.4 mm GT-vs-depth floor — so sub-cm method differences were undecidable. Three
convention traps there were caught only by *rendering and eyeballing*, never by a
metric: the pose annotates the CAD **bbox centre**; the only `MANO_LEFT.pkl` on the
box is a **fabricated** mirror of the right hand; and HOT3D ships two model sets
whose canonicals disagree (**pose the eval GLBs, not the display GLBs**). We moved
to **HOT3D** for its mocap-grade GT. It has no depth sensor, so an adapter
(`make_rc_input.py`) rectifies the Aria fisheye onto a virtual pinhole camera and
**ray-casts GT depth** from the posed CADs + hand meshes — giving a *calibrated
RGB-D* benchmark, which later turned out to be the decisive property.

**2. The optimization pipeline (icpj → icpjgr).** A gated, subagent-driven campaign
built the pipeline in stacked arms, each accepted only if it passed a lexicographic
gate with no clip regressing:

- **T1 hand-aware segmentation** — the big win. Both catastrophic clips were stage-1
  SAM2 mask failures on hand-held objects (mug absorbed the forearm; spatula leaked
  onto the table). Root cause the metrics hid: the prompt pixel can land *on the
  occluding hand*. Fix: track hands as SAM2 objects and subtract them, prompt K≤5
  candidate clicks vetoed against the hand mask, pick the best-scoring track.
  → **mug 60.7→7.0, spatula 158.8→20.5 mm.**
- **T2 chroma attitude search** — depth is azimuth-blind on symmetric objects, so
  scoring rotation hypotheses on **LAB-chroma** (does the texture line up?) and
  **spread-gating** (only act when texture is discriminative) fixed the bottle and
  self-disabled elsewhere. → **bottle 19.9→9.9 mm.**
- **T3 grasp-rigidity** — during fast in-hand rotation the object co-rotates with
  the wrist; a contact-detected, speed-gated rigidity term gave a small rotation-tail
  gain.

Net baseline→best (**icpjgr**): **worst-clip chamfer 158.8→21.2 mm, mean chamfer
49.1→15.5 mm (3.2×).** Rotation median stayed ~flat, floored by the symmetric cube.

**3. The learned-method bake-off (T4).** Can a learned method beat the pipeline?
Five cloned repos + two external methods were revived on Blackwell (sm_120) — "one
env for all" is infeasible (cu118/cu121/cu130 + custom CUDA), so each was built by
cloning the working `sam3d5090` stack. The decisive variable was **whether the
method consumes the calibrated RGB-D**:

- **Discard the depth → lose by 10–160×.** HORT (mono, no metric scale, wrong object
  on 4/6) and ForeHOI (great *shape* — 2.7–16 mm — but self-estimated depth places
  it at the wrong 3D location).
- **Consume the depth → beat icpjgr on accuracy.** Given the *same* SAM-3D mesh +
  depth + mask, **Any6D wins chamfer on 5/6** clips (often 2–3× lower), and
  **FoundationPose** wins decisively where its tracker holds (rotation down to ~3°).

The finding that reframed the project: a learned per-frame RGB-D estimator is *more
accurate* than the hand-built registration; the pipeline's real contribution is
**robustness through temporal optimization** — it never drifts (unlike FP `track`)
and never symmetry-flips (unlike Any6D's per-frame independence).

**4. The combined method — best of both.** `combined_refine.py` post-processes the
learned per-frame poses with the pipeline's temporal layer: **symmetry-flip
resolution** (per frame pick the symmetry-equivalent rotation closest to its
neighbours — fixes flips without touching placement) plus **data-anchored
translation jitter smoothing** (a universal free win: position jitter p90 66→7 mm at
<1 mm chamfer; rotation smoothing is left off — it distorts the residual flips). On
the bottle this **beats icpjgr on every metric** (chamfer 5.2 vs 9.9, rot p90 17 vs
73) by keeping Any6D's accuracy and fixing its rotation.

---

## Headline comparison — chamfer median (mm), mesh-controlled

| clip | icpjgr | HORT | ForeHOI | FoundationPose | Any6D | combined |
|---|---|---|---|---|---|---|
| bottle_bbq | 9.9 | ✗ | 392 | 3.2ʳ / 93ᵈ | 5.2 | **5.2** |
| mug_white | 7.0 | ~ | 1147 | **2.3** | 3.4 | 3.4 |
| vase | 17.7 | ✗ | 212 | 6.6 | 6.4 | 6.6 |
| spatula_red | 21.2 | ✗ | 178 | 11.2 | 9.6 | 9.6 |
| potato_masher | 18.8 | ✗ | 704 | 8.6ʳ / 598ᵈ | 12.0 | 12.2 |
| puzzle_toy (cube) | 18.5 | ~ | 391 | 18.9 | 21.3 | 21.3 |

ᵈ default `track` drifts · ʳ `register_each` recovers · ✗ wrong object / no scale ·
~ plausible but unscaled. Rotation & full table in `scores/LEADERBOARD.md`.

## Lessons (the durable takeaways)

- **Calibration is a moat.** On calibrated RGB-D, the method that uses it wins;
  monocular learned methods are strong at *shape* but cannot place.
- **Accuracy and robustness are different axes.** Learned per-frame estimators win
  accuracy; temporal optimization wins robustness. The product is the combination —
  swap stage-4's registration core for a learned estimator, keep the temporal layer.
- **Mesh-control every comparison.** SAM-3D generation is nondeterministic; reusing
  the incumbent's stage 0–3 was the only way to attribute a difference to the method.
- **Render and eyeball.** Every convention trap and load-bearing bug was caught by
  looking at output, not by a metric.
- **The gate metric is not the goal.** Worst-chamfer/median-rot_traj under-credited
  T2 (chamfer), T3 (tail), and the entire Any6D result (placement).

## What to build next

1. Swap stage-4's registration core for a learned per-frame estimator
   (FoundationPose-register / Any6D) inside the pipeline, keeping the temporal layer.
2. Flip-aware SO(3) rotation smoother + per-frame depth-anchored basin selection
   (fixes the masher's sustained wrong-basin, which pure temporal smoothing can't).
3. Hand-aware segmentation remains the highest-leverage upstream fix for any method.
4. Better SAM-3D texture fidelity so the photometric/attitude terms work on more
   objects (the cube needs the *real* sticker layout).

---

## Directory structure

```
compare/hot3d/
├── README.md                  # this overall summary
├── docs/                      # REFLECTION.md (full journey), T4_RESULTS.md, T4_NOTES.md
├── scores/                    # LEADERBOARD.md, BEST_ARM, batch_summary_<arm>.json
├── overlays/                  # rc_vs_gt_<clip>_<method>.mp4, hort_*, gt_overlay_*
├── make_rc_input.py           # adapter: Aria fisheye -> pinhole RGB + ray-cast GT depth
├── run_batch.py               # run a pipeline arm on the 6-clip bench, score, overlay
├── gt_pose_eval_hot3d.py      # score a run vs mocap GT (chamfer/centroid/rot_traj/shape)
├── leaderboard.py             # gate + render scores/LEADERBOARD.md
├── probe_clips.py             # select single-object interaction clips from HOT3D
├── make_rc_vs_gt_overlay.py   # GT-vs-estimate side-by-side video
├── gt_overlay_hot3d.py        # GT objects+hands overlay (sanity)
├── hort_hot3d_overlay.py      # HORT reprojection overlay
├── run_any6d_hot3d.py         # Any6D per-frame RGB-D pose (mesh-controlled)
└── combined_refine.py         # combined method: learned pose + flip-fix + jitter smooth
```

Run dirs: `render_and_compare/runs/hot3d_<clip>_<method>/stage8_eval/pseudo_gt.npz`
= `{obj_verts, obj_faces, obj_poses[T,4,4]}`, the input to `gt_pose_eval_hot3d.py`.
Frozen bench inputs: `/workspace/datasets/hot3d/rc_input_<num>_<clip>/`.

### Reproduce

```bash
PY=/workspace/miniconda3/envs/rc5090/bin/python
$PY gt_pose_eval_hot3d.py /workspace/datasets/hot3d/rc_input_002034_bottle_bbq \
    ../../render_and_compare/runs/hot3d_bottle_bbq_002034_icpjgr    # score vs GT
$PY leaderboard.py render                                           # -> scores/LEADERBOARD.md
$PY combined_refine.py <any6d_run_dir> <out_run_dir>               # combined method
```

Envs: `rc5090` (pipeline+eval), `forehoi5090` (ForeHOI/FoundationPose/Any6D),
`hort5090` (HORT). Blackwell revival recipes in `docs/T4_NOTES.md`.
