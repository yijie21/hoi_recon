# HOT3D object-recovery benchmark

Reconstruct a hand-held object's 6-DoF trajectory and shape from an egocentric video, and score
it against motion-capture-grade ground truth. This folder is the live benchmark harness: it runs
each method ("arm") on the same 6 HOT3D clips, scores every one against the mocap ground truth,
and renders the results into [`scores/LEADERBOARD.md`](scores/LEADERBOARD.md).

> See [`../../GLOSSARY.md`](../../GLOSSARY.md) for what every method code, metric, and env name
> means. This page uses plain names and links the glossary once, here.

The detailed campaign log lives in [`docs/`](docs/) — [`docs/REFLECTION.md`](docs/REFLECTION.md)
(full journey), [`docs/T4_RESULTS.md`](docs/T4_RESULTS.md) (learned-method bake-off numbers). This
page is the short summary.

## What we found, in order

1. **Switched benchmark: HOI4D → HOT3D.** HOI4D's own annotations were too noisy (±1.5 cm / ±20 px
   scatter) to tell methods apart. HOT3D has mocap-grade ground truth instead. It has no depth
   sensor, so `make_rc_input.py` rectifies the Aria fisheye video onto a pinhole camera and
   ray-casts ground-truth depth from the posed CAD + hand meshes — giving a calibrated RGB-D
   input, which turned out to matter a lot (see point 3).

2. **Built the registration pipeline (`icpjgr`), one gated step at a time** (each step accepted
   only if no clip got worse):
   - **Hand-aware segmentation** — the biggest win. The two worst clips failed because the
     object-mask prompt point could land on the occluding hand. Tracking and subtracting the hand
     mask fixed it: mug 60.7→7.0 mm, spatula 158.8→20.5 mm.
   - **Chroma attitude search** — depth alone can't tell a symmetric object's rotation apart, so
     rotation hypotheses are scored by texture alignment (only when the texture is informative
     enough). Fixed the bottle: 19.9→9.9 mm.
   - **Grasp-rigidity** — during fast in-hand rotation, treating the object as rigidly attached to
     the wrist gave a small further gain on the rotation tail.
   - Net result: worst-clip placement error 158.8→21.2 mm, mean 49.1→15.5 mm (3.2× better).
     Rotation error stayed about flat, floored by the symmetric puzzle-toy cube.

3. **Learned methods vs. the pipeline (the T4 bake-off).** Five cloned repos plus two external
   methods were revived on the RTX 5090 (Blackwell) box. The deciding factor was **whether a
   method uses the calibrated depth**:
   - Discard the depth → lose by 10–160× on placement: HORT (monocular, no metric scale, wrong
     object on 4/6 clips) and ForeHOI (its shape is good — 2.7–16 mm — but self-estimated depth
     puts that shape in the wrong place).
   - Consume the depth → beat `icpjgr` on accuracy: given the same mesh + depth + mask, **Any6D**
     (`any6dp`) wins placement on 5/6 clips (often 2–3× lower), and **FoundationPose** wins
     decisively where its tracker holds (rotation down to ~3°).

   The registration pipeline's real advantage turned out to be **robustness**: it never drifts
   and never flips a symmetric object 180°, unlike the learned per-frame estimators.

4. **Combined method: learned placement + pipeline robustness.** `combined_refine.py` takes a
   learned method's per-frame poses and applies two fixes from the pipeline's temporal layer:
   picking the symmetry-equivalent rotation closest to neighboring frames (fixes 180° flips), and
   smoothing translation jitter (position jitter p90 66→7 mm, at under 1 mm placement cost). On
   the bottle clip this beats `icpjgr` on every metric (placement 5.2 vs 9.9 mm, rotation p90 17°
   vs 73°).

## Headline numbers — placement error (mm, median), same mesh across methods

| clip | icpjgr | HORT | ForeHOI | FoundationPose | Any6D | combined |
|---|---|---|---|---|---|---|
| bottle_bbq | 9.9 | ✗ | 392 | 3.2 / 93 | 5.2 | **5.2** |
| mug_white | 7.0 | ~ | 1147 | **2.3** | 3.4 | 3.4 |
| vase | 17.7 | ✗ | 212 | 6.6 | 6.4 | 6.6 |
| spatula_red | 21.2 | ✗ | 178 | 11.2 | 9.6 | 9.6 |
| potato_masher | 18.8 | ✗ | 704 | 8.6 / 598 | 12.0 | 12.2 |
| puzzle_toy (cube) | 18.5 | ~ | 391 | 18.9 | 21.3 | 21.3 |

FoundationPose shows two numbers where they differ: `register_each` mode / default `track` mode
(track mode drifts on these clips). `✗` = wrong object or no metric scale; `~` = plausible shape
but unscaled. Full table with rotation numbers: [`scores/LEADERBOARD.md`](scores/LEADERBOARD.md).

## Lessons

- **Calibration is decisive.** Given calibrated RGB-D, the methods that use the depth win;
  monocular learned methods can get the shape right but not the placement.
- **Accuracy and robustness are different axes.** Learned per-frame estimators win accuracy;
  temporal smoothing wins robustness — the combined method keeps both.
- **Every comparison must be mesh-controlled.** Object mesh generation (SAM-3D) is
  nondeterministic, so every arm reuses the incumbent's early stages to isolate the method under
  test.
- **Render and look at the output.** Every convention trap and bug here was caught by eyeballing
  an overlay, not by a metric.
- **A single gate metric under-credits real wins** — always keep the raw per-clip numbers, not
  just the pass/fail verdict.

## What to build next

1. ✅ Done — put a learned pose core (Any6D) inside the pipeline, keeping the temporal layer (arm
   `any6dp`). Wins on placement; trades off against rotation. See [`docs/T5_NOTES.md`](docs/T5_NOTES.md).
2. ⛔ Tried and abandoned — recovering the learned core's rotation directly (several approaches,
   all failed). `any6dp` and `icpjgr` remained a placement-vs-rotation trade-off. See
   [`docs/T5_NOTES.md`](docs/T5_NOTES.md).
3. Hand-aware segmentation remains the highest-leverage fix for any method, upstream of this one.
4. Better SAM-3D texture fidelity, so texture-based rotation scoring works on more objects.

## Directory structure

```
compare/hot3d/
├── README.md                  # this summary
├── docs/                      # REFLECTION.md (full journey), T4_RESULTS.md, T4_NOTES.md, T5_NOTES.md
├── scores/                    # LEADERBOARD.md, BEST_ARM, batch_summary_<arm>.json
├── overlays/                  # curated best-per-clip rc_vs_gt_<clip>_<method>.mp4 (5 fpauto + icpjgr masher)
├── make_rc_input.py           # adapter: Aria fisheye -> pinhole RGB + ray-cast GT depth
├── run_batch.py               # run a pipeline arm on the 6-clip bench, score, overlay
├── gt_pose_eval_hot3d.py      # score a run vs mocap GT (chamfer/centroid/rot_traj/shape)
├── leaderboard.py             # gate + render scores/LEADERBOARD.md
├── probe_clips.py             # select single-object interaction clips from HOT3D
├── make_rc_vs_gt_overlay.py   # GT-vs-estimate side-by-side video
├── gt_overlay_hot3d.py        # GT objects+hands overlay (sanity check)
├── hort_hot3d_overlay.py      # HORT reprojection overlay
├── run_any6d_hot3d.py         # Any6D per-frame RGB-D pose (mesh-controlled)
└── combined_refine.py         # combined method: learned pose + flip-fix + jitter smooth
```

Run outputs: `render_and_compare/runs/hot3d_<clip>_<method>/stage8_eval/pseudo_gt.npz` =
`{obj_verts, obj_faces, obj_poses[T,4,4]}`, the input to `gt_pose_eval_hot3d.py`.
Frozen bench inputs: `/workspace/datasets/hot3d/rc_input_<num>_<clip>/`.

## Reproduce

```bash
PY=/workspace/miniconda3/envs/rc5090/bin/python
$PY gt_pose_eval_hot3d.py /workspace/datasets/hot3d/rc_input_002034_bottle_bbq \
    ../../render_and_compare/runs/hot3d_bottle_bbq_002034_icpjgr    # score vs GT
$PY leaderboard.py render                                           # -> scores/LEADERBOARD.md
$PY combined_refine.py <any6d_run_dir> <out_run_dir>               # combined method
```

Envs: `rc5090` (pipeline+eval), `forehoi5090` (ForeHOI/FoundationPose/Any6D), `hort5090` (HORT).
Blackwell revival recipes in [`docs/T4_NOTES.md`](docs/T4_NOTES.md).
