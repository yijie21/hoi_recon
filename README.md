# hoi_recon — Hand-Object Interaction Reconstruction

Recover a **4D hand-object interaction** from an egocentric/monocular video: the
object's 3D shape, where it sits and how it turns every frame, plus the hand's pose
every frame. This repo holds our reconstruction **pipeline**, several third-party
methods revived for **comparison**, and the **benchmark study** that scores them all
against motion-capture ground truth on the HOT3D dataset.

> **New here? Read [`GLOSSARY.md`](GLOSSARY.md) first** — it decodes every method name
> (`icpjgr`, `fpauto`, …), metric, and environment in plain language. This README uses
> those names sparingly and always with their plain meaning.

## The best result

The best reconstruction combines the **best object track** with the **best hand track**:

- **Object → the learned core (`fpauto`, FoundationPose).** Best object placement — average
  3D error **~8 mm** across the clips, beating the earlier learned core on both placement and
  rotation. Exception: the potato masher (a spinning symmetric object) keeps the **registration
  pipeline (`icpjgr`)**, which handles that rotation better.
- **Hand → the hand-reprojection optimizer.** Slides the MANO hand model until it lines up with
  the observed hand: the hand lands at **2–4 px** in the image (from 5–57 px before), on every clip.

**Full numbers:** [`compare/hot3d/scores/LEADERBOARD.md`](compare/hot3d/scores/LEADERBOARD.md).
**Deliverable videos:** `compare/hot3d/overlays/hoi_best_<clip>.mp4` — three panels,
`[ original | object | object + hand ]`, both meshes backprojected onto the video.

### How to reproduce it

Run on one HOT3D clip (envs: `rc5090` = pipeline, `forehoi5090` = FoundationPose,
`sam3d5090` = object mesh + hand optimizer). From `compare/hot3d/`, for the bottle clip:

```bash
RC5=/workspace/miniconda3/envs/rc5090/bin/python
FH5=/workspace/miniconda3/envs/forehoi5090/bin/python
RC=/workspace/datasets/hot3d/rc_input_002034_bottle_bbq
RUN=../../render_and_compare/runs/hot3d_bottle_bbq_002034_icpjgr
FP=../../render_and_compare/runs/hot3d_bottle_bbq_002034_fpauto
CFG=/workspace/code/hoi_recon/render_and_compare/configs/real_forehoi_icp_joint_grasp.yaml

$RC5 run_batch.py selection_fixed.json --arm icpjgr --config $CFG   # 1. pipeline: hand + object mesh
$FH5 run_fp_hot3d.py $RC $RUN $FP --mode auto                        # 2. best object (FoundationPose)
$RC5 run_hand_reproj.py $RUN $RC                                     # 3. best hand
$RC5 make_hoi_best_overlay.py $FP $RUN overlays/hoi_best_bottle_bbq_002034.mp4   # 4. combined overlay
$RC5 gt_pose_eval_hot3d.py $RC $FP                                   # 5a. score the object
$RC5 gt_hand_eval_hot3d.py /workspace/datasets/hot3d/clips/clip-002034 $RC \
     before=$RUN after=$RUN/hand_reproj_opt/out.npz                  # 5b. score the hand
```

Note: pass the config as an **absolute path** — `run_batch` runs the pipeline from the
`render_and_compare/` directory. Full setup from a fresh machine (envs, weights, MANO,
data) is in [`render_and_compare/REPRODUCE.md`](render_and_compare/REPRODUCE.md).

## What we learned (short version)

Our registration pipeline cut object placement error **3.2×** versus a naive baseline and
never fails catastrophically. Swapping in a **learned RGB-D pose core** (FoundationPose)
improved placement further. The one hard wall is **rotation of symmetric objects**: their
orientation is genuinely ambiguous from depth, and on hand-held objects the hand hides the
one distinguishing feature — several attempts to fix this all failed, so we bound the
worst case with the registration pipeline instead. Full story: the campaign notes under
[`compare/hot3d/docs/`](compare/hot3d/docs/).

## Where to read next

| If you want… | Read |
|---|---|
| Every name/metric decoded | **[`GLOSSARY.md`](GLOSSARY.md)** |
| The current strategy + how each pipeline stage works | **[`BEST_STRATEGY.md`](BEST_STRATEGY.md)** |
| The head-to-head numbers | [`compare/hot3d/scores/LEADERBOARD.md`](compare/hot3d/scores/LEADERBOARD.md) |
| The full experiment log (what worked, what didn't) | [`compare/hot3d/docs/`](compare/hot3d/docs/) — `T5_NOTES` (learned core), `T6_NOTES` (FoundationPose), `REFLECTION` |
| To set up and run the pipeline | [`render_and_compare/README.md`](render_and_compare/README.md) + [`render_and_compare/REPRODUCE.md`](render_and_compare/REPRODUCE.md) |

## Repository layout

Three self-contained parts:

- **[`render_and_compare/`](render_and_compare/)** — the main pipeline (installable package
  `hoi_recon`): 9 cached stages from depth → object mesh → placement → grasp. Runs in env `rc5090`.
- **[`compare/`](compare/)** — the benchmark study. `compare/hot3d/` is the live harness that
  adapts HOT3D clips, runs each method, and scores it against mocap ground truth. Third-party
  comparison methods (Any6D, FoundationPose, ForeHOI, HORT) are cloned as sibling folders.
- **[`egoaero/`](egoaero/)** — a separate egocentric HOI package (asset-free reconstruction).

Each method is a self-contained peer: its own environment, dependencies, and gitignored
runtime files (`runs/`, `checkpoints/`, `third_party/`). It reads the same input (video +
optional depth/intrinsics) and writes the same output
(`stage8_eval/pseudo_gt.npz` = object mesh + per-frame poses), scored by
`compare/hot3d/gt_pose_eval_hot3d.py`.

## Environments

Four conda environments on the RTX 5090 (Blackwell) box — see [`GLOSSARY.md`](GLOSSARY.md#conda-environments-the-rtx-5090--blackwell-box)
for what each runs. Older pre-Blackwell environments have no compatible GPU kernels and are
not usable; setup recipes are in the recalled `hoi-recon-*` notes and
[`compare/hot3d/docs/T4_NOTES.md`](compare/hot3d/docs/T4_NOTES.md).
