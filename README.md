# hoi_recon — Hand-Object Interaction Reconstruction

Recover a **4D hand-object interaction** from an egocentric/monocular video: the
object's 3D shape, where it sits and how it turns every frame, plus the hand's pose
every frame. This branch contains exactly one thing: **the best method**, runnable from a
fresh clone, plus the HOT3D benchmark harness that scores it against motion-capture
ground truth. (The research history — alternative methods, bake-offs, ablations — lives
on the `dev` branch.)

> **New here? Read [`GLOSSARY.md`](GLOSSARY.md) first** — it decodes every method name,
> metric, and environment in plain language.

## The method

The reconstruction combines the **best object track** with the **best hand track**:

1. **Registration pipeline (`icpjgr`)** — the 9-stage `render_and_compare` pipeline:
   segmentation (hand-aware), hand mesh, SAM-3D object mesh, then registration of the mesh
   onto depth + silhouette with a grasp-closure step. Produces the object mesh, the initial
   hand, and a rotation-robust object track.
2. **Learned object core (`fpauto`, FoundationPose)** — re-estimates the object's 6-DoF
   track from the calibrated RGB-D using the pipeline's mesh, with a drift-gated mode
   selector. Best placement (**~8 mm** average) and rotation; ships on 5 of 6 clips. The
   exception is the potato masher (a spinning, near-symmetric object), which keeps the
   registration pipeline's rotation-robust track.
3. **Hand-reprojection optimizer** — slides the MANO hand until it lines up with the
   observed hand pixels: **2–4 px** image accuracy (from 5–57 px before), on every clip.

**Numbers:** [`compare/hot3d/RESULTS.md`](compare/hot3d/RESULTS.md).
**Deliverable videos:** `compare/hot3d/overlays/hoi_best_<clip>.mp4` — three panels,
`[ original | object | object + hand ]`, both meshes backprojected onto the video.

## How to run it

One-time setup (conda envs, checkpoints, MANO, HOT3D data):
[`render_and_compare/REPRODUCE.md`](render_and_compare/REPRODUCE.md).

Then, from `compare/hot3d/`, for the bottle clip (envs: `rc5090` = pipeline,
`forehoi5090` = FoundationPose, `sam3d5090` = object mesh + hand optimizer):

```bash
RC5=/workspace/miniconda3/envs/rc5090/bin/python
FH5=/workspace/miniconda3/envs/forehoi5090/bin/python
RC=/workspace/datasets/hot3d/rc_input_002034_bottle_bbq
RUN=../../render_and_compare/runs/hot3d_bottle_bbq_002034_icpjgr
FP=../../render_and_compare/runs/hot3d_bottle_bbq_002034_fpauto
CFG=/workspace/code/hoi_recon/render_and_compare/configs/real_forehoi_icp_joint_grasp.yaml

$RC5 run_batch.py selection.json --arm icpjgr --config $CFG          # 1. pipeline: hand + object mesh
$FH5 run_fp_hot3d.py $RC $RUN $FP --mode auto                        # 2. best object (FoundationPose)
$RC5 run_hand_reproj.py $RUN $RC                                     # 3. best hand
$RC5 make_hoi_best_overlay.py $FP $RUN overlays/hoi_best_bottle_bbq_002034.mp4   # 4. combined overlay
$RC5 gt_pose_eval_hot3d.py $RC $FP                                   # 5a. score the object
$RC5 gt_hand_eval_hot3d.py /workspace/datasets/hot3d/clips/clip-002034 $RC \
     before=$RUN after=$RUN/hand_reproj_opt/out.npz                  # 5b. score the hand
```

Pass the config as an **absolute path** — `run_batch` runs the pipeline from the
`render_and_compare/` directory.

The pipeline also runs on your own video without HOT3D (no GT scoring):
see [`render_and_compare/README.md`](render_and_compare/README.md)
(`python -m hoi_recon.cli --video your.mp4 --out runs/yours --real ...`).

## Why this combination

The registration pipeline never fails catastrophically but plateaus at ~15 mm placement;
the learned RGB-D core is 2–3× more accurate but can drift or flip a symmetric object
180°. The shipped method takes the learned core's accuracy where it holds and falls back
to the registration pipeline where it doesn't. The one hard wall is **rotation of
symmetric objects**: their orientation is genuinely ambiguous from depth, and on hand-held
objects the hand hides the one distinguishing feature — so the worst case is bounded with
the registration pipeline rather than "fixed".

## Repository layout

- **[`render_and_compare/`](render_and_compare/)** — the pipeline (installable package
  `hoi_recon`): 9 cached stages from depth → object mesh → placement → grasp. Runs in env
  `rc5090`. Docs: [`README.md`](render_and_compare/README.md) (usage),
  [`DESIGN.md`](render_and_compare/DESIGN.md) (architecture),
  [`REPRODUCE.md`](render_and_compare/REPRODUCE.md) (setup from a fresh machine).
- **[`compare/hot3d/`](compare/hot3d/)** — the HOT3D benchmark harness: clip adapter,
  batch driver, FoundationPose arm, hand optimizer driver, GT scorers, results. Also hosts
  the separate clean-training-clips pipeline ([`HOI_CLIPS.md`](compare/hot3d/HOI_CLIPS.md)).

## Environments

Three conda environments on an RTX 5090 (Blackwell, `sm_120`) box — `rc5090`,
`sam3d5090`, `forehoi5090`; what each runs is in
[`GLOSSARY.md`](GLOSSARY.md#conda-environments-the-rtx-5090--blackwell-box), and the
build recipes are in [`render_and_compare/REPRODUCE.md`](render_and_compare/REPRODUCE.md).
Pre-Blackwell environments (cu118/cu121) have no `sm_120` kernels and will not run on
this hardware.
