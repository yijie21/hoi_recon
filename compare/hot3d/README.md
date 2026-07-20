# HOT3D benchmark harness

Run the reconstruction method on 6 fixed HOT3D clips and score it against
motion-capture-grade ground truth. HOT3D has no depth sensor, so
`make_rc_input.py` rectifies the Aria fisheye video onto a pinhole camera and
ray-casts ground-truth depth from the posed CAD + hand meshes — giving the
pipeline a calibrated RGB-D input.

> See [`../../GLOSSARY.md`](../../GLOSSARY.md) for what every method name, metric, and env
> name means. Numbers: [`RESULTS.md`](RESULTS.md).

> **Building the clean single-object HOI training clips** (separate from the benchmark) is
> documented in [`HOI_CLIPS.md`](HOI_CLIPS.md) — one command regenerates the segment dataset +
> its fast PyTorch dataloader.

## Running the method

The shipped result = registration pipeline (hand + object mesh) → FoundationPose object
track (`fpauto`, 5 of 6 clips; the near-symmetric potato masher keeps the registration
pipeline's rotation) → hand-reprojection optimizer. The copy-paste recipe for one clip is
in the top-level [`README.md`](../../README.md); the batch driver:

```bash
RC5=/workspace/miniconda3/envs/rc5090/bin/python
$RC5 run_batch.py selection.json --arm icpjgr \
    --config /abs/path/to/render_and_compare/configs/real_forehoi_icp_joint_grasp.yaml
```

`run_batch.py` adapts each clip (cached), runs the pipeline, and scores the run.
`selection.json` lists the 6 bench clips. Pass the config as an **absolute path** — the
pipeline runs from `render_and_compare/`.

## Directory structure

```
compare/hot3d/
├── README.md                  # this page
├── RESULTS.md                 # the shipped per-clip numbers
├── HOI_CLIPS.md               # clean single-object training-clip pipeline (separate)
├── make_rc_input.py           # adapter: Aria fisheye -> pinhole RGB + ray-cast GT depth
├── run_batch.py               # run the pipeline on the 6-clip bench, score each run
├── run_fp_hot3d.py            # FoundationPose object track on top of a pipeline run (env forehoi5090)
├── run_hand_reproj.py         # hand-reprojection optimizer (env sam3d5090 subprocess)
├── gt_pose_eval_hot3d.py      # score the object vs mocap GT (chamfer/centroid/rot_traj)
├── gt_hand_eval_hot3d.py      # score the hand vs GT (reprojection px, 3D)
├── make_hoi_best_overlay.py   # deliverable overlay: [original | object | object+hand]
├── selection.json             # the 6 bench clips
├── overlays/                  # hoi_best_<clip>.mp4 deliverable videos
├── tests/                     # harness unit tests (no GPU needed)
└── gen_clean_clips.py + materialize_clips.py + precompute_segments.py
    + build_hoi_dataset.py + hoi_dataset.py + render_clean_overlay.py   # HOI_CLIPS.md pipeline
```

Run outputs: `render_and_compare/runs/hot3d_<clip>_<arm>/stage8_eval/pseudo_gt.npz` =
`{obj_verts, obj_faces, obj_poses[T,4,4]}`, the input to `gt_pose_eval_hot3d.py`.
Frozen bench inputs: `/workspace/datasets/hot3d/rc_input_<num>_<clip>/`.

## Practices that keep the numbers honest

- **Every comparison is mesh-controlled.** Object mesh generation (SAM-3D) is GPU-nondeterministic,
  so any rerun must reuse the incumbent run's early stages (`run_batch.py` auto-promotes
  identical-mask stage2/stage3 dirs) to isolate the change under test.
- **Render and look at the output.** Every convention trap and bug in this harness's history was
  caught by eyeballing an overlay, not by a metric.
- **Keep the raw per-clip numbers**, not just an aggregate verdict.

The full research history — alternative methods, bake-offs, ablations, campaign notes — lives on
the `dev` branch.
