# HOT3D results — the shipped 4D hand-object reconstruction

*Measured 2026-07-13 on 6 HOT3D clips with mocap-grade ground truth. Lower is better on
every metric.* Method names are decoded in [`GLOSSARY.md`](../../GLOSSARY.md).

## What the numbers mean
- **Placement (mm)** — average 3D gap between the reconstructed object and the true object,
  both placed in the scene. Under ~5 mm is a tight fit.
- **Rotation (deg)** — how well the object's turning matches truth, as median / 90th-percentile
  frame error. Large values mean the orientation is genuinely ambiguous (round/symmetric objects).
- **Hand fit (px)** — how far the reconstructed hand lands from the real hand in the image.
  2–4 px is pixel-accurate.

## The shipped result (best object + best hand)

| clip | object method | placement (mm) | rotation med/p90 (deg) | hand fit (px) |
|---|---|---|---|---|
| bottle_bbq | learned core (`fpauto`) | 2.9 | 53.5/161.9 | 2.3 |
| mug_white | learned core (`fpauto`) | 4.1 | 12.2/33.5 | 3.8 |
| vase | learned core (`fpauto`) | 5.4 | 6.4/58.6 | 2.7 |
| spatula_red | learned core (`fpauto`) | 12.1 | 5.4/10.4 | 2.6 |
| puzzle_toy | learned core (`fpauto`) | 15.3 | 21.1/93.7 | 3.2 |
| potato_masher | registration pipeline (`icpjgr`) | 22.0 | 62.4/81.8 | 1.9 |

The potato masher is a spinning, near-symmetric object: the learned core's per-frame rotation
drifts on it, so that one clip ships the rotation-robust registration pipeline instead.

## The hand optimizer's effect (image reprojection, px)

| clip | before | after |
|---|---|---|
| bottle_bbq | 56.6 | 2.3 |
| mug_white | 23.3 | 3.8 |
| vase | 35.4 | 2.7 |
| spatula_red | 20.9 | 2.6 |
| puzzle_toy | 5.4 | 3.2 |
| potato_masher | 8.5 | 1.9 |

**Deliverable videos:** [`overlays/hoi_best_<clip>.mp4`](overlays/) — three panels,
`[ original | object | object + hand ]`, both meshes backprojected onto the video.

## Regenerating a score

```bash
PY=/workspace/miniconda3/envs/rc5090/bin/python
$PY gt_pose_eval_hot3d.py <rc_input_dir> <run_dir>        # object vs mocap GT
$PY gt_hand_eval_hot3d.py <clip_dir> <rc_input_dir> \
    before=<run_dir> after=<run_dir>/hand_reproj_opt/out.npz   # hand vs GT
```

The full multi-method comparison (earlier arms, ablations, campaign notes) lives on the
`dev` branch.
