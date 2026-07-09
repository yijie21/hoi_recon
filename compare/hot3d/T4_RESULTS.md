# T4 — learned-method bake-off vs the pipeline (icpjgr) on HOT3D

Both feasible learned methods were revived on Blackwell (sm_120) and run on
the frozen 6-clip HOT3D bench against our best arm **icpjgr**. Scored by the
same mocap-GT eval (`gt_pose_eval_hot3d.py`).

## Environments revived (both by cloning `sam3d5090`, the box's working Blackwell stack)

- **hort5090** = clone sam3d5090 + HORT deps + rebuilt `pointnet2_ops`;
  `CUDA_HOME=$CONDA_PREFIX` (the clone ships a full cu128 toolkit). Weights
  local in `hort/weights/`.
- **forehoi5090** = clone sam3d5090 + ForeHOI deps + nvdiffrast built against
  the env's own 12.8 nvcc; FoundationPose `mycuda` not needed (off the
  tracking path). Weights local at `/workspace/code/ForeHOI/` (17 GB).

("One env for all" remains infeasible — HOLD/EasyHOI/do-as-i-do need
cu118/cu121 + 5–8 custom CUDA extensions with no sm_120 wheels; see T4_NOTES.)

## Object placement / pose — chamfer median (mm) vs mocap GT

| clip | **icpjgr** | HORT | ForeHOI |
|---|---|---|---|
| vase | **17.7** | ✗ wrong object | 212 |
| potato_masher | **18.8** | ✗ wrong object | 704 |
| bottle_bbq | **9.9** | ✗ wrong object | 392 |
| puzzle_toy | **18.5** | (plausible, unscaled) | 391 |
| mug_white | **7.0** | (plausible, unscaled) | 1147 |
| spatula_red | **21.2** | ✗ wrong object | 178 |

**icpjgr wins decisively on every clip (10–160×).**

## Why — and the fair nuance (shape vs placement)

The gap is not that the learned reconstructions are bad — it's that they
**discard the calibrated depth + intrinsics the benchmark provides**, which is
exactly the signal icpjgr exploits.

- **HORT** (mono single-image): no metric scale at all (a WiLoR crop-camera
  heuristic, ~7–9× off), no temporal consistency (independent per frame), and
  its LangSAM text prompt grabbed the *wrong* object on 4/6 clips. Overlays
  `compare/hot3d/hort_*.mp4`.
- **ForeHOI** (feed-forward video recon): its *canonical shape* is genuinely
  competitive — alignment-invariant shape-ICP is **bottle 2.7 mm, masher
  3.3 mm, vase 7.4 mm, spatula 16.3 mm** — but its wild path self-estimates
  depth with DepthAnything3, which is badly off on egocentric HOT3D (vase
  frame-0: GT 1.03 m vs DA3 0.42 m), so a well-shaped object lands at the
  wrong 3D location → huge chamfer. Overlays
  `compare/hot3d/rc_vs_gt_hot3d_{vase,potato_masher}_forehoi.mp4`.

| clip | ForeHOI canonical **shape** ICP (mm) | placement chamfer (mm) |
|---|---|---|
| bottle_bbq | 2.7 | 392 |
| potato_masher | 3.3 | 704 |
| vase | 7.4 | 212 |
| spatula_red | 16.3 | 178 |
| puzzle_toy | 48.2 (symmetric) | 391 |
| mug_white | 222.9 (wrong object) | 1147 |

## Conclusion

On a benchmark **with** calibrated RGB-D (HOT3D via our fisheye→pinhole
rectification + rendered GT depth), the optimization-based pipeline that
consumes that calibration (icpjgr) decisively beats the learned feed-forward
methods at 3D placement/pose. The learned methods' strength is
category-agnostic *shape* from a single view (ForeHOI's 2.7–16 mm), and they
would be more competitive in an **uncalibrated monocular wild** setting where
no metric depth is available — precisely the regime they were designed for.
This validates the pipeline's architecture for this task: use a learned shape
prior (SAM-3D) but ground pose in the calibrated depth via registration,
rather than trusting monocular depth end-to-end.
