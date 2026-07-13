# T4 — learned-method bake-off vs the pipeline (icpjgr) on HOT3D

> **In plain terms.** This is the actual bake-off: the hand-built pipeline (`icpjgr`) compared
> against four learned pose-estimation methods (HORT, ForeHOI, Any6D, FoundationPose) on the same
> video clips and 3D meshes. Methods that ignore the depth-camera data (HORT, ForeHOI) lost
> badly. Methods that use it (Any6D, FoundationPose) actually beat the hand-built pipeline at
> placing the object correctly — but were less stable, sometimes drifting away or flipping the
> object's rotation. Conclusion: the best design combines a learned method's accuracy with the
> hand-built pipeline's stability, which is what the later `any6dp` and `fpauto` arms do. Method
> codes decoded in [GLOSSARY.md](../../../GLOSSARY.md).

Five learned methods were revived on Blackwell (sm_120) and compared against
our best pipeline arm **icpjgr** on the frozen 6-clip HOT3D bench, scored by
the same mocap-GT eval (`gt_pose_eval_hot3d.py`). All comparisons are
**mesh-controlled**: every method registers/places the *same* SAM-3D stage-3
mesh icpjgr used, so only the pose/placement method differs.

## The decisive split: does the method consume the calibrated RGB-D?

HOT3D (via our fisheye→pinhole rectification) gives calibrated intrinsics +
metric depth. Whether a method *uses* that signal cleanly predicts the result.

### Group A — discard the calibrated depth → lose to icpjgr by 10–160×

| clip (chamfer mm) | icpjgr | HORT | ForeHOI |
|---|---|---|---|
| bottle_bbq | 9.9 | ✗ wrong object | 392 |
| mug_white | 7.0 | (unscaled) | 1147 |
| vase | 17.7 | ✗ wrong object | 212 |
| spatula_red | 21.2 | ✗ wrong object | 178 |
| potato_masher | 18.8 | ✗ wrong object | 704 |
| puzzle_toy | 18.5 | (unscaled) | 391 |

- **HORT** (mono single-image): no metric scale, no temporal consistency,
  LangSAM grabbed the wrong object on 4/6.
- **ForeHOI** (feed-forward video): *canonical shape* is excellent (shape-ICP
  bottle 2.7, masher 3.3, vase 7.4, spatula 16.3 mm) but its wild path
  self-estimates depth with DepthAnything3 (off ~2.5× on egocentric HOT3D), so
  a well-shaped object lands at the wrong 3D location. Shape good, placement
  bad.

### Group B — consume the calibrated RGB-D → per-frame pose STRONGER than icpjgr

| clip | icpjgr chamfer | **Any6D** | **FoundationPose (track)** | FP (register-each) |
|---|---|---|---|---|
| bottle_bbq | 9.9 | **5.2** | 93 (drift) | **3.2** |
| mug_white | 7.0 | **3.4** | **2.3** (rot 2.7°!) | — |
| vase | 17.7 | **6.4** | **6.6** (rot 3.0°!) | — |
| spatula_red | 21.2 | **9.6** | **11.2** | 12.8 |
| potato_masher | 18.8 | **12.0** | 598 (drift) | **8.6** |
| puzzle_toy (cube) | 18.5 | 21.3 | 18.9 (tie) | — |

Given the *same* mesh + depth + mask, both learned RGB-D render-and-compare
estimators **beat icpjgr's registration on placement** — Any6D wins chamfer on
5/6; FoundationPose wins decisively where its tracker holds (vase, mug: 3–8×
lower chamfer *and* rotation error down to ~3°) and, in drift-free
`register_each` mode, beats icpjgr on every clip it was run (bottle 9.9→3.2).

## What icpjgr still wins: robustness / temporal consistency

The learned per-frame estimators are more accurate but less *stable*:

- **Any6D** registers each frame independently → symmetry-flip outliers:
  rot_traj p90 **153–173°** on masher/cube/spatula (a minority of frames snap
  to a 180°-equivalent pose, uncorrected). Median rotation is fine
  (bottle 4.5°, vase 8.5°) but the tail is unusable as-is.
- **FoundationPose** default `track` mode has no drift recovery → catastrophic
  blow-ups on 2/6 (masher 598 mm, bottle 93 mm). `register_each` fixes it but
  costs ~7–8× compute.
- **icpjgr** never fails catastrophically: worst-clip chamfer 21 mm, rot_traj
  p90 42–127° — its single-trajectory temporal optimization bounds every clip.
- The **cube** is hard for everyone (24-fold symmetry + a thin-slab SAM-3D
  mesh, extent [34,9,32] mm): Any6D 21.3, FP 18.9, icpjgr 18.5 — all ~tied,
  all with terrible rotation, because a flat symmetric slab has almost no
  observable per-frame orientation.

## Conclusion (revised — the important finding)

On a benchmark **with** calibrated RGB-D, a learned per-frame pose estimator
(Any6D, FoundationPose) that consumes that depth is **more accurate than our
hand-built depth+silhouette registration** — often 2–3× lower chamfer, and for
FoundationPose far lower rotation error where it holds. The methods that lost
badly (HORT, ForeHOI) did so only because they *discard* the metric depth, not
because learned recon is weak.

What our pipeline contributes is **robustness through temporal optimization**:
it never drifts (unlike FP `track`) and never symmetry-flips (unlike Any6D
per-frame), because it solves one smooth trajectory constrained across all
frames. The learned estimators solve each frame in isolation and pay for it in
the tail.

**The best system combines them**: a learned per-frame RGB-D pose estimator
(Any6D / FoundationPose `register`) for accuracy, wrapped in icpjgr's
temporal-trajectory + symmetry-resolution layer for stability. That is the
concrete next build this bake-off points to — swap stage-4's registration core
for a learned per-frame estimator, keep the joint temporal/grasp/attitude
optimization on top.

## Environments (all revived by cloning `sam3d5090`, the box's working Blackwell stack)

- **hort5090**: + HORT deps + rebuilt pointnet2_ops; `CUDA_HOME=$CONDA_PREFIX`.
- **forehoi5090**: + ForeHOI deps + nvdiffrast (env's own cu128 nvcc); also runs
  FoundationPose (`fp_track.py`; `mycpp` prebuilt for sm_120, `mycuda` not
  needed) and Any6D (`run_any6d_hot3d.py`; `mycpp` cpu build + 258 MB weights).
- "One env for all" remains infeasible for HOLD/EasyHOI/do-as-i-do (cu118/cu121
  + 5–8 custom CUDA extensions with no sm_120 wheels) — see T4_NOTES.md.

Per-method reports: `.superpowers/sdd/run-{hort,forehoi,foundationpose,any6d}.md`.
Summaries: `compare/hot3d/batch_summary_{icpjgr,forehoi,any6d,fp}.json`.
