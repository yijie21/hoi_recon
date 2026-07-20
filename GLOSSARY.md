# Glossary — what the names and numbers mean

A plain-language decoder for the abbreviations used across this repo.

## The task in one line
Given an egocentric video of a hand using an object, recover the **4D hand-object
interaction**: the object's 3D shape + where it is and how it turns every frame, plus
the hand's pose every frame. We test on **HOT3D**, a dataset with motion-capture
ground truth, so every result can be scored against the truth.

## The method's parts

| Code | Plain name | What it is |
|---|---|---|
| **`icpjgr`** | Registration pipeline | The hand-built pipeline arm. Fits the object mesh onto the depth map + object silhouette, then closes the grasp by nudging the hand. Rotation-robust — never fails catastrophically. (`icp` = the fitting algorithm, `j` = joint depth+silhouette, `gr` = grasp-rigidity step.) Provides the mesh + hand + fallback object track. |
| **`fpauto`** | Learned object core (FoundationPose) | A learned RGB-D pose estimator (NVIDIA's FoundationPose) run on the pipeline's object mesh, with a drift-guard that picks its best per-clip mode. **Best object placement + rotation** — the object track shipped on 5 of 6 clips. (`fp` = FoundationPose, `auto` = the automatic drift-gated selector.) |

**The hand** is reconstructed separately by the **hand-reprojection optimizer**: it takes
the pipeline's initial hand and slides the MANO hand model until it lines up with the hand
pixels in the image. Used with both object tracks.

## Metrics (all: lower is better)
| Name | Plain meaning |
|---|---|
| **chamfer (mm)** / "placement" | Average 3D distance between the reconstructed object and the true object, both posed in the scene. Captures both position and shape. Under ~5 mm is tight. |
| **centroid (cm)** | Error in the object's center position only. |
| **rot_traj (deg), med / p90** | How well the object's *frame-to-frame turning* matches the truth — median and 90th-percentile. This is the fair rotation metric for symmetric objects. |
| **rot_abs (deg)** | Single-frame absolute orientation error. Deliberately high for symmetric objects (a bottle rotated about its axis looks identical), so we rely on `rot_traj` instead. |
| **canonical ICP (mm)** | Pure shape error (best-fit alignment, pose removed) — how good the object *mesh* is, separate from where it was placed. |
| **hand reprojection (px)** | How far the reconstructed hand lands from the real hand in the image. 2–4 px is pixel-accurate. |
| **hand chamfer (mm)** | 3D distance between the reconstructed hand and the true hand mesh. |

## Pipeline stages (`render_and_compare`)
The pipeline runs 9 cached steps, `stage0`…`stage8`:
`0` depth + camera · `1` detect & segment hand/object · `2` hand mesh · `3` object 3D
shape · `4` place & rotate the object · `5` coarse fit · `6` rectify · `7` grasp
optimization · `8` evaluate. Each caches its output and is skipped on re-run.

## Model components (off-the-shelf building blocks)
| Name | Job |
|---|---|
| **MoGe** | Predicts metric depth + camera intrinsics from one image (stage 0). |
| **SAM 2** | Segments the object and hands in every frame (stage 1). |
| **WiLoR** | Detects hand bounding boxes (stage 1). |
| **HaMeR** | Reconstructs the MANO hand mesh (stage 2). |
| **SAM-3D** | Generates the object's textured 3D mesh from the masked image (stage 3). |
| **MANO** | The standard parametric hand model (license-gated). |
| **FoundationPose** | Learned RGB-D object pose estimator — the core of `fpauto`. |

## Dataset
**HOT3D** — egocentric hand-object clips with mocap-grade ground truth (the benchmark).
6 fixed clips: bottle, mug, vase, potato masher, spatula, puzzle toy.

## Conda environments (the RTX 5090 / Blackwell box)
| Env | Runs |
|---|---|
| **`rc5090`** | The main pipeline + scoring + overlays. |
| **`sam3d5090`** | SAM-3D object mesh generation + the hand optimizer. |
| **`forehoi5090`** | FoundationPose (the learned object core). |

## File landmarks
- **`rc_input_<id>_<object>/`** — a HOT3D clip converted into pipeline input (video + depth + intrinsics).
- **`stage8_eval/pseudo_gt.npz`** — the pipeline's final output: object mesh + per-frame object poses.
- **`hoi_best_<clip>.mp4`** — the deliverable overlay: `[ original video | object | object + hand ]`.
