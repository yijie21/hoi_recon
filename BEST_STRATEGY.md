# Strategy — how we reconstruct hand-object interactions

*What we're doing, the method we ship today, what we learned, and what's left.
Every method name and metric is decoded in [`GLOSSARY.md`](GLOSSARY.md) — read that
first. The full experiment log is in [`compare/hot3d/docs/`](compare/hot3d/docs/);
the numbers in [`compare/hot3d/scores/LEADERBOARD.md`](compare/hot3d/scores/LEADERBOARD.md).*

## The goal

From an egocentric video of a hand using an object, recover the object's 3D shape +
where it sits and how it turns every frame, plus the hand. We test on **HOT3D**, which
has motion-capture ground truth, so every result is scored against the truth.

One property of HOT3D turned out to decide the whole project: it gives us **calibrated
depth**. (HOT3D has no depth sensor, so we ray-cast depth from the ground-truth object
and hand meshes.) Accurate depth is the single biggest lever in the pipeline.

## The method we ship

The best reconstruction combines the **best object track** with the **best hand track**:

- **Object → the learned core** (`fpauto`, built on FoundationPose). A learned RGB-D
  pose estimator run on our object mesh, with a drift-guard that picks its best per-clip
  mode. Best placement (~8 mm average error) and best rotation. **Exception:** the potato
  masher (a spinning symmetric object) keeps the **registration pipeline** (`icpjgr`),
  which bounds its rotation better.
- **Hand → the hand optimizer.** Slides the MANO hand model until it lines up with the
  observed hand pixels — the hand lands at **2–4 px** in the image (from 5–57 px before).

**The key insight that reframed the project:** on calibrated depth, a *learned* per-frame
pose estimator that consumes the depth beats our hand-built geometric fitting on accuracy.
Our pipeline's real contribution is **temporal robustness** — it never drifts or flips.
The winning recipe is the combination: a learned pose core, kept honest by temporal logic.

## How the pipeline works (9 cached stages)

The pipeline (`render_and_compare/`) runs 9 stages; each caches its output and is skipped
on re-run. Full detail in [`render_and_compare/DESIGN.md`](render_and_compare/DESIGN.md);
the highlights:

- **Stage 0 — depth + camera.** Rectify the fisheye video to a normal camera and get
  metric depth. On HOT3D the adapter (`compare/hot3d/make_rc_input.py`) ray-casts the
  ground-truth depth. *Use calibrated depth wherever it exists — never per-frame monocular
  depth, which "breathes" frame to frame.*
- **Stage 1 — hand-aware segmentation (the single biggest fix).** Track both hands and the
  object as separate objects, and reject object masks that leak onto a hand. Both of the
  original catastrophic failures were stage-1 mask errors (a mug mask that swallowed the
  forearm → 25 cm blob; a spatula mask that leaked onto the table → 80 cm mesh). Fixing
  this: mug 60.7 → 7.0 mm, spatula 158.8 → 20.5 mm. Code: `hoi_recon/mask_qa.py`.
- **Stage 2 — hand mesh.** HaMeR produces the per-frame MANO hand.
- **Stage 3 — object 3D shape.** SAM-3D generates one textured mesh (in env `sam3d5090`),
  reused for every frame. The shape is already good enough — don't invest in refining it.
- **Stage 4 — object placement + rotation (the core).** Either the **registration pipeline**
  (fit the mesh to depth + silhouette, `object_icp.py`) or the **learned core** (FoundationPose
  / Any6D, `run_fp_hot3d.py` / `object_any6d.py`).
- **Stages 5–7 — grasp closure.** The grasp optimizer trusts the object track and moves the
  **hand** to close the grasp, rather than dragging the object off its track.
- **Stage 8 — evaluation.** Scores the object (placement + rotation) against ground truth.

*Discipline: every comparison is **mesh-controlled** — methods reuse the same object mesh, so
SAM-3D's randomness can't masquerade as a real difference between methods.*

## What we learned

**Placement vs. rotation is a genuine trade-off (a hard wall).** A learned core wins on
placement decisively. But **rotation of symmetric objects cannot be fixed** — their
orientation is genuinely ambiguous from depth, and on a hand-held object the one
distinguishing feature is exactly what the hand hides. We tried four ways to fix rotation
and **all failed**:

1. Depth-anchored basin selection — redundant with the depth-consuming core.
2. A grasp-rigidity prior (use the wrist to constrain rotation) — hurt.
3. Surgical symmetry-flip fixing — neutral.
4. Baking the video texture onto the mesh — smeared and circular.

**Do not re-attempt the rotation-prior family** (temporal / attitude / texture priors) —
it is exhausted. `fpauto` later beat the earlier learned core on *both* axes not by adding
a corrective prior, but by being a better estimator (a uniform-scale mesh + a flip-free
tracker). The lesson: fix rotation with a better *estimator*, not a corrective prior.

**Other durable lessons:** calibrated depth is a moat; accuracy and robustness are
different axes (learned pose wins one, temporal optimization wins the other); mesh-control
every comparison; and **render and eyeball everything** — visual inspection caught every
convention trap and load-bearing bug that no metric did.

## What's left (highest value first)

1. **Wire `fpauto` into the pipeline as a one-command arm** (it currently runs via a
   standalone driver). See the follow-ups in [`compare/hot3d/docs/T6_NOTES.md`](compare/hot3d/docs/T6_NOTES.md).
2. **Cross-dataset validation** (DexYCB / HO-Cap have real sensor depth) — check the
   placement win holds off HOT3D's rendered depth. The harness is dataset-agnostic given an
   adapter like `make_rc_input.py`.
3. **Extend the benchmark** past the current 6 (→12) clips, especially non-symmetric objects.
4. **The hand side** — the campaign was object-focused; hand quality, contact, and joint
   hand+object optimization are the other half. (The hand optimizer above is the first step here.)
5. **A genuinely new rotation attack** — only if revisiting the wall. The prior-based family
   is exhausted; unexplored ideas are a *learned* attitude prior or feature-tracking for
   relative rotation. Hard, low odds — the wall is real.

## Runnable recipe

```bash
# envs: rc5090 (pipeline + eval), sam3d5090 (SAM-3D + hand optimizer), forehoi5090 (FoundationPose)
cd compare/hot3d
RC5=/workspace/miniconda3/envs/rc5090/bin/python
FH5=/workspace/miniconda3/envs/forehoi5090/bin/python
CFG=/workspace/code/hoi_recon/render_and_compare/configs/real_forehoi_icp_joint_grasp.yaml

$RC5 run_batch.py selection_fixed.json --arm icpjgr --config $CFG   # 1. pipeline (hand + object mesh)
$FH5 run_fp_hot3d.py <rc_input> <icpjgr_run> <fpauto_run> --mode auto   # 2. best object (FoundationPose)
$RC5 run_hand_reproj.py <icpjgr_run> <rc_input>                        # 3. best hand
$RC5 make_hoi_best_overlay.py <fpauto_run> <icpjgr_run> overlays/hoi_best_<clip>.mp4   # 4. overlay
$RC5 gt_pose_eval_hot3d.py <rc_input> <fpauto_run>                     # 5. score
$RC5 leaderboard.py render                                             # regenerate the leaderboard
```

Full setup from a fresh machine: [`render_and_compare/REPRODUCE.md`](render_and_compare/REPRODUCE.md).
Pass the config as an **absolute path** — `run_batch` runs the pipeline from `render_and_compare/`.

## Caveats worth knowing

- **Benchmark inputs:** `/workspace/datasets/hot3d/rc_input_<num>_<clip>/`. Six fixed clips:
  bottle, mug, vase, potato masher, spatula, puzzle toy.
- **The acceptance gate (`leaderboard.py`) under-credits real wins** — always keep the raw
  per-clip placement/rotation numbers, not just the pass/fail verdict.
- **Convention traps** (all caught by rendering, never by a metric): HOT3D ships two object
  model sets — pose the `object_models_eval` meshes (in meters), not the display meshes; the
  only left-hand MANO file on this box is a fabricated mirror of the right hand; poses are
  quaternion `wxyz` world transforms.
- **Environments:** older pre-Blackwell environments have no compatible GPU kernels — do not
  use. Rebuild recipes: [`compare/hot3d/docs/T4_NOTES.md`](compare/hot3d/docs/T4_NOTES.md).
