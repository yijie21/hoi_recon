# hoi_recon — Compositional 4D Hand-Object Interaction Reconstruction

A modular, **runnable** pipeline that reconstructs 4D hand-object interaction (HOI)
from a monocular RGB video by composing best-of-breed perception models, then
refining them with contact-aware geometric optimization.

The design follows the three-stage spine of **CHOIR** (Contact-aware 4D HOI
Reconstruction, arXiv:2605.20992): *coarse contact-agnostic init → spatial
rectification → contact-aware joint optimization*, expanded into explicit,
swappable modules so each stage's error can be measured.

This repo is built as a **research / error-characterization rig**, not just a demo:
it runs end-to-end **today** in `mock` mode (no checkpoints needed) by generating a
synthetic HOI scene and injecting realistic per-stage perception error, so you can
watch the refinement stages drive the error back down (`stage8_eval`). Swap each
perception stage to its real backend once weights are downloaded.

```
Stage 0  Preprocess & camera     video → frames, intrinsics, camera traj, metric depth
Stage 1  Detect & track          → hand boxes/sides, object box, masks (modal+amodal)
Stage 2  Hand reconstruction     → per-frame MANO + world-space stabilized motion
Stage 3  Object shape + 6D pose  → object mesh + 6D pose trajectory
Stage 4  Spatial alignment       → hands & object in ONE metric world frame
Stage 5  Contact-agnostic fit    → coarse 4D HOI (temporally smooth, still floating)
Stage 6  Generative rectify      → corrected relative placement + contact correspondences
Stage 7  Contact-aware optim     → final 4D HOI (hand, object, 6D traj, contact maps)
Stage 8  Evaluation              → per-stage error attribution, pseudo-GT export
```

## Quickstart (mock mode — runs now, no weights)

```bash
conda env create -f environment.yml
conda activate hoi_recon

# Run the whole pipeline on a synthetic HOI clip and print the error report.
python -m hoi_recon.cli --out runs/demo --mock --num-frames 48

# Or via the console script after `pip install -e .`
hoi-recon --out runs/demo --mock
```

You should see a table at the end showing hand joint error, object translation
error, penetration depth and contact F1 *before* (raw perception) vs *after*
(contact-aware optimization), e.g.:

```
  metric                                raw(percep) →  refined
  ------------------------------------------------------------
  hand MPJPE (mm)                         5.979 →      2.570 mm  (+57%)
  hand jitter/accel                       0.007 →      0.001     (+83%)
  object transl err (mm)                 22.333 →     20.648 mm  (+8%)
  penetration depth sum                  22.704 →      6.874     (+70%)
  contact-frame gap (mm)                  2.722 →      1.776 mm  (+35%)
  contact F1                              0.592 →      0.560     (-0.03)
```

How to read it (this is the whole point of the rig):
- **Hand** error and jitter fall sharply — temporal stabilization (stage5) removes
  the zero-mean monocular jitter injected by stage2.
- **Penetration** and **contact-frame gap** fall sharply — the contact-aware
  optimization (stage6→7) is doing its job: physically plausible, in-contact HOI.
- **Object translation** improves modestly: stage6 rectification does most of the
  object localization; stage7 mainly refines penetration.
- **Contact F1** is roughly flat (a small dip): reducing penetration nudges the
  object outward, which trades a few proximity-contacts. This *real tension*
  (less penetration ⇄ fewer contacts) is exactly the kind of trade-off you want a
  research rig to expose, not hide. Raise the object shape-scale error in
  `stage3_object.py` to watch the contact-recovery bottleneck get worse — that is
  the model-free object-shape error in the DESIGN.md budget.

## View the 4D reconstruction (viser)

Visualize the final reconstructed hand-object interaction in your browser —
animated over time, with contact highlighting:

```bash
pip install viser                         # if not already in the env
python -m hoi_recon.viz.viser_app --run runs/demo     # or:  hoi-recon-view --run runs/demo
# then open the printed http://localhost:8080 URL
```

In the viewer:
- the **object** is a mesh transformed by its per-frame 6D pose;
- the **hand** is a point cloud (MANO mesh if a real backend provides faces) — contact
  candidate fingertips are orange, and **vertices in active contact turn red**;
- toggle **contact lines** to draw segments from each in-contact hand vertex to the
  object surface; toggle **joints** for the 21-keypoint skeleton;
- use the **frame** slider or **play / pause** + **speed** to scrub the 4D interaction;
- the panel shows live active-contact count and min surface gap per frame.

Point `--stage` at any stage bundle to compare, e.g. the coarse fit vs the final:

```bash
hoi-recon-view --run runs/demo --stage stage5_coarse_fit   # floating / penetrating
hoi-recon-view --run runs/demo --stage stage7_contact_optim # contact-consistent
```

## Real mode (GPU)

**For step-by-step reproduction on a fresh machine (exact versions, checkpoint tree,
troubleshooting) see [`REPRODUCE.md`](REPRODUCE.md).** Quick version:

```bash
# 0. one-time: env + third-party repos + python deps + weights
conda env create -f environment.yml && conda activate hoi_recon
bash scripts/setup_third_party.sh      # clone model repos into third_party/
bash scripts/setup_real.sh             # torch(cu128) + MoGe + SAM2 + ultralytics + HaMeR deps
bash scripts/download_checkpoints.sh   # fetch MoGe / SAM2 / WiLoR / HaMeR weights (hf+wget)
#   then place MANO_RIGHT.pkl (license) — see that script's final notes
#   plus the second conda env for the heavy differentiable components:
#   `sam3d-objects` (SAM-3D-Objects + PyTorch3D; see third_party/sam-3d-objects/doc/setup.md)

# 1. run the composed pipeline on a clip — THE VALIDATED CONFIGURATION
python -m hoi_recon.cli --video examples/grab.mp4 --out runs/grab --real \
    --hand hamer --object sam3d --depth moge --config configs/new.yaml
hoi-recon-view --run runs/grab           # view the 4D result
```

**`--config configs/new.yaml` matters.** It switches on the redesigned
differentiable pipeline (the result in `runs/grab`):

- `backend.object_pose: render_compare` — object **rotation** is recovered from the
  object's own image evidence: a fast numpy silhouette tracker
  (`hoi_recon/object_pose_track.py`) seeds a **differentiable render-and-compare**
  refinement (PyTorch3D; silhouette IoU + a *photometric* term against the SAM-3D
  textured mesh, which recovers the spin-about-axis DOF a silhouette can't see).
- `optim.differentiable: true` — stage 7 runs the **joint optimizer**: MANO
  articulation + object 6D optimized together under silhouette / photometric /
  contact / non-penetration energies, so the fingers actually curl to grasp
  (instead of the rigid-hand fallback in `hoi_recon/joint_grasp.py`).
- `backend.sam3d_env: sam3d-objects` — the heavy components (SAM-3D mesh
  generation, render-compare, joint optimizer, optionally VGGT / FoundationPose)
  run as cached **subprocesses in that second conda env**, because their torch /
  numpy pins conflict with this env's MoGe/SAM2 stack.

Without `--config configs/new.yaml` you get the older path: silhouette-only object
rotation and the rigid (non-articulated) grasp optimizer.

### What each real backend uses (verified on an RTX 5080 / CUDA 12.8)

| stage | backend | model | status |
|------|---------|-------|--------|
| 0 depth + intrinsics | `--depth moge` | **MoGe-2** (metric depth, camera K; identity extrinsics) | ✅ validated path |
| 0 consistent camera + depth | `--depth vggt` | **VGGT** (one consistent camera traj + depth, sam3d env subprocess) | ⚙️ wired+validated, but **up-to-scale** — metric scale resolution inside the optimizer is WIP |
| 0 depth + camera poses | `--depth da3` | **Depth-Anything-3** (metric depth + intrinsics + real extrinsics) | ⚙️ wired (clone+install DA3 to use) |
| 1 hand detection | — | **WiLoR YOLO** detector (no detectron2) | ✅ working |
| 1 object mask | — | **SAM 2.1** (point-prompted, propagated) | ✅ working |
| 2 hand → MANO | `--hand hamer` | **HaMeR** (boxes from stage 1; depth-anchored into the metric frame; MANO params threaded through to the stage-7 optimizer) | ✅ working — needs **MANO** (license) |
| 2 hand, MANO-free | `--hand depthlift` | hand box + MoGe depth → corresponded 3D grid | ✅ working (fallback, no license needed) |
| 3 object shape | `--object sam3d` | **SAM-3D-Objects** textured mesh (sam3d env subprocess), metric-scaled from depth; fails soft to the model-free **depth-lift** convex hull | ✅ working |
| 3 object 6D pose | `object_pose: render_compare` | silhouette tracker → differentiable render-and-compare (silhouette + photometric); alternatives: `silhouette`, `foundationpose`, `hand` | ✅ working |
| 5–7 align / smooth / joint optim | — | this repo's geometry + the differentiable joint optimizer (sam3d env) | ✅ working |

All stages run end-to-end on real video today (with MANO in place for `--hand
hamer`; use `--hand depthlift` to run without the license). Stage 8 additionally
writes reprojection-overlay videos (`*_reproj.mp4` in the run dir) to validate the
reconstruction against the input video.

### Real-mode notes / caveats

- **Two conda envs.** The main `hoi_recon` env runs stages 0–2 (MoGe, SAM2, YOLO,
  HaMeR). The `sam3d-objects` env (name configurable via `backend.sam3d_env`) hosts
  SAM-3D-Objects, PyTorch3D render-compare, the joint optimizer, VGGT and
  FoundationPose — invoked via `conda run` subprocesses with results cached per run
  dir (e.g. `stage3_object/sam3d/object.npz`, `stage3_object/rc/poses.npz`,
  `stage7_contact_optim/jo/out.npz`; delete a cache file to recompute that piece).
- **MANO is license-gated.** Register at https://mano.is.tue.mpg.de, accept the
  license, and place `MANO_RIGHT.pkl` under `checkpoints/mano/` (flat or the
  `mano_v1_2/models/` archive layout both work). It cannot be downloaded via
  `hf`/`gdown`. Until then `--hand hamer` stops with a clear `BackendNotAvailable`
  pointing here — or run MANO-free with `--hand depthlift`.
- **chumpy / numpy.** The official MANO `.pkl` is loaded through `chumpy`, whose
  import breaks on `numpy>=1.24`; this repo patches the removed numpy aliases at
  runtime (`_patch_numpy_for_chumpy`) so HaMeR works in the main env without
  downgrading numpy.
- **Hand placement.** HaMeR's own absolute depth is unreliable (fabricated focal
  length), so the hand is re-anchored to the MoGe metric depth at the hand box —
  hand and object share one metric camera frame.
- **Object stays image-grounded.** A deliberate design invariant: the depth-lift
  centroid track reprojects onto the real object to a few px, so the optimizers
  keep the object on that track (strong prior) and move/articulate the **hand** to
  close the grasp — not the other way around.
- **Camera extrinsics.** With `--depth moge` they are identity (static-camera
  assumption). For moving-camera clips use `--depth vggt` (consistent geometry,
  scale WIP) or `--depth da3` (metric depth + real poses;
  `pip install -e third_party/Depth-Anything-3` first).
- **Object prompt.** SAM2 is prompted at the detected hand-box centre (the held
  object sits in the grasp); replace with an interacting-object detector or a user
  click for tricky scenes.

Real backends live in `hoi_recon/backends/real_perception.py`; each import-guards its
dependency and raises a clear `BackendNotAvailable` with setup instructions if a repo
or weight is missing — the pipeline degrades gracefully instead of crashing opaquely.

## Layout

```
hoi_recon/
  cli.py            entry point
  pipeline.py       stage orchestration, caching, resumability
  config.py         yaml + CLI config
  bundle.py         on-disk inter-stage IO (arrays.npz + meta.json + assets)
  geometry.py       SE3, meshes, KNN, Umeyama, normals, penetration
  object_pose_track.py  silhouette-vs-SAM2-mask object rotation tracker (numpy/cv2)
  joint_grasp.py    rigid joint hand+object grasp optimizer (torch; non-articulated fallback)
  mock/scene.py     deterministic synthetic HOI + ground-truth contacts
  stages/           stage0..stage8
  backends/real_perception.py  GPU backends: MoGe, VGGT, DA3, SAM2, YOLO, HaMeR,
                    SAM-3D, depth-lift, render-compare + joint-optimizer subprocess drivers
  viz/viser_app.py  interactive 4D HOI web viewer
  viz/reproject.py  reprojection-overlay validation videos
configs/            new.yaml (VALIDATED differentiable pipeline) / default / egocentric / third_person
scripts/            setup_third_party.sh, setup_real.sh, download_checkpoints.sh, run_demo.sh, view_demo.sh
scripts/subprocess_entries/  entry scripts run in the sam3d-objects env (sam3d_infer.py,
                    render_compare.py, joint_opt.py, vggt_geom.py, fp_track.py);
                    installed into the third_party/ clones by setup_third_party.sh
third_party/        populated by setup_third_party.sh (gitignored)
checkpoints/        populated by download_checkpoints.sh
```

## Why mock mode injects error on purpose

Your research goal is to *characterize where errors enter* a composed pipeline so
you can later distill it into a single feed-forward network. In `mock` mode:

- `stage2` (hand) injects monocular **depth/translation** ambiguity + jitter.
- `stage3` (object) injects **shape-scale** error + 6D **pose drift**.
- `stage4–7` are the real geometric algorithms that fight that error back down.
- `stage8` compares every stage against the synthetic ground truth.

So the rig is a controlled sandbox for the exact thesis of the project: *each
module introduces error; contact-aware joint reasoning removes it.* On real video
the same `stage8` exports `stage7` output as pseudo-GT for the feed-forward model.

See `DESIGN.md` for the full stage-by-stage spec, model choices, and error budget.
