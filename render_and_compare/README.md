# hoi_recon — the reconstruction pipeline

A runnable pipeline that reconstructs a **4D hand-object interaction** from a monocular
video: it chains together best-of-breed perception models (depth, hand, object shape),
then refines them with contact-aware geometric optimization. It is built as a research
rig — every stage's error can be measured — not just a demo.

> Names like `combined.yaml`, `render_compare`, MANO, SAM-3D are decoded in
> [`../GLOSSARY.md`](../GLOSSARY.md). Setup on a fresh machine: [`REPRODUCE.md`](REPRODUCE.md).
> Architecture + error budget: [`DESIGN.md`](DESIGN.md).

## The 9 stages

```
Stage 0  Preprocess & camera     video → frames, intrinsics, camera trajectory, metric depth
Stage 1  Detect & track          → hand boxes/sides, object box, masks
Stage 2  Hand reconstruction     → per-frame MANO hand + stabilized motion
Stage 3  Object shape + 6D pose  → object mesh + per-frame position/rotation
Stage 4  Spatial alignment       → hands & object in ONE metric world frame
Stage 5  Coarse fit              → temporally smooth 4D interaction (still floating)
Stage 6  Rectify                 → corrected placement + contact correspondences
Stage 7  Contact-aware optim     → final 4D interaction (hand curls to grasp, no penetration)
Stage 8  Evaluation              → per-stage error report + final output export
```

Each stage caches its output to `stage<N>_*/` and is skipped on re-run (`--force` to
recompute, `--stages` to select a subset). Final output:
`runs/<name>/stage8_eval/pseudo_gt.npz` = object mesh + per-frame object poses.

## Quickstart (mock mode — runs now, no weights)

Mock mode builds a synthetic interaction, injects realistic perception error, and lets
you watch the refinement drive it back down — the fast way to see the pipeline work.

```bash
conda env create -f environment.yml && conda activate hoi_recon
python -m hoi_recon.cli --out runs/demo --mock --num-frames 48   # prints an error report
```

The report compares each metric *before* (raw perception) vs *after* (contact-aware
optimization) — e.g. hand joint error 6.0 → 2.6 mm, penetration 22.7 → 6.9. Hand error,
jitter, and penetration fall sharply; object placement improves modestly; contact F1 is
roughly flat (reducing penetration trades a few proximity contacts — a real tension the
rig is designed to expose).

## View the result (browser viewer)

```bash
pip install viser
hoi-recon-view --run runs/demo                       # animated 4D interaction
hoi-recon-view --run runs/demo --stage stage5_coarse_fit   # compare an earlier stage
```

The object is a posed mesh; the hand is a mesh/point cloud with **contact vertices in
red**. Scrub with the frame slider or play/pause; the panel shows live contact count and
surface gap per frame.

## Real mode (GPU)

Full setup — exact versions, checkpoint tree, MANO, second env, troubleshooting — is in
**[`REPRODUCE.md`](REPRODUCE.md)**. Short version:

```bash
conda env create -f environment.yml && conda activate hoi_recon
bash scripts/setup_third_party.sh      # clone model repos into third_party/
bash scripts/setup_real.sh             # torch (cu128) + MoGe + SAM2 + WiLoR/HaMeR deps
bash scripts/download_checkpoints.sh   # fetch MoGe / SAM2 / WiLoR / HaMeR weights
#   then place license-gated MANO_RIGHT.pkl under checkpoints/mano/
#   plus a second conda env (SAM-3D + PyTorch3D) for the heavy differentiable steps

# best single-command pipeline configuration:
python -m hoi_recon.cli --video examples/grab.mp4 --out runs/grab_combined --real \
    --config configs/combined.yaml
hoi-recon-view --run runs/grab_combined
```

> **Note on "best".** `configs/combined.yaml` is the best *single-command* pipeline. The
> best *benchmark* result (object + hand, scored on HOT3D) comes from the harness in
> [`../compare/hot3d/`](../compare/hot3d/), which swaps in a learned object core
> (FoundationPose) and a dedicated hand optimizer — see the top-level
> [`../README.md`](../README.md).

### Which config?

| config | what it is | when |
|---|---|---|
| **`combined.yaml`** | **best pipeline** — image-based object rotation + a cleaner hand init (CHOIR fit) + contact-aware optimization | use this |
| `new.yaml` | the prior validated path (same object tracker, plain hand — no CHOIR fit) | the baseline `combined.yaml` builds on |
| `choir.yaml` | reproduces the CHOIR *coarse* init only, for A/B study | research, not production |

`combined.yaml` = `new.yaml` **plus** one addition: a per-frame rigid hand fit in stage 2
that registers the MANO hand to its 2D keypoints + wrist depth, cutting the coarse hand's
reprojection error from ~63 px to ~10 px. Everything else is identical.

### What each real backend uses (verified on RTX 5080 / CUDA 12.8)

| stage | option | model | status |
|---|---|---|---|
| 0 depth + intrinsics | `--depth moge` | MoGe-2 (metric depth + camera; static-camera assumption) | ✅ validated |
| 0 moving camera | `--depth vggt` / `--depth da3` | VGGT (up-to-scale) / Depth-Anything-3 (metric + real poses) | ⚙️ wired |
| 1 hand detection | — | WiLoR YOLO detector | ✅ |
| 1 object mask | — | SAM 2.1 (point-prompted, propagated) | ✅ |
| 2 hand → MANO | `--hand hamer` | HaMeR, depth-anchored; `combined.yaml` adds the CHOIR fit | ✅ needs MANO (license) |
| 2 hand, MANO-free | `--hand depthlift` | hand box + MoGe depth (no license) | ✅ fallback |
| 3 object shape | `--object sam3d` | SAM-3D textured mesh; falls back to a depth-lift hull | ✅ |
| 3 object rotation | `object_pose: render_compare` | silhouette tracker → differentiable render-and-compare | ✅ |
| 5–7 align / smooth / grasp | — | this repo's geometry + joint optimizer | ✅ |

### Key caveats

- **Two conda environments.** The main `hoi_recon` env runs stages 0–2 (MoGe, SAM2, YOLO,
  HaMeR). A second env (SAM-3D + PyTorch3D, name set by `backend.sam3d_env`) runs the heavy
  differentiable steps as cached subprocesses. See [`REPRODUCE.md`](REPRODUCE.md).
- **MANO is license-gated** (https://mano.is.tue.mpg.de) — place `MANO_RIGHT.pkl` under
  `checkpoints/mano/`. It can't be auto-downloaded; without it, use `--hand depthlift`.
  (The repo patches the numpy/chumpy incompatibility at runtime, so no numpy downgrade is needed.)
- **The object stays image-grounded** (design invariant). The object's tracked position
  reprojects onto the real object to a few pixels, so the optimizer trusts that track and
  moves/curls the **hand** to close the grasp — not the other way around.
- **Backends degrade gracefully.** Each real backend (`hoi_recon/backends/real_perception.py`)
  raises a clear `BackendNotAvailable` with setup instructions if a repo or weight is missing,
  instead of crashing opaquely.

## Layout

```
hoi_recon/
  cli.py / pipeline.py / config.py   entry point, stage orchestration + caching, config
  bundle.py / geometry.py            inter-stage IO; SE3, meshes, KNN, penetration
  object_pose_track.py               silhouette-based object rotation tracker
  joint_grasp.py                     rigid hand+object grasp optimizer (torch)
  stages/                            stage0..stage8
  backends/real_perception.py        GPU backends (MoGe, SAM2, YOLO, HaMeR, SAM-3D, ...)
  viz/                               browser viewer + reprojection-overlay videos
configs/    combined.yaml (best) / new.yaml / choir.yaml / default / egocentric / third_person
scripts/    setup_third_party.sh, setup_real.sh, download_checkpoints.sh, subprocess_entries/
third_party/ + checkpoints/          populated by the setup scripts (gitignored)
```

See [`DESIGN.md`](DESIGN.md) for the full stage-by-stage architecture and error budget.
