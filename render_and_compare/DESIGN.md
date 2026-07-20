# DESIGN — Architecture & implementation

Goal: from a monocular RGB video, recover **4D HOI** (hand-object interaction) = per-frame
hand motion (MANO), object 3D shape + 6D pose trajectory, and contact (when/where). This
compositional pipeline is the **teacher / error-characterization rig** whose cleaned outputs
become pseudo-ground-truth + distillation targets for a future single feed-forward model. It
runs in `mock` mode (synthetic scene, no weights needed) and `real` mode (GPU backends).

Spine follows **CHOIR** (arXiv:2605.20992): coarse contact-agnostic init → generative spatial
rectification → contact-aware joint optimization. Expanded here into swappable modules so
every stage's error is measurable.

Target setting (per project decision): **model-free / unknown object** is the primary branch
(SAM-3D-Objects / BundleSDF); capture setting kept general (third-person via HaMeR+Dyn-HaMR,
egocentric via HaWoR) behind one hand interface.

Companion docs: [README.md](README.md) (usage), [REPRODUCE.md](REPRODUCE.md) (setup + run).

---

## Terminology

Project-specific vocabulary for this pipeline — what a term *is*, not how it's implemented.
Design decisions live in [`docs/adr/`](docs/adr/). For the repo-wide decoder of method codes
(`icpjgr`, `any6dp`, `fpauto`, ...), metrics, and conda environments, see the top-level
[GLOSSARY.md](../GLOSSARY.md).

**Hand**
- **Hand reprojection** — the projection of the estimated MANO hand into an RGB frame.
  "Correct" reprojection means the projected hand lands on the *observed* hand pixels.
  Improving it is the subject of
  [ADR-0001](docs/adr/0001-hand-reprojection-optimizer.md). _Avoid_: hand overlay, hand
  alignment (both overloaded).
- **kp2d** — HaMeR's 21 per-frame 2D hand keypoints, in full-image pixels, OpenPose joint
  order, already un-mirrored for left hands. The hand's primary image-space evidence.
  _Avoid_: joints2d, keypoints (ambiguous with the 3D `joints`).
- **Hand-reprojection optimizer** — the frozen-object `joint_opt.py` pass that aligns the
  MANO hand to `kp2d` + hand silhouette (image-first), correcting wrist 6D + finger
  articulation while contact stays soft. _Avoid_: hand refiner.
- **Frozen-object arm** — an arm whose object trajectory is held fixed downstream of stage 4
  (e.g. `any6dp`, `fpauto` — see [GLOSSARY.md](../GLOSSARY.md)), so stage-7 optimization moves
  only the hand. Contrast with `joint_grasp`, which moves both.

**Object**
- **Arm** — one end-to-end pipeline configuration + pose core, named and benchmarked
  (`icpjgr`, `fpauto` — see [GLOSSARY.md](../GLOSSARY.md)). The unit of
  comparison in the HOT3D benchmark harness (`compare/hot3d/`).
- **Mesh-controlled** — a comparison in which every arm registers/places the *same* SAM-3D
  stage-3 mesh, so only the pose/placement method differs (removes SAM-3D GPU nondeterminism
  as a confound).

---

## Pipeline stages

The pipeline has 9 stages (`stage0`…`stage8`); each caches a self-contained bundle to disk.

### At a glance

| stage | does | real backend |
|---|---|---|
| 0 preprocess | video → frames, metric depth, camera | MoGe-2 (depth+K), optional VGGT/DA3/VIPE |
| 1 detect/track | hand boxes + object masks | WiLoR-YOLO + SAM 2.1 |
| 2 hand | per-frame MANO + (CHOIR) isolated fit | HaMeR (+ optional Dyn-HaMR) |
| 3 object | textured mesh + 6D pose track | SAM-3D-Objects + render-compare / CHOIR tracker |
| 4 align | one metric frame (+ CHOIR ray-scale) | numpy geometry |
| 5 coarse fit | temporal smoothing | numpy |
| 6 rectify | contact correspondences (+ object placement) | numpy |
| 7 contact optim | **final 4D HOI** (the "fine stage") | `joint_opt` **or** `choir_fine_opt` (sam3d env) |
| 8 eval | error report + reprojection overlay videos | numpy + cv2 |

### Full stage spec

**Stage 0 — Preprocess & camera**
- In: raw RGB video.
- Out: frames; intrinsics `K[3,3]`; extrinsics `[T,4,4]` (world→cam); metric depth.
- Models: VIPE (extrinsics/intrinsics); MoGe-v2 / Depth-Anything-V2 / Metric3D-v2 (metric
  depth); DROID-SLAM fallback.
- Errors: monocular scale ambiguity; camera drift on low parallax; rolling shutter / blur.
- Log: depth reprojection error & confidence; VIPE reprojection residual; parallax.

**Stage 1 — Detect, sides & segmentation (2D cues)**
- In: frames.
- Out: hand boxes + L/R; interacting-object box; object masks (modal + amodal); object point
  tracks.
- Models: WiLoR det-head (+ interacting-object box head) or 100DOH; SAM 2; amodal video seg
  (Chen 2025); CoTracker3.
- Errors: mask leakage/loss under occlusion; L/R swaps & ID switches; track drift on
  specular/rotating objects.
- Log: mask IoU stability; hand-object mask overlap; track confidence.

**Stage 2 — Hand reconstruction (per-frame → world)**
- In: frames + hand boxes/sides + camera.
- Out: per-frame MANO (θ,β,orient,transl); world-space stabilized trajectory; joints3d.
- Models: HaMeR (per-frame) + Dyn-HaMR (temporal/world); HaWoR (egocentric).
- Errors (dominant): root-depth/translation ambiguity; jitter; β drift across frames.
- Log: 2D keypoint reprojection vs wrist depth; acceleration (jitter); β variance.

**Stage 3 — Object shape + 6D pose (model-free primary)**
- In: frames + amodal masks + metric depth + camera + point tracks.
- Out: object mesh (canonical) + scale; per-frame 6D pose `[T,4,4]`.
- Models: SAM-3D-Objects (anchor mesh + guarded follow-track) **and/or** BundleSDF (RGB-D
  neural SDF over whole clip). CAD branch (FoundationPose/MegaPose) kept as a calibration
  control.
- Errors (highest in model-free): anchor-frame shape/scale ambiguity; 6D drift in occluded
  contact phase; symmetry flips.
- Log: multi-view silhouette IoU of mesh; per-frame mask-reprojection IoU; rotational jumps.

**Stage 4 — Spatial alignment**
- In: hand motion + object trajectory + depth + camera.
- Out: hand & object in ONE metric world frame; resolved global scale gauge.
- Method: express both via camera extrinsics; solve one global similarity (Umeyama) to metric
  depth.
- Errors: residual hand↔object scale mismatch — quantify the contact-frame surface gap. This
  is the misalignment CHOIR's later stages exist to fix.

**Stage 5 — Contact-agnostic 4D fit (coarse)** ← initial watchable result
- In: aligned scene + masks + 2D keypoints.
- Out: temporally smooth hand motion + object 6D trajectory; constant β; **no contact
  reasoning**.
- Method: joint per-clip smoothing + (silhouette/keypoint reprojection in real mode).

**Stage 6 — Generative rectification + contact correspondences**
- In: coarse 4D HOI + object geometry.
- Out: rectified relative placement + per-frame **barycentric contact correspondences**.
- Model: flow-matching grasp prior trained on GraspPair (≈500k DexGraspNet grasps); predicts
  ray-depth corrections. Mock/fallback: heuristic snap-to-contact.
- Correspondences: KNN (k≈50) on object surface, valid if distance < 2 cm and surface-normal
  angle < 60°.

**Stage 7 — Contact-aware joint optimization (final)**
- In: rectified frames + correspondences + stage-1/5 evidence.
- Out: refined hand motion, object shape, 6D trajectory, per-frame contact maps.
- Losses: `L_contact` (pull active hand verts to barycentric anchors) + `L_pen` (one-sided
  non-penetration) + `L_silhouette` + `L_anchor` (prior) + `L_temporal`, with a periodically
  rebuilt soft contact cache.

**Stage 8 — Evaluation & error attribution (research payload)**
- Per-stage residual + confidence; ablate one module at a time.
- Metrics: hand MPJPE/PA-MPJPE + accel; object ADD(-S), mask IoU, traj smoothness; contact
  F1/IoU, penetration depth/volume; contact-frame surface gap.
- Export: stage-7 outputs as pseudo-GT; intermediate signals (masks, depth, stage-6 grasp
  corrections, contact maps) as distillation targets for the single feed-forward network.

### Error-budget intuition (what the feed-forward model must internalize hardest)
1. Monocular **scale/depth at the wrist** (stage 2/4).
2. **Occluded-contact relative placement** (stage 6) — image evidence underdetermines it;
   needs a grasp prior.
3. Model-free **object shape/scale** (stage 3).

---

## Code structure

```
hoi_recon/                         main package (conda env: hoi_recon, torch 2.11)
  cli.py            entry point (python -m hoi_recon.cli)
  pipeline.py       stage orchestration, caching, --stages selection
  config.py         YAML + CLI config (attribute-accessible Config; _REPO_ROOT anchor)
  bundle.py         on-disk inter-stage IO (arrays.npz + meta.json + assets)
  geometry.py       SE3, meshes, KNN, normals, penetration
  object_pose_track.py   silhouette-vs-SAM2 object rotation tracker (numpy/cv2)
  joint_grasp.py    rigid joint hand+object grasp optimizer (torch; non-articulated fallback)
  choir.py          CHOIR coarse algorithm: hand isolated fit (Eq1), 60deg guard, ray-scale
  stages/           stage0..stage8 (each a run(ctx) -> Bundle)
  backends/
    real_perception.py   GPU backend drivers (MoGe, SAM2, YOLO, HaMeR, SAM-3D, render-compare,
                         joint optimizer, CHOIR fine optimizer, Dyn-HaMR, VIPE, FoundationPose)
  viz/
    viser_app.py    interactive 4D HOI web viewer (hoi-recon-view)
    reproject.py    reprojection-overlay validation videos
  choir_fine/       === CHOIR fine-stage library (pure, unit-tested) ===
    presets.py      energy-term weight presets: CHOIR_FAITHFUL (locked) / COMBINED_V2
    terms_torch.py  differentiable energy terms: contact(Eq6), penetration(Eq23),
                    velocity/acceleration, keypoint_reproj (Geman-McClure)
    registry.py     assemble_energy(weights, values) — sum weighted active terms
    contact.py      barycentric contact correspondence (top-K, distance+normal gates)
    phases.py       clip phase segmentation (pre_static/approach/manipulation/release/post)
    anatomical.py   MANO twist-splay-bend constraint
    metrics.py      contact-gap + penetration proxy metrics
    step.py         compute_geometric_terms(state) -> term-value dict (used by the optimizer)

scripts/
  subprocess_entries/<repo>/   entry scripts run in the sam3d-objects env (torch 2.5):
    sam-3d-objects/  sam3d_infer.py, render_compare.py, choir_object_fit.py,
                     joint_opt.py, choir_fine_opt.py     <- the two Stage-7 optimizers
    vggt/vggt_geom.py | Dyn-HaMR/dynhamr_track.py | vipe/vipe_camera.py | FoundationPose/fp_track.py
  object_confidence.py   per-frame confidence + jitter + hand/object metrics (eval tool)
  compare_coarse.py      side-by-side coarse-HOI overlay video
  setup_*.sh             env / third_party / checkpoint / choir-env setup

configs/             YAML presets (see below)
tests/               pytest unit tests: test_choir_fine_*.py + test_smoke.py
third_party/         cloned model repos (gitignored; entry scripts installed by setup_third_party.sh)
checkpoints/         model weights (gitignored)
runs/                per-clip outputs (gitignored)
```

**Two conda envs.** The main `hoi_recon` env runs stages 0–6 + orchestration. Heavy
differentiable components (SAM-3D, PyTorch3D render-compare, the Stage-7 optimizers, VGGT,
FoundationPose) run as cached **subprocesses** in a `sam3d-objects` env via `conda run`,
because their torch/numpy pins conflict. The Stage-7 optimizer subprocess imports the tested
`hoi_recon.choir_fine` library via `PYTHONPATH` (set by the driver).

## Configs (pick with `--config`)

| config | pipeline | Stage-7 optimizer |
|---|---|---|
| `default.yaml` | mock (synthetic, no weights) | numpy object-only optim |
| `new.yaml` | our validated real pipeline | `joint_opt` (render-compare object + articulated grasp) |
| `combined.yaml` | best-of-both (CHOIR hand fit + our object) | `joint_opt` |
| `choir.yaml` | CHOIR coarse reproduction (A/B study) | — (coarse only) |
| `choir_faithful.yaml` | full CHOIR stack | `choir_fine_opt`, `fine_preset: choir_faithful` |
| `combined_v2.yaml` | our stack + improvement toggles | `choir_fine_opt`, `fine_preset: combined_v2` |
| `egocentric/third_person.yaml` | scene-tuned variants | — |

The Stage-7 dispatch (`stage7_contact_optim.py`) is additive: `fine_preset` set → the new
registry optimizer (`choir_fine_opt`); else `optim.differentiable` → `joint_opt`; else
fallbacks.

(For the two arms benchmarked as the current best result on HOT3D, `icpjgr` and `fpauto` —
see [GLOSSARY.md](../GLOSSARY.md) — the run recipe is in [REPRODUCE.md](REPRODUCE.md), not
this config table, which predates that work.)

---

## The CHOIR fine-stage work (design note)

Built via spec → plan → TDD. The **term-registry optimizer**:

- **Energy terms** (`choir_fine/terms_torch.py`) are pure, unit-tested torch functions
  matching CHOIR §4.3 equations. The **registry** (`registry.py`) sums `weight * value` over
  active terms.
- **Presets** (`presets.py`): `choir_faithful` = CHOIR §7.3 weights, **locked + tested**
  against drift; `combined_v2` = faithful + our toggles (currently `hand_sil` on).
- **The optimizer** (`scripts/subprocess_entries/sam-3d-objects/choir_fine_opt.py`) assembles
  `compute_geometric_terms(state)` + render terms via the registry, runs per-group Adam
  (object 3e-4 / finger 5e-4 / wrist 5e-5, 800 iters), selected by `fine_preset`.
- Validated end-to-end on `runs/grab` (loss converges, sane 4D output).

### Key measured finding (fine-stage ablation, same coarse, vary only `fine_preset`)

| | choir_faithful | combined_v2 |
|---|---|---|
| hand precision (median) | 0.680 | **0.695** |
| hand IoU (median) | 0.523 | **0.555** |
| hand centroid err p10 (worst frames) | 32.9px | **24.3px** (−26%) |
| object dc-IoU / mask_cov | 0.953 / 0.980 | tied |
| penetration sum | 42.2 | 45.2 |

- `combined_v2`'s `hand_sil` toggle measurably improves hand registration (esp. the
  worst-frame tail), object tied — the "build-and-improve" payoff, ablated cleanly.
- **But both are far worse than our `joint_opt` baseline (0.86 precision / 35px)** on the
  hand, because faithful CHOIR weights `anc_2d=0.5 ≪ contact=1000` — it optimizes a *grasp
  prior*, not *image registration*. Surpassing the baseline needs more toggles (raise
  `anc_2d`, add the axis-split anchor), each now a one-line preset change.

### Running & evaluating this work

```bash
# CHOIR-faithful fine stage  /  our improved fine stage
python -m hoi_recon.cli --video examples/grab.mp4 --out runs/x --real --config configs/choir_faithful.yaml
python -m hoi_recon.cli --video examples/grab.mp4 --out runs/y --real --config configs/combined_v2.yaml

# metrics (per-frame confidence, jitter, hand/object registration, contact, penetration)
python scripts/object_confidence.py --run runs/grab

# unit tests for the choir_fine library
conda run -n hoi_recon python -m pytest tests/ -q   # 55 passing
```

(For the main pipeline's setup + run instructions, see [REPRODUCE.md](REPRODUCE.md).)

### What's next for this work (planned, not yet built)

From the CHOIR fine-stage design spec (dev branch):
1. **Tune `combined_v2`** — add `anc_2d`/axis-split toggles to beat the 35px hand baseline.
2. **Eval/ablation harness** (`scripts/ablate_fine.py`) — formalize the per-toggle A/B table.
3. **Phase 2 — generative ray-depth rectifier** (flow-matching on GraspPair / DexGraspNet;
   compute available, dataset gated).
4. **HO3D GT benchmark** adapter — MPJPE + object pose error for a defensible "beats CHOIR"
   claim.

*Session arc (commits `b24cf4a`..`7a1faf9`): brainstormed the fine-stage design → 3 TDD plans
→ foundation library → terms+registry → optimizer integration (smoke-validated) → fine-stage
ablation. 55 unit tests passing.*
