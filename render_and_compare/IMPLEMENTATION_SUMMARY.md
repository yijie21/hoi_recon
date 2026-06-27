# Implementation Summary

A reading guide to the `hoi_recon` codebase: what it does, how the code is organized, the
configs, and where the CHOIR fine-stage work sits. Companion docs:
`README.md` (usage), `DESIGN.md` (stage spec), `RESEARCH_DIRECTIONS.md` (research roadmap),
`docs/superpowers/specs|plans/` (the fine-stage design + implementation plans).

---

## 1. What it is

A modular pipeline that reconstructs **4D hand-object interaction (HOI)** from a monocular RGB
video, following the three-stage spine of **CHOIR** (arXiv:2605.20992):
*coarse contact-agnostic init → spatial rectification → contact-aware joint optimization*.
It runs in `mock` mode (synthetic, no weights) and `real` mode (GPU backends).

The 9-stage pipeline (each stage caches a self-contained bundle to disk):

| stage | does | real backend |
|---|---|---|
| 0 preprocess | video → frames, metric depth, camera | MoGe-2 (depth+K), optional VGGT/DA3/VIPE |
| 1 detect/track | hand boxes + object masks | WiLoR-YOLO + SAM 2.1 |
| 2 hand | per-frame MANO + (CHOIR) isolated fit | HaMeR (+ optional Dyn-HaMR) |
| 3 object | textured mesh + 6D pose track | SAM-3D-Objects + render-compare / CHOIR tracker |
| 4 align | one metric frame (+ CHOIR ray-scale) | numpy geometry |
| 5 coarse fit | temporal smoothing | numpy |
| 6 rectify | contact correspondences (+ object placement) | numpy |
| 7 contact optim | **final 4D HOI** (the "fine stage") | joint_opt **or** choir_fine_opt (sam3d env) |
| 8 eval | error report + reprojection overlay videos | numpy + cv2 |

---

## 2. Code structure

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
  choir_fine/       === CHOIR fine-stage library (NEW; pure, unit-tested) ===
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

configs/             YAML presets (see §3)
tests/               pytest unit tests (55 passing): test_choir_fine_*.py + test_smoke.py
docs/superpowers/    specs/ (fine-stage design) + plans/ (3 implementation plans)
third_party/         cloned model repos (gitignored; entry scripts installed by setup_third_party.sh)
checkpoints/         model weights (gitignored)
runs/                per-clip outputs (gitignored)
```

**Two conda envs.** The main `hoi_recon` env runs stages 0–6 + orchestration. Heavy
differentiable components (SAM-3D, PyTorch3D render-compare, the Stage-7 optimizers, VGGT,
FoundationPose) run as cached **subprocesses** in a `sam3d-objects` env via `conda run`, because
their torch/numpy pins conflict. The Stage-7 optimizer subprocess imports the tested
`hoi_recon.choir_fine` library via `PYTHONPATH` (set by the driver).

---

## 3. Configs (pick with `--config`)

| config | pipeline | Stage-7 optimizer |
|---|---|---|
| `default.yaml` | mock (synthetic, no weights) | numpy object-only optim |
| `new.yaml` | our validated real pipeline | `joint_opt` (render-compare object + articulated grasp) |
| `combined.yaml` | best-of-both (CHOIR hand fit + our object) | `joint_opt` |
| `choir.yaml` | CHOIR coarse reproduction (A/B study) | — (coarse only) |
| **`choir_faithful.yaml`** | full CHOIR stack | **`choir_fine_opt`**, `fine_preset: choir_faithful` |
| **`combined_v2.yaml`** | our stack + improvement toggles | **`choir_fine_opt`**, `fine_preset: combined_v2` |
| `egocentric/third_person.yaml` | scene-tuned variants | — |

The Stage-7 dispatch (`stage7_contact_optim.py`) is additive: `fine_preset` set → the new
registry optimizer (`choir_fine_opt`); else `optim.differentiable` → `joint_opt`; else fallbacks.

---

## 4. The CHOIR fine-stage work (this session)

Built via spec → plan → TDD (superpowers workflow). The **term-registry optimizer**:

- **Energy terms** (`choir_fine/terms_torch.py`) are pure, unit-tested torch functions matching
  CHOIR §4.3 equations. The **registry** (`registry.py`) sums `weight * value` over active terms.
- **Presets** (`presets.py`): `choir_faithful` = CHOIR §7.3 weights, **locked + tested** against
  drift; `combined_v2` = faithful + our toggles (currently `hand_sil` on).
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

- `combined_v2`'s `hand_sil` toggle measurably improves hand registration (esp. the worst-frame
  tail), object tied — the "build-and-improve" payoff, ablated cleanly.
- **But both are far worse than our `joint_opt` baseline (0.86 precision / 35px)** on the hand,
  because faithful CHOIR weights `anc_2d=0.5 ≪ contact=1000` — it optimizes a *grasp prior*, not
  *image registration*. Surpassing the baseline needs more toggles (raise `anc_2d`, add the
  axis-split anchor), each now a one-line preset change.

---

## 5. How to run

```bash
# our best real pipeline
python -m hoi_recon.cli --video examples/grab.mp4 --out runs/grab --real --config configs/combined.yaml
hoi-recon-view --run runs/grab                     # 4D viewer in browser

# CHOIR-faithful fine stage  /  our improved fine stage
python -m hoi_recon.cli --video examples/grab.mp4 --out runs/x --real --config configs/choir_faithful.yaml
python -m hoi_recon.cli --video examples/grab.mp4 --out runs/y --real --config configs/combined_v2.yaml

# metrics (per-frame confidence, jitter, hand/object registration, contact, penetration)
python scripts/object_confidence.py --run runs/grab

# tests
conda run -n hoi_recon python -m pytest tests/ -q   # 55 passing
```

---

## 6. What's next (planned, not yet built)

From `docs/superpowers/specs/2026-06-12-choir-fine-stage-design.md`:
1. **Tune `combined_v2`** — add `anc_2d`/axis-split toggles to beat the 35px hand baseline.
2. **Eval/ablation harness** (`scripts/ablate_fine.py`) — formalize the per-toggle A/B table.
3. **Phase 2 — generative ray-depth rectifier** (flow-matching on GraspPair / DexGraspNet;
   compute available, dataset gated).
4. **HO3D GT benchmark** adapter — MPJPE + object pose error for a defensible "beats CHOIR" claim.

---

*Session arc (commits `b24cf4a`..`7a1faf9`): brainstormed the fine-stage design → 3 TDD plans →
foundation library → terms+registry → optimizer integration (smoke-validated) → fine-stage
ablation. 55 unit tests passing.*
