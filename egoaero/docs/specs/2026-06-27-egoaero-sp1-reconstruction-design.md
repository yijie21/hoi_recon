# EgoAERO reproduction — SP1: Asset-free hand-object reconstruction (design)

Date: 2026-06-27
Status: approved (brainstorming) → ready for implementation plan
Method folder: `egoaero/` (a self-contained method in the HOI-reconstruction workbench)

## 0. Context: the full reproduction and where SP1 sits

We are reproducing **EgoAERO** (*Learning Dexterous Manipulation from a Single
Egocentric Video without Object Assets*; parsed paper at
`egoaero/egoaero.pdf_by_PaddleOCR-VL-1.6.md`). The user chose **full-pipeline (A+B+C)**
scope, decomposed into four sub-projects built in order:

- **SP1 — Reconstruction pipeline (Part A, Sec 2.1, App A–C)** ← *this spec*. The spine.
- **SP3 — Online quality assessment (Part C, Sec 3, App E)** — depends on SP1 output.
- **SP2 — Two-stage residual RL policy (Part B, Sec 2.2, App D)** — depends on SP1; Isaac Gym + assets.
- **SP4 — EgoDex-R dataset schema + collection loop (App F)** — schema/synthetic only.

Each sub-project gets its own spec → plan → implementation cycle. This document covers
**only SP1**.

### Operating principle (applies to every sub-project)

The paper has substantial gaps (no released data/assets, missing hyperparameters, an
incomplete Appendix B). Agreed handling:

> **Faithful where specified; principled documented defaults at every gap (cited to the
> system it is borrowed from); mock data where none exists; every deviation/assumption
> logged in `egoaero/ASSUMPTIONS.md`.**

This makes the reproduction runnable end-to-end today and auditable against the paper.

## 1. Goal of SP1

From a single egocentric RGB-D video, reconstruct **contact-consistent hand-object
trajectories without object assets**: per-frame MANO hand, an object mesh, the object
6-DoF pose trajectory, and contact maps — all expressed in a stable table frame.

SP1 must:
- Run **end-to-end in `--mock` mode today** (synthetic ego scene, no weights/data), the
  same "runs now" guarantee as the sibling `render_and_compare` method.
- Expose **swappable real backends** (HaWoR, SAM3, SLAM, BundleSDF-style tracker, SAM3D)
  as optional stubs that the mock path does not require.
- Emit output conforming to the **workbench method contract** (root `README.md`): a
  per-clip `egoaero/runs/<clip>/` bundle with MANO hand + object mesh + object 6-DoF
  trajectory + contact maps, plus the shared eval metrics, so `egoaero` is directly
  comparable to `render_and_compare`.

Out of scope for SP1: RL policy learning (SP2), quality-assessment scoring (SP3), the
dataset/collection loop (SP4).

## 2. Architecture

A staged pipeline mirroring the proven `render_and_compare` shape: each stage is a pure
`run(ctx) -> Bundle`, caches a self-contained bundle to disk, and is independently
unit-testable. Stages map 1:1 to paper subsections.

```
egoaero/
  egoaero/                      package (self-contained; no imports from sibling methods)
    cli.py                      entry: python -m egoaero.cli --mock --out runs/demo
    pipeline.py                 stage orchestration, caching, --stages selection
    config.py                   YAML + CLI config; _METHOD_ROOT anchor
    bundle.py                   on-disk inter-stage IO (arrays.npz + meta.json + assets)
    contract.py                 final workbench-contract writer/validator
    core/                       vendored minimal shared core (adapted, not sibling-imported)
      mano.py                   MANO forward kinematics + standard fingertip/pad vertex sets
      geometry.py               SE3 / Lie exp-log, KNN, normals, signed-distance, penetration
      mock_scene.py             synthetic ego RGB-D HOI scene generator (+ injected errors)
    stages/
      stage0_ego_io.py          2.1   RGB-D frames, intrinsics, timestamps
      stage1_semantic.py        2.1.1 target/related objects, SAM3 prompts, seed frame, stage labels
      stage2_track.py           2.1.2/App A  coarse RANSAC init + memory-pool pose graph
      stage3_mesh.py            2.1.2/App B  neural SDF field → coarse mesh, SAM3D fine, align
      stage4_hand.py            2.1.3 HaWoR MANO + RGB-D depth translation correction
      stage5_ego_comp.py        2.1.4 SLAM → table frame, hand-pixel down-weight, smooth
      stage6_contact.py         2.1.5/App C  adaptive contact optimization (faithful)
      stage7_eval.py            reconstruction metrics vs mock GT
    backends/
      real.py                   optional real-backend drivers (HaWoR/SAM3/ORB-SLAM3/BundleSDF/SAM3D)
    configs/
      mock.yaml                 synthetic, no weights (default)
      real.yaml                 real backends (gated on installs)
  docs/specs/                   this spec
  tests/                        per-stage unit tests + end-to-end mock smoke test
  ASSUMPTIONS.md                every default/deviation from the paper, with citation
  README.md                     method README (contract compliance + usage)
  runs/, checkpoints/, third_party/   gitignored runtime artifacts (under the method folder)
```

`config.py` anchors `_METHOD_ROOT` via `__file__` so `runs/`, `checkpoints/`,
`third_party/` resolve under `egoaero/` (same self-contained pattern proven in the
sibling method).

## 3. Stage specifications

Each entry: **purpose**, **inputs → outputs**, **mock behavior** (what runs now),
**real backend**, **faithfulness** (and which defaults land in `ASSUMPTIONS.md`).

### Stage 0 — ego-io (2.1)
- **Purpose:** provide the RGB-D observation stream.
- **I/O:** (mock: none) → frames `I_{1:T}`, depth `D_{1:T}`, intrinsics `K`, timestamps.
- **Mock:** `core/mock_scene.py` generates a synthetic ego clip — a hand grasping/moving/
  placing an object on a table, with (a) ego head motion baked into the camera trajectory,
  (b) rendered depth, (c) object & hand masks, (d) partial hand-over-object occlusion. It
  also stores **ground-truth** hand MANO, object pose, object mesh, and table frame for
  the eval stage and for error injection downstream.
- **Real backend:** load real RGB-D video + intrinsics.
- **Faithfulness:** scene generator is a test fixture, not from the paper; documented.

### Stage 1 — semantic preprocessing (2.1.1)
- **Purpose:** identify the manipulated object (+ related objects), produce SAM3 prompts,
  pick the seed frame, and label coarse manipulation stages (grasp/move/place) for the
  contact stage's operation prior.
- **I/O:** frames + task description → `M^O_{1:T}`, `M^H_{1:T}` masks, target/related object
  ids, seed-frame index, per-frame stage labels.
- **Mock:** return GT object/hand masks and stage labels from the synthetic scene; seed
  frame = least-occluded frame by visible-object-area.
- **Real backend:** MLLM (keyframe sampling + prompt generation) → SAM3 segmentation.
- **Faithfulness:** method faithful; MLLM identity, keyframe count, prompt format, and the
  "less-occluded" criterion are unspecified in the paper → defaults logged.

### Stage 2 — object tracking + memory-pool pose graph (2.1.2 / App A)
- **Purpose:** 6-DoF object trajectory `T_{1:T}` without assets.
- **I/O:** RGB-D + `M^O` → `T_t ∈ SE(3)` per frame; keyframe memory pool `P`.
- **Sub-steps (faithful to App A):**
  1. **Coarse init `T̃_t`:** back-project visible object region, establish frame-to-frame
     RGB-D correspondences, lift to 3D, **RANSAC** rigid fit, pick max-inlier hypothesis.
  2. **Memory pool:** quality score `q_t = α_v A^O_t + α_d R^D_t + α_θ C^θ_t − α_h H^occ_t`;
     insert when above threshold and adds view coverage.
  3. **Keyframe subset:** `s(k,t) = β_o Overlap − β_r d_R(R_k,R̃_t) + β_q q_k`; top-K.
  4. **Pose graph** over `V_t = {t} ∪ K_t`: minimize
     `Σ λ_f E_feat + λ_g E_geo + λ_s E_sdf + λ_m E_mask + λ_p E_pose` (Eqs in App A);
     update on the Lie algebra `T_i ← exp(δξ^) T_i`; write back to memory.
- **Mock:** start from GT object pose with **injected per-frame drift + occlusion gaps**;
  run the *real* memory-pool optimizer so the module is genuinely exercised and the eval
  shows tracking error driven back down.
- **Real backend:** BundleSDF/FoundationPose-style tracker [16,17].
- **Faithfulness:** structure faithful; **all weights (α,β,λ), top-K, thresholds, the
  feature matcher, the robust kernel ρ, the optimizer, and the unspecified `E_mask`
  equation are documented defaults** (cited to BundleSDF [16]) → `ASSUMPTIONS.md`.

### Stage 3 — neural field + coarse-to-fine mesh (2.1.2 / App B)
- **Purpose:** object mesh `M_O` without assets.
- **I/O:** posed keyframe RGB-D → coarse mesh `M^coarse_O` (zero level set of `Ω_Θ`),
  SAM3D fine mesh `M^sam_O`, aligned to coarse (rigid+scale) → final `M_O`.
- **Mock:** use the synthetic object mesh + surface noise; run the alignment step for real.
- **Real backend:** online neural SDF field training + SAM3D [19].
- **Faithfulness:** **Appendix B is a stub** — field architecture, ray sampling, and all
  five loss terms (`L_surf, L_free, L_occ, L_rgb, L_eik`) and their λ weights are
  undefined. We adopt standard neural-SDF definitions (occlusion-aware surface/free-space
  /eikonal losses per BundleSDF [16] / NeuS-style) as documented defaults → `ASSUMPTIONS.md`.
  Mesh alignment = Umeyama rigid+scale (default, logged).

### Stage 4 — hand pose (2.1.3)
- **Purpose:** per-frame MANO + depth-corrected global translation.
- **I/O:** RGB (+ depth) → `(θ_t, β_t, ^{C_t}T_{H_t})`; sequence-average shape `β̄`;
  camera-frame hand mesh/joints; depth translation correction `Δp^C_t`.
- **Mock:** synthetic MANO from the scene + **injected monocular global depth bias**;
  run the *real* depth-correction so the eval shows the bias removed.
- **Real backend:** HaWoR [21] → MANO; depth correction via robust residual of visible
  hand surface vs predicted vertices.
- **Faithfulness:** method faithful; the robust estimator and neighborhood size for the
  translation correction are defaults → logged.

### Stage 5 — ego-motion compensation (2.1.4)
- **Purpose:** move all states into one stable **table frame** `T`.
- **I/O:** RGB-D + hand masks → camera trajectory; transform hand+object into table frame;
  light temporal smoothing of object trajectory and hand-root translation.
- **Mock:** invert the known synthetic head/camera trajectory; estimate the table frame by
  plane-fitting the tabletop points of the first reliable frame.
- **Real backend:** ORB-SLAM3 [22] with hand-pixel down-weighting.
- **Faithfulness:** method faithful; **table-frame definition, the SLAM hand-pixel
  down-weighting mechanism, and smoothing params are defaults** → logged. No table/vertical
  constraints imposed on the object (explicit paper instruction).

### Stage 6 — adaptive contact optimization (2.1.5 / App C) — **faithful**
- **Purpose:** bounded, geometry-level contact correction of the replay hand; object pose,
  object mesh, MANO shape, and articulation are **unchanged**.
- **I/O:** hand verts/joints `V^T_t, J^T_t`, object mesh `M_O`, object pose `T_{O,t}` →
  corrected `V^{rT}_t, J^{rT}_t` + contact maps; written back to table and camera frames.
- **Steps (faithful to App C):** active window (validity + frame ranges + optional stage
  prior); three contact regions (thumb pulp, dynamically-selected opposing non-thumb
  fingertip via min-distance Eq, thenar/hukou); whole-hand translation toward contact via
  `d^k_t` aggregation + clip; finite-window **triangular** temporal smoothing + boundary
  taper; local finger offsets weighted by MANO finger chain `α^f_i`; **penetration
  push-back** with depth ReLU + clip.
- **Paper constants (used verbatim):** contact gap **0.5 mm**, thenar gap **1.8 mm**, max
  whole-hand translation **34 mm**, max local finger displacement **15 mm**, max penetration
  push-back **8 mm**, smoothing window **9**, boundary transition **6 frames**, whole-hand
  rotation **disabled (0°)**.
- **Defaults (logged):** region weights `w_k`, per-region gap `g_k` for the opposing finger,
  the finger-chain weight profile `α^f_i`, and penetration threshold `ε`.
- **Real backend:** none — pure geometry, runs identically in mock and real.

### Stage 7 — reconstruction eval
- **Purpose:** quantify reconstruction quality and the contact-opt improvement.
- **Metrics:** object rotation/translation error vs GT (mock), hand joint error, contact gap
  (median per finger), penetration depth sum, temporal jitter — reported **before vs after**
  contact optimization (the "watch error fall" report), plus the shared workbench metric set
  for cross-method comparison.

## 4. Data flow & contract

Stages chain through `bundle.py` (`arrays.npz` + `meta.json` + assets) under
`egoaero/runs/<clip>/stageN/`. `contract.py` writes the final workbench-contract layout
(per-frame MANO hand, object mesh, object 6-DoF trajectory, contact maps in the table
frame) and validates it, so `egoaero` plugs into the workbench Methods table next to
`render_and_compare`.

## 5. Testing strategy

- **Per-stage unit tests** (style of existing `tests/test_choir_fine_*`): pose-graph
  reduces injected drift; depth correction removes injected bias; contact-opt honors the
  App-C bounds and reduces penetration/contact-gap; mesh alignment recovers a known
  rigid+scale transform; table-frame transform composes correctly.
- **End-to-end `--mock` smoke test**: run all 8 stages; assert the Stage-7 report shows
  penetration and contact-gap **down** after Stage 6, and object/hand error **down** after
  Stages 2/4 — the same end-to-end validation the sibling method uses.
- Tests are CWD-safe and require no weights/data.

## 6. Deliverables

1. `egoaero/` self-contained method (package, stages, core, configs, CLI).
2. Runnable `python -m egoaero.cli --mock --out runs/demo` producing the contract bundle +
   before/after error report.
3. `egoaero/ASSUMPTIONS.md` enumerating every default/deviation with paper-section and
   citation.
4. `egoaero/README.md` (contract compliance + usage) and a Methods-table row in the root
   `README.md`.
5. Passing unit + smoke tests.

## 7. Deferred (later sub-projects)

SP3 (quality assessment, App E), SP2 (two-stage RL policy, App D — Isaac Gym, retargeting,
Er/Et/Ej/Eft/SR), SP4 (EgoDex-R schema + collection loop, App F). Each follows its own
spec → plan → implementation cycle after SP1 lands.
