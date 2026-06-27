# CHOIR Fine Stage — Build-and-Improve Design Spec

**Date:** 2026-06-12
**Status:** Approved design, pending implementation plan
**Paper:** CHOIR — Contact-aware 4D HOI Reconstruction (arXiv:2605.20992), Stage 2
(generative spatial rectification) + Stage 3 (contact-aware joint optimization).

---

## 1. Goal & context

Reproduce CHOIR's *fine stage* and **build on it** ("build-and-improve"): start from
a faithful CHOIR reproduction and fold in our improvements, aiming to beat CHOIR — not
just match it. We have already built a strong **combined coarse** stage (`configs/
combined.yaml`: our render-compare object + CHOIR hand isolated fit). This spec covers
the two fine stages that sit on top of that coarse init.

CHOIR has **no public code**, so "beat CHOIR" means: our improved fine stage vs **our
own faithful CHOIR reproduction**, measured on shared metrics, plus a GT benchmark.

Decisions locked during brainstorming:
- **Scope:** both fine stages, **sequenced** — Stage 3 (optimization) first, Stage 2
  (generative rectifier) as Phase 2.
- **Purpose:** build-and-improve (CHOIR base + our improvements from the start).
- **Eval:** proxy metrics (fast) **and** a GT benchmark (defensible).
- **Phase 2 data:** compute available (local A100s), DexGraspNet acquisition is a
  **gated prerequisite** → Stage 3 must run fully independent of Phase 2.

## 2. Architecture (Approach 1: one configurable optimizer + toggles)

A single **term-registry optimizer**: `L_HOI = Σ wᵢ · termᵢ(state)`, where each energy
term is a named function and its weight comes from config. "Faithful CHOIR" vs
"improved" is *which terms are active and their weights* — so per-improvement ablation
is a one-line config change.

Three config presets:
- **`choir_faithful`** — CHOIR's exact terms/weights/LRs/phases (§7.3). **Locked** and
  guarded by a test asserting the weights, so it cannot silently drift.
- **`combined`** — today's shipping pipeline (`configs/combined.yaml` behavior),
  unchanged; regression-tested.
- **`combined_v2`** — faithful CHOIR + our improvements, each an independent toggle.

Decoupling: Stage 3 and Stage 2 communicate through a **rectified-init + contact-
correspondence interface**, so Phase 1 ships and runs without Phase 2's trained model.
The Stage 3 optimizer runs in the existing `sam3d-objects` env as a cached subprocess,
like `joint_opt.py` today.

```
Phase 1 (data-free):  Stage 3 contact-aware joint optimization
  - evolve joint_opt.py -> named-term registry optimizer
  - presets: choir_faithful (locked) | combined (current) | combined_v2 (faithful+toggles)
  - shared upstream: barycentric contact correspondence + phase segmentation
Phase 2 (gated on DexGraspNet):  Stage 2 generative ray-depth rectifier
  - standalone module + clean interface Stage 3 consumes
  - faithful (per-frame scalar Δz flow-matching) + improved (temporal-native, object-grounded)
  - GraspPair data pipeline specced; training gated on data acquisition
Cross-cutting:  evaluation harness (proxies + HO3D GT) + drift-guarding tests
```

## 3. Phase 1 — Stage 3 contact-aware joint optimization

### 3.1 Energy-term registry

`L_HOI = λ_con·L_con + λ_pen·L_pen + λ_sil·L_sil + λ_anc·L_anc + λ_temp·L_temp`
(CHOIR Eq 5). Mapping CHOIR faithful → our current `joint_opt.py` → action:

| term | CHOIR faithful (§7.3) | our current joint_opt.py | action |
|---|---|---|---|
| L_contact | soft **barycentric** corr., top-K=8, softmax σ=0.01, confidence c_{t,i}; w=1000 | nearest-vertex pull, w=5 | **upgrade to barycentric top-K** (new component 3.2) |
| L_template/bridge/gap/patch | contact-family stabilizers, annealed | — | **add** (faithful), optional in v2 |
| L_pen | one-sided, ε=5mm, clip 0.04, w=500 | one-sided signed-dist, w=30 | reweight to faithful (have the mechanism) |
| L_sil (object) | **amodal** mask MSE, w=500 | don't-care IoU + photometric | faithful=amodal MSE; v2 toggle T1/T2 = our don't-care + photometric |
| L_anc | 2D(0.5) + anat(30) + pose(100) + obj-pose(100) | kp2d(3) + hand-anchor(z6/xy1) + obj-priors(50/50) | reweight; **add anatomical** (twist-splay-bend) |
| L_temp | 8 terms: vel(500/500/200/200/500) + acc(1000/5000/3000) | vel(2) + accel(hand1/obj20) | **expand to full 8-term set**, faithful weights |
| L_hand_sil | — (none in CHOIR) | hand silhouette vs SAM2 hand mask, w=1 | **v2 toggle T3** (our improvement) |

Variables optimized (both presets): MANO pose θ (per frame) + per-frame translation +
wrist rotation; object 6D (R, T) per frame. Shape/betas and object scale fixed from the
coarse stage. Rotations in 6D / rotation-matrix space (avoid axis-angle wrap-around).

Optimizer: Adam, **800 iterations**, per-group LRs — object **3e-4**, finger pose
**5e-4**, wrist rot/transl **5e-5** (CHOIR §7.3).

### 3.2 New shared components

1. **Barycentric contact correspondence** (replaces nearest-vertex contact):
   - canonical alignment (Procrustes) → sample **10,000** object surface points (fixed seed)
   - per MANO contact vertex (fingertips, pads, palm): K=50 NN → keep valid by **2cm**
     distance + **60°** normal cone → store top-K=8 as (face_id, barycentric)
   - **periodic cache rebuild** during optimization from current geometry (thresholds
     **5cm** / **60°**, top-K=8, softmax σ=0.01), with temporal memory for stability
   - implements CHOIR Eq 6 + §7.3.
2. **Phase segmentation** — pre-static / approach / manipulation / release / post-static;
   contact terms active only on frames with correspondences, other terms full-sequence;
   missing phases skipped. Thresholds per CHOIR §7.3 (phase-detection details to be
   reproduced; flagged as a faithful-approximation risk in §6).
3. **Anatomical (twist-splay-bend) MANO constraint** — penalize anatomically invalid
   finger poses; w=30.

### 3.3 `combined_v2` improvement toggles (each independently ablatable)

- **T1 don't-care occlusion IoU** (object silhouette) — occlusion-aware vs plain amodal
  MSE. *Evidence:* heavy-occlusion object IoU held where FoundationPose dropped.
- **T2 photometric** — recovers the spin DOF a silhouette can't see.
- **T3 hand silhouette** vs SAM2 hand mask — CHOIR has no hand image term; *evidence:*
  hand centroid err 44→35px, worst-frame tail 17→9px.
- **T4 object rotation anchor + stronger object accel** — *evidence:* rotation jitter
  p90 0.132→0.063, worst-frame shake −23%.
- **T5 axis-split hand anchor** (strong-z / weak-xy) — kp2d wins laterally while depth
  stays metric.

## 4. Phase 2 — Stage 2 generative ray-depth rectifier (gated on data)

### 4.1 Faithful variant (CHOIR)

Flow-matching network predicting a **per-frame scalar ray-depth correction Δz**:
- `v_θ(z_τ, τ | J^h, r, s, P^o, n^o)` (Eq 4): token embeddings (τ, z, hand-joints, scale,
  ray) → self-attention → **PointNet++ object encoder** (coordinate-sensitive, not
  rotation-invariant) → cross-attention → **AdaLN** time conditioning → scalar velocity.
- Inference: ODE-integrate from noisy z₀ over τ∈[0,1] → Δz; apply `J^h ← J^h + Δz·r` on
  interaction frames.

**Technical nuance (shapes the improvement):** sliding the hand along the camera ray by
Δz **preserves its 2D projection** — 2D keypoints cannot constrain Δz (depth is exactly
the unobservable DOF, which is why a learned grasp prior is needed). So improvements come
from **temporal structure** and **leaning on the strong object**, not from 2D terms here
(those live in Stage 3).

### 4.2 Improved variant (`v2`)

1. **Temporally-native** — condition on a window of frames / predict a short Δz
   trajectory, smooth by construction (vs CHOIR's independent per-frame Δz de-jittered
   later by Stage 3).
2. **Grounded on our strong object** — condition on our render-compare metric object pose
   (~7px reprojection) instead of a canonical anchor mesh only.
3. **Occlusion/confidence-aware** — weight the correction by per-frame object visibility;
   occluded frames lean on the temporal prior.

### 4.3 Interface (Stage 2 ↔ Stage 3 decoupling)

```
Stage 2 emits:  rectified_hand_init[T] (or Δz[T])  +  contact_correspondences (barycentric)
Stage 3 consumes:  rectified init -> pose-prior anchor;  correspondences -> contact term
WITHOUT Stage 2:  Stage 3 falls back to combined-coarse init + builds correspondences
                  from coarse geometry  -> Phase 1 runs standalone
```

### 4.4 GraspPair data pipeline (specced; training gated)

```
DexGraspNet grasps  ->  PyBullet stability filter (gravity + perturbation, keep ~200/obj)
  ->  sample object surface points + normals (P^o, n^o)
  ->  inject anisotropic ray-aligned noise on the hand (large along-ray var, mild in-plane,
      anatomy-preserving MANO)  ->  training pair (noisy hand + ray + obj geom -> clean target)
Training:  flow-matching velocity loss; compute on local A100s; DATASET = gated prereq.
```
Until DexGraspNet is acquired, Phase 2 = module + interface + data-pipeline code,
testable on a tiny **synthetic** grasp stub (validates the training-loop shape).

## 5. Evaluation harness

**Proxy metrics** (fast, every preset, no GT) — extend `scripts/object_confidence.py`
(already: reprojection IoU, mask coverage, centroid error, jitter) with:
- **contact gap** (median hand↔object surface distance at contact frames)
- **penetration depth** (summed signed penetration)
- **hand registration** (kp2d reprojection residual / MPJPE-proxy)

Run on `choir_faithful` / `combined` / `combined_v2` → **A/B + ablation table**, each
toggle T1–T5 its own row.

**GT benchmark** (defensible): a **HO3D** adapter (recommended; DexYCB as a second option)
reporting **hand MPJPE** + **object pose error** (rotation/translation, ADD-S) as CHOIR
does. Components: GT loader, runner feeding GT clips through the pipeline, metric
computation. Gating step for any published "beats CHOIR" claim.

## 6. Testing & faithful-preset locking

- **Faithful-preset lock test** — asserts `choir_faithful`'s weights/LRs/active terms
  match the CHOIR §7.3 numbers verbatim; fails if anyone edits them. The anti-drift guard
  that makes Approach 1 safe.
- **Per-term unit tests** — each registry term computes correctly on known inputs
  (gradient sign, zero-at-optimum).
- **Regression test** — `combined` preset reproduces today's `runs/grab_combined` within
  tolerance, so the refactor does not break the shipping pipeline.
- **Stage 2 synthetic-stub test** — flow-matching model + data pipeline run end-to-end on
  a tiny synthetic grasp set (no DexGraspNet), validating the training-loop shape before
  real data arrives.
- **Ablation harness** — `scripts/ablate_fine.py` runs all presets/toggles on a clip and
  emits the comparison table.

## 7. Faithfulness caveats / known gaps

- **Amodal masks**: CHOIR's L_sil uses amodal masks (Chen et al. 2025 amodal segmenter,
  not in our repo). Faithful preset approximates amodal with modal SAM2 + rendered-mesh
  extent, documented as a gap; or wire an amodal segmenter later. `combined_v2` sidesteps
  this (don't-care IoU is occlusion-aware by design).
- **Phase-detection thresholds**: CHOIR defers exact phase-detection schedules to
  supplementary; we reproduce them as faithfully as the text allows and flag residual
  approximation.
- **Stage 2 training hyperparameters** (epochs/batch/lr schedule) are not fully specified
  in the paper; we choose standard flow-matching defaults and document the deviation.

## 8. Success criteria

- **Phase 1 ships**: `choir_faithful` and `combined_v2` run end-to-end on `grab.mp4`,
  produce final 4D HOI, and the ablation table attributes each toggle's effect.
- **Improvement demonstrated** on proxies: `combined_v2` beats `choir_faithful` on contact
  gap, penetration, hand registration, and jitter (target: each no worse, ≥2 strictly
  better), object metrics held.
- **Defensible claim**: HO3D hand MPJPE + object pose error reported for both presets;
  `combined_v2` ≤ `choir_faithful`.
- **No regression**: `combined` preset reproduces today's `runs/grab_combined`.
- **Drift-guarded**: faithful-preset lock test passes.

## 9. Out of scope

- Re-implementing CHOIR's coarse stage (done; `configs/choir.yaml`).
- Multi-hand / two-hand interaction.
- Real-time inference.
- Wiring Dyn-HaMR/VIPE envs (separate, `scripts/setup_choir_envs.sh`).
