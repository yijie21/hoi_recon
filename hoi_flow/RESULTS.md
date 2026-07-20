# hoi_flow refiner — v1 campaign results

Consolidated evaluation of the trained bridge flow-matching HOI refiner.
Checkpoint under test: `hoi_flow/runs/v2_full/ckpt_best.pt` (best-by-val, **step 6000**;
trained with per-sample corruption scale `s ~ U(0, 1.5)`).

Names follow the project [`GLOSSARY.md`](../GLOSSARY.md): *coarse* = the shipped object/hand
stack's output (fpauto object + hand-reprojection optimizer); *refined@N* = the refiner run
for N Euler steps; *along-ray* = camera-depth error, *perp* = image-plane error.

---

## 1. What the refiner is

A **bridge flow-matching** model that takes a *coarse* 4D hand-object trajectory and refines
it toward the true trajectory, conditioned on the egocentric observations. It operates on a
**57-D per-frame state** (object 6D-rotation + translation; per hand: wrist 6D-rotation +
translation + 15 PCA MANO thetas) over 32-frame windows. The bridge interpolant is
source-anchored — a straight path from the (Gaussian-smoothed) coarse `x0` to the GT
endpoint `x1`, noise vanishing at `t=1` — so an untrained net is the identity refiner and the
model only learns the *correction*. Conditioning is **pluggable** (RGB via frozen DINOv2,
ray-cast depth, seg-mask, object point cloud, intrinsics), fed as cross-attention tokens with
per-modality dropout so any stream can be absent at test time. The core is a **47M-parameter
DiT** temporal transformer (d=512, depth=8, 8 heads) predicting the endpoint field.

---

## 2. Tier-1 — controlled evaluation

### 2a. Synthetic multi-scale val (the "preserve good / fix bad" question)

All 220 held-out val segments (participants P0002/P0015), windowed exactly as training's val
(fixed seed), corrupted at a **fixed** scale `s`. `s=0.25` is a near-perfect source (~6 mm
object); `s=1.0` is the full calibrated coarse-error regime (~25 mm object).

| scale | variant | obj_along | obj_perp | obj_total | obj_rot | hand_along | hand_perp | theta_mae |
|---|---|---|---|---|---|---|---|---|
| 0.25 | coarse | 3.68 | 3.98 | **6.25** | 3.08 | 10.44 | 3.78 | 0.010 |
| 0.25 | refined@8 | 3.85 | 6.61 | **8.68** | 4.97 | 11.15 | 6.70 | 0.021 |
| 0.25 | refined@1 | 3.95 | 6.54 | 8.64 | 5.10 | 12.32 | 6.70 | 0.020 |
| 0.5 | coarse | 7.35 | 7.96 | **12.50** | 6.42 | 21.16 | 7.63 | 0.020 |
| 0.5 | refined@8 | 5.72 | 8.22 | **11.43** | 6.74 | 16.99 | 9.17 | 0.022 |
| 0.5 | refined@1 | 5.80 | 8.21 | 11.38 | 6.79 | 17.77 | 9.25 | 0.021 |
| 1.0 | coarse | 14.70 | 15.93 | **25.00** | 13.77 | 42.01 | 15.24 | 0.040 |
| 1.0 | refined@8 | 9.95 | 12.57 | **18.17** | 10.76 | 24.24 | 15.51 | 0.025 |
| 1.0 | refined@1 | 9.70 | 12.59 | 18.15 | 10.75 | 23.96 | 15.60 | 0.025 |

(mm except obj_rot deg, theta_mae rad. Full CSV: `runs/v2_full/tier1_synthetic.csv`.)

**Reading it.** There is a clear **crossover around `s ≈ 0.4`**:

- **`s=1.0` (bad source): the refiner clearly helps.** Object depth (along-ray) 14.7→9.9 mm,
  object total 25→18 mm, hand depth 42→24 mm, articulation 0.040→0.025. This is its trained
  sweet spot — the monocular depth under-constraint it exists to fix.
- **`s=0.5` (mild): roughly break-even** on totals (fixes depth, slightly costs image-plane).
- **`s=0.25` (near-perfect): the refiner makes it WORSE** — object total 6.25→8.68 mm, and it
  *adds* image-plane error everywhere (obj_perp 4.0→6.6, hand_perp 3.8→6.7). The model has a
  **correction floor (~8-9 mm object, ~6-7 mm perp)** it cannot go below; a source already
  better than that floor is pulled up to it.
- **The win is concentrated in the along-ray (depth) component and the hand.** The
  image-plane (perp) component is never improved and is slightly degraded at low corruption.
- **refined@8 ≈ refined@1 on synthetic** — the multi-step sampler behaves like a near-idempotent
  regressor here; extra steps neither help nor hurt in-distribution.

### 2b. Real-pairs stratified (the design-domain question)

The 202 real FoundationPose coarse pairs over P0002/P0015, stratified by the median coarse
**object translation** error vs the segment GT. The distribution is **bimodal** exactly as
[`CALIBRATION.md`](data/CALIBRATION.md) documents (good ≈ 45%, catastrophic ≈ 50%). Hand
columns are gated to the grasping hand the coarse actually provides (real coarse is
single-hand) and pooled over the MANO-carrying files only (`n_mano`).

| stratum | n | n_mano | variant | obj_total | obj_rot | hand_along | hand_perp | theta_mae |
|---|---|---|---|---|---|---|---|---|
| good (<30mm) | 90 | 31 | coarse | **4.2** | **3.3** | 40.4 | 11.0 | 0.577 |
| good (<30mm) | 90 | 31 | refined@8 | **22.7** | **12.9** | 30.2 | 27.5 | 0.506 |
| mid (30-100mm) | 10 | 5 | coarse | **66.7** | 135.2 | 51.4 | 9.8 | 0.555 |
| mid (30-100mm) | 10 | 5 | refined@8 | **66.2** | 126.2 | 19.7 | 24.7 | 0.467 |
| catastrophic (>=100mm) | 102 | 24 | coarse | **329.2** | 131.7 | 48.4 | 13.2 | 0.651 |
| catastrophic (>=100mm) | 102 | 24 | refined@8 | **305.2** | 126.1 | 56.2 | 29.8 | 0.546 |

(mm except obj_rot deg, theta_mae rad. Full CSV: `runs/v2_full/tier1_realpairs.csv`.)

**Reading it (object = the clean signal):**

- **Good stratum (the refiner's design domain): it REGRESSES.** Coarse is already 4.2 mm /
  3.3° — far below the correction floor — and the refiner pushes it to 22.7 mm / 12.9°. This
  is the same floor effect as synthetic `s=0.25`, and it is the same population as the
  benchmark clips.
- **Mid stratum: break-even** (66.7→66.2 mm) — no harm, no object fix.
- **Catastrophic stratum: unrecoverable** (329→305 mm), *as designed* — the bridge is
  source-anchored, so a coarse that has locked onto the wrong instance stays there. This is an
  upstream re-detection failure, not a refinement one.
- **Hand:** depth (along-ray) genuinely improves where there is room (mid 51→20, good 40→30),
  and articulation improves everywhere, but the refiner **inflates image-plane (perp)** every
  time (good 11→28, mid 10→25). Net hand quality is a wash on the good stratum and a win on
  the mid stratum. The real-coarse hand path also inherits the MANO global-wrist convention
  caveat in `CALIBRATION.md` — read the hand rows as indicative, not exact.

---

## 3. The ablation — what pixels buy (Tier-1 val, s=1, final EMA)

Final `refined@8` val rows (24-seg val set, corruption s=1) across the campaign runs:

| run | conditioning | corrupt_scale | obj_total | obj_rot | hand_along | hand_perp (image-plane) | theta_mae |
|---|---|---|---|---|---|---|---|
| coarse (baseline) | — | — | 26.10 | 13.17 | 42.01 | **15.35** | 0.040 |
| `v1_posesonly` | obj_points + K only | s=1 | 22.41 | 12.21 | 22.51 | **25.37** | 0.0252 |
| `v1_full` | + rgb/depth/seg | s=1 | 21.96 | 11.56 | 20.29 | **18.08** | 0.0250 |
| `v2_full` (shipped) | + rgb/depth/seg | s~U(0,1.5) | **18.15** | 8.81 | 21.41 | **15.34** | 0.0234 |

(mm except obj_rot deg. Rows are end-of-training final val; the shipped `ckpt_best` is the
best-by-val step-6000 EMA — near-identical object, slightly lower hand_along ~17.1 mm.)

**The story.** Both poses-only and full-pixel models recover hand **depth** equally well
(hand_along 42→~22) — grasp/temporal geometry alone fixes depth. But **poses-only WRECKS the
image plane**: it drives hand_perp *above* the coarse baseline (15.4→**25.4 mm**), because
without pixels the model cannot see where the hand/object actually project and drifts
laterally while correcting depth. Adding pixels (`v1_full`) rescues the image plane
(25.4→18.1 mm). The corruption-scale fix (`v2_full`) then pins hand_perp back to the coarse
level (15.3 mm) and cuts object total to 18.2 mm. **Pixels buy image-plane accuracy; the
corruption schedule buys not-overcorrecting.**

---

## 4. Tier-2 — HOT3D benchmark (real coarse stack, mocap GT)

`refined@8` = shipped flowref output; `coarse` = the incumbent object arm as it ships. Object
scored by chamfer via `gt_pose_eval_hot3d.py`; hand by nearest-GT-hand centroid offset. Log:
`scratchpad/tier2_v2.log`.

| clip | arm | gauge mm | chamfer coarse | chamfer ref@8 | chamfer ref@1 | rot_traj coarse | rot_traj ref@8 | hand off coarse | hand off ref |
|---|---|---|---|---|---|---|---|---|---|
| bottle_bbq_002034 | fpauto | 3.0 | **2.9** | **17.7** | 11.2 | 53.5 | 61.0 | 18 | 47 |
| mug_white_001970 | fpauto | 3.5 | **4.1** | **11.8** | 11.0 | 12.2 | 17.4 | 53 | 134 |
| vase_002500 | fpauto | 5.3 | **5.4** | **19.9** | 15.6 | 6.4 | 10.1 | 41 | 76 |
| spatula_red_001990 | fpauto | 5.0 | **12.1** | **29.0** | 21.7 | 5.4 | 7.1 | 19 | 51 |
| puzzle_toy_001964 | fpauto | 15.3 | **15.3** | **13.5** | 13.0 | 21.1 | 32.2 | — | — |
| potato_masher_002349 | icpjgr | 6.8 | **22.0** | **52.4** | 36.9 | 62.4 | 59.8 | 92 | 108 |

(chamfer/rot in mm/deg median. ref@8 is the shipped multi-step output; ref@1 is the
single-step control.)

**Reading it.** On the five near-perfect fpauto clips the coarse object is 2.9–12.1 mm —
below the refiner's floor — so `refined@8` **regresses every one** (bottle 2.9→17.7, mug
4.1→11.8, vase 5.4→19.9, spatula 12.1→29.0). Only **puzzle_toy** (coarse already 15.3 mm,
above the floor) marginally improves (15.3→13.5). Hand offset gets worse everywhere. This is
Tier-1's floor effect on the good stratum, playing out on the benchmark exactly as predicted.
Note that **ref@1 (single step) is consistently less damaging than ref@8** on the real
benchmark — the multi-step sampler *amplifies* the over-correction against real structured
coarse error.

### What we learned (the v1 regression and the corruption-scale fix)

The **v1** checkpoint (trained at a *fixed* corruption scale `s=1.0`) was far worse on the
benchmark: bottle **2.9→32.8 mm**, mug 4.1→24.3, vase 5.4→20.9, spatula 12.1→27.6, masher
22.0→85.8 (`scratchpad/tier2_v1.log`). **Root cause:** training only at `s=1` meant the model
never saw a near-perfect source, so it learned to *always* apply a full calibrated-magnitude
correction — catastrophic on an already-excellent fpauto coarse. **Fix:** per-sample
`corrupt_scale = U(0, 1.5)` (the `v2` change) exposes the model to sources spanning
near-perfect → full-calibrated. This **halved** the benchmark damage (bottle 32.8→17.7) and
made the model helpful at `s=1` while merely-mildly-harmful at `s=0.25` — but it **did not
cross zero**: a ~8-20 mm correction floor remains, so the refiner still degrades the
sub-floor benchmark sources.

---

## 5. Verdict, limitations & next steps

**Benchmark-safety verdict: `ckpt_best` is NOT safe to ship on the good-mode clips.** On five
of six benchmark clips it makes the object chamfer 2–6× worse and the hand offset worse; it is
within-noise-or-better on only one clip (puzzle_toy, the single clip whose coarse sits above
the correction floor). The refiner genuinely wins where the source is *substantially* wrong:
synthetic `s ≥ ~0.5`, the real **mid** stratum (hand depth 51→20 mm), articulation
everywhere, and object depth in the full-calibrated regime. It is not yet a safe drop-in
behind the already-excellent fpauto stack.

**Limitations**

1. **Correction floor / no source-quality gate.** The single biggest issue: the model always
   applies a correction of its trained typical magnitude, so it cannot preserve a source that
   is already better than ~8-20 mm. Needs either a **confidence/quality gate** (refine only
   when coarse is uncertain or above-floor) or an **identity-preserving objective** (penalize
   moving an already-consistent source).
2. **Object diversity.** Trained on the 30-object HOT3D vocabulary; generalization to unseen
   object geometry is unverified.
3. **Catastrophic coarse mode is out of scope.** The ~50% of real windows that lock onto a
   wrong instance are an **upstream re-detection** problem; the source-anchored bridge cannot
   recover them. Re-localization would need the noise→clean conditional variant + a
   catastrophic corruption mode — the **v2 design fork documented in `CALIBRATION.md`**, not a
   v1 patch.
4. **FK / contact `extra_losses` not yet enabled.** The bridge loss has a hook for FK-consistency
   and hand-object contact terms (`bridge.py` `extra_losses`); it is unused — training is pure
   masked endpoint-MSE. These could add the geometric constraints the image plane currently
   lacks.
5. **Image-plane accuracy.** Even with pixels the refiner adds perp error at low corruption;
   an explicit 2D-reprojection guidance/loss term is the obvious lever.
6. **potato_masher metric caveat.** It is a spinning symmetric object on the icpjgr arm; its
   chamfer mixes origin-vs-surface placement with the ambiguous symmetry axis, so its absolute
   numbers are noisier than the fpauto clips — read its regression as directional.
7. **Single-hand real coarse.** Real coarse pairs carry only the grasping hand; the model
   hallucinates any second present hand from scratch. Hand metrics here are gated to the
   provided hand, and the real-coarse hand path inherits the MANO global-wrist convention
   caveat (`CALIBRATION.md`).
8. **seg_mask is train-only.** Tier-2 drops seg_mask as test-time GT leakage; the model relies
   on modality dropout to cope. Tier-1 synthetic keeps it (faithful to training), so the two
   tiers are not on identical conditioning.

**Next steps (priority order):** (1) add a source-quality gate or identity-preserving loss to
kill the floor regression — this is what stands between the model and benchmark-safety;
(2) add a 2D-reprojection guidance term for the image plane; (3) enable the FK/contact
extra_losses; (4) if re-localization is wanted, pursue the noise→clean v2 fork for the
catastrophic mode.

---

## 6. Reproduce

```bash
# env: rc5090 ; GPU 1 ; TORCH_HOME=/workspace/huggingface_cache/torch
cd /workspace/code/hoi_recon

# --- train the v2 refiner (per-sample corruption scale s~U(0,1.5) is the base.yaml default) ---
CUDA_VISIBLE_DEVICES=1 python -m hoi_flow.train --config hoi_flow/configs/base.yaml \
    train.name=v2_full train.steps=30000 train.lr=2e-4 train.warmup=1000 train.num_workers=8
# best-by-val weights -> hoi_flow/runs/v2_full/ckpt_best.pt

# --- Tier-1 (both tables -> runs/v2_full/tier1_synthetic.csv, tier1_realpairs.csv, tier1_tables.md) ---
CUDA_VISIBLE_DEVICES=1 python -m hoi_flow.eval_tier1 \
    --ckpt hoi_flow/runs/v2_full/ckpt_best.pt --device cuda

# --- Tier-2 (HOT3D benchmark; writes render_and_compare/runs/hot3d_<clip>_flowref/ + scores) ---
CUDA_VISIBLE_DEVICES=1 python compare/hot3d/run_refine_bench.py \
    --ckpt hoi_flow/runs/v2_full/ckpt_best.pt --clips all --n_steps 8 --device cuda:0
```
