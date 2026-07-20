# Coarse-error calibration for the flow-matching HOI refiner

Measured per-frame error of the **shipped coarse stack** against HOT3D mocap GT on the 6
benchmark clips, so the refiner can train on calibrated corruptions of clean GT instead of
guessed noise. Produced by [`measure_calibration.py`](measure_calibration.py) →
[`calibration.json`](calibration.json); consumed by [`corrupt.py`](corrupt.py).
All numbers are in the **pinhole camera frame**: translation **mm**, rotation **deg**.

## TL;DR (pooled over the 5 `fpauto` clips — the shipped object arm)

| quantity | along-ray (depth) | perpendicular (image plane) |
|---|---|---|
| **Object** trans bias / sigma | −1.2 / **21.5 mm** | ~0 / **13.5 mm** |
| **Object** trans AR(1) rho | 0.71 | 0.78 |
| **Hand** trans bias / sigma | −12.8 / **59.5 mm** | ~2 / **13.1 mm** |
| **Hand** trans AR(1) rho | 0.95 | 0.93 |

- **Object rotation (smooth):** bias 11.8°, sigma 11.8°, rho 0.79. **Flips:** onset rate
  **0.036 /frame**, ~15 % of frames flipped, magnitude ~103°, mean duration 4.3 frames (max 23).
- **Hand articulation residual** (after removing the rigid offset): **6.8 mm** median.
- **MANO GT reliability:** articulation **USABLE** (8.0 mm rigid-aligned), but the stored
  global wrist pose is in a **different camera convention** — see the loud note below.

Depth (along-ray) error dominates both object and hand — exactly the monocular
under-constraint the refiner exists to fix. The hand's perpendicular error is tiny because
the shipped hand-reprojection optimizer pins the 2D image plane (2–4 px); its depth is free.

## How it was measured

**Object — canonical alignment first.** Est poses pose a SAM-3D mesh; GT poses pose the
HOT3D eval GLB — *different canonical frames*, so pose matrices are not directly comparable.
We solve one constant `A` (4×4) per clip, `A = argmin_A Σ_t ‖T_est,t·A − T_gt,t‖_F`
(rotation via SVD of `Σ R_est^T R_gt`, translation closed-form), **robustified** by iterating
and dropping flip frames (>60°) before refitting. Sanity: median translation after alignment
is 5.8–27 mm on the `fpauto` clips (matches the leaderboard chamfer scale), confirming `A` is
sane. The per-frame residual `E_t = T_gt,t⁻¹ · (T_est,t·A)` gives the error we characterise.

**Decomposition.** The camera-frame translation error `e_t` is split into a scalar **along
the view-ray** (`e_t·r̂_t`, `r̂_t` = unit direction to the object) and a **2-D perpendicular**
part in a fixed basis from the mean view direction. Rotation error is the geodesic angle of
`E_t`.

**Temporal model — AR(1)+white.** Each scalar series is fit as an AR(1) drift plus
independent white jitter, identified from the lag-1/lag-2 autocorrelation
(`rho_ar = r₂/r₁`, `var_ar/var = r₁²/r₂`; then `sigma_ar = sigma·√f`,
`sigma_white = sigma·√(1−f)`). These three knobs regenerate the observed sigma, rho **and**
frame-to-frame jitter.

**Flips.** Frames with geodesic error >60° are flip frames; contiguous runs are flip events.
We record onset rate, flipped-frame fraction, magnitude, and the empirical duration list
(resampled at corruption time).

**Hand.** Est hand is MANO (778 v); GT is UmeTrack (different topology) → correspondence-free.
Per frame we match the nearest GT hand by centroid, take the **translation offset**
(est − GT centroid) as the placement error (decomposed along/perp, AR-fit), and the
**post-offset one-sided chamfer** as the articulation residual.

## Object error, per clip

| clip | arm | align med (mm) | along bias/sigma/rho | perp sigma | rot bias/sigma | flip rate / state |
|---|---|---|---|---|---|---|
| bottle_bbq | fpauto | 5.8 | −0.7 / 10.4 / 0.21 | 9.3 | 12.4 / 11.1 | 0.140 / 0.45 |
| mug_white | fpauto | 8.1 | −1.4 / 3.9 / 0.95 | 5.3 | 14.6 / 12.3 | 0.007 / 0.01 |
| vase | fpauto | 20.1 | 5.0 / 24.1 / 0.89 | 13.3 | 10.8 / 12.9 | 0.013 / 0.09 |
| spatula_red | fpauto | 27.1 | −0.7 / 32.0 / 0.57 | 15.0 | 6.1 / 3.6 | 0.000 / 0.00 |
| puzzle_toy | fpauto | 24.7 | −8.0 / 22.0 / 0.91 | 19.6 | 16.5 / 13.9 | 0.020 / 0.21 |
| potato_masher | **icpjgr** | 103.9 | −45.5 / 97.4 / 0.99 | 25.1 | 37.1 / 12.3 | 0.013 / 0.38 |
| **pooled** (5 fpauto) | — | — | **−1.2 / 21.5 / 0.71** | **13.5** | **11.8 / 11.8** | **0.036 / 0.15** |
| pooled_all (6) | — | — | −8.5 / 47.3 / 0.75 | 16.1 | 15.1 / 14.6 | 0.032 / 0.19 |

`potato_masher` uses the **icpjgr** (registration) arm, not `fpauto`, and is a spinning
symmetric object: 104 mm align error, 38 % flipped, rho 0.99. It is a genuine outlier and is
**excluded from the pooled default** (kept as `pooled_all` and per-clip). Use its per-clip
profile only for masher-like symmetric objects.

## Hand error, per clip

| clip | along bias/sigma/rho | perp sigma | articulation resid (mm) |
|---|---|---|---|
| bottle_bbq | −15.6 / 13.4 / 0.89 | 6.2 | 7.0 |
| mug_white | −75.7 / 60.4 / 0.96 | 13.0 | 6.6 |
| vase | 32.1 / 58.0 / 0.99 | 17.2 | 7.5 |
| spatula_red | 8.0 / 23.0 / 0.96 | 8.6 | 6.0 |
| potato_masher | −104.9 / 75.9 / 0.99 | 16.6 | 7.2 |
| **pooled** | **−12.8 / 59.5 / 0.95** | **13.1** | **6.8** |

`puzzle_toy` is absent: the hand optimizer flagged **0 / 150 visible frames** (grasping hand
off-frame/occluded), so it contributes no hand stats. Hand **depth** error is large and its
**sign flips per clip** (est can be nearer or farther than GT); the pooled bias hides this, so
prefer the per-clip profile when clip identity is known. Perpendicular error is uniformly small
(6–17 mm) — the 2D-reprojection constraint. `rho ≈ 0.95–0.99` ⇒ depth error is a slow drift,
near a random walk.

## MANO GT reliability — READ THIS

We FK'd HOT3D's stored per-frame MANO (right hand, betas = 0) and compared to the UmeTrack GT
landmarks over 20 training segments:

| alignment removed | median landmark error |
|---|---|
| none (raw) | **1030 mm** |
| per-frame translation | 61 mm |
| per-frame rigid (Kabsch) | **8.0 mm** (p25/p75 7.5 / 8.3) |

**Verdict:** the MANO **articulation (thetas) is reliable** — 8 mm after rigid alignment is at
the MANO-vs-UmeTrack cross-topology floor, well under the 10 mm bar, so thetas are a usable
flow target. **BUT the stored global wrist pose is NOT usable as-is:** it sits in a different
camera convention (~1 m raw offset; z-sign flipped). The residual gap is a **near-constant
rotation** (per-frame Kabsch rotations cluster to within 3.8° of their mean inside a segment),
i.e. a fixed frame convention, not noise — so it is recoverable by a fixed transform, not a
per-frame refit. This matches `extract_gt_hands.py`'s "MANO thetas unreliable" warning: it is
the **global placement** that is unreliable as-stored, not the finger pose.

**Action for the refiner:** you may use HOT3D MANO **thetas** directly as the hand-articulation
target. You must **re-derive the global wrist_xform into the working camera frame** before using
it (a per-segment MANO-fit-to-UmeTrack global-pose pass, or solving the fixed convention
transform once) — do **not** feed the raw `hand_mano` wrist_xform against camera-frame targets.

## `corrupt.py` — API and knobs

```python
from hoi_flow.data.corrupt import load_stats, object_profile, hand_profile, \
                                   corrupt_object, corrupt_hand
stats = load_stats()                                   # reads calibration.json
op    = object_profile(stats, "pooled")                # or a clip name / "pooled_all"
rng   = np.random.default_rng(0)
coarse = corrupt_object(poses_gt, op, rng, view_dirs=None)         # [T,4,4] -> [T,4,4]
coarse_hand = corrupt_hand(joints, hand_profile(stats), rng)        # [T,J,3] -> [T,J,3]
coarse_mano = corrupt_hand({"wrist_xform": w6, "theta": t15},       # MANO params path
                           hand_profile(stats), rng)
```

- **`corrupt_object(poses_gt, stats, rng, view_dirs=None, overrides=None)`** — AR(1)+white
  translation error split along-ray / perpendicular, a smooth AR rotation wobble, and Bernoulli
  **flip events** (rate, resampled duration, ~180° about a random axis). Rotation is applied about
  the object centre so orientation and translation errors are independent.
- **`corrupt_hand(hand, stats, rng, view_dirs=None, overrides=None, theta_sigma=0.05,
  wrist_rot_deg=4.0)`** — dict input `{wrist_xform[T,6], theta[T,K]}` corrupts the wrist
  translation (along-ray AR offset), wrist rotation (small AR wobble, `wrist_rot_deg`), and each
  theta (`theta_sigma`, white); ndarray input `[T,J,3]` applies the same along/perp offset as a
  rigid joint shift. No flips for hands.
- **Every knob is overridable** via `overrides` (deep-merged): e.g.
  `object_profile(stats, "pooled", {"flip": {"event_rate": 0.0}})` to disable flips, or
  `{"trans_along": {"sigma_ar": 30}}` to widen depth error. `theta_sigma` / `wrist_rot_deg` are
  **not calibrated** (HOT3D thetas are reliable; we have no measured coarse-theta error) — defaults
  are conservative; tune to taste.

**Self-test** (`python -m hoi_flow.data.corrupt`) corrupts a real GT segment 40× and reports the
regenerated stats against the calibration targets. They match to within noise, except **hand
along-ray sigma realizes ~15 % low** (49.5 vs 59.5 mm): with `rho ≈ 0.95` over 150-frame clips the
AR(1) does not fully explore its stationary distribution — a finite-length effect that *matches
what real coarse tracks of that length actually look like*, so it is left as-is.

## Caveats

- **Per-clip variance is large** (object along-sigma 3.9→32 mm; hand depth bias flips sign). The
  pooled default is a reasonable middle for clip-agnostic training; pass a per-clip profile when
  the clip is known.
- **`potato_masher` is a different arm (icpjgr) and a symmetric spinner** — excluded from the
  pooled default; use `pooled_all` or its per-clip profile only for masher-like objects.
- Alignment `A` and the along/perp split assume the object stays roughly centred (view-ray drifts
  slowly); true for these clips.
- Stats use betas = 0 for MANO FK and the shipped hand-reprojection optimizer output as "est".

## Real-pairs realism check (2026-07-15, first 94 windows of the P0002+P0015 batch)

Real coarse quality is **bimodal**: 44% of windows land ≤30 mm object error (matching this
file's calibrated stats — the regime the corruption model reproduces), but 54% are
catastrophic (100–700 mm): in cluttered pantry/kitchen scenes with fast large pick-and-place
motion (GT excursions 0.3–1 m), FoundationPose registers onto a **wrong nearby instance** or
loses the track and stays put (verified visually: coarse static on a shelf twin while GT moves
with the hand). Utensils/dishes in open scenes are fine (bowl 7 mm, spatula 5 mm, flask 5 mm).
Consequences: (1) Tier-1 real evaluation must stratify by coarse quality — the refiner's design
domain is the good mode; (2) the catastrophic mode is an upstream re-detection problem, NOT a
refinement problem (the bridge formulation is deliberately anchored to the source); (3) if
re-localization from far-off coarse is ever wanted, that argues for the noise→clean conditional
variant + a catastrophic corruption mode — a v2 design fork, not a v1 patch.
