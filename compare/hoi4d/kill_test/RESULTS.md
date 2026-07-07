# b6 kill test — RESULTS (2026-07-03)

## Verdict: **KILL** (pre-registered criterion failed decisively)

> **Pre-registered criterion** (from the idea-loop b6 card): *fit per-frame affine
> (a_t, b_t) from MoGe depth to HOI4D GT depth on background pixels only; **GO** if
> it cuts hand/object-pixel depth error ≥50% across ≥10 clips — else kill.*

**Result: 0/12 clips pass. Median error reduction = 5.6% (mean 7.6%) vs the 50% bar.**
Three clips are *negative* (the background-fitted per-frame gauge is worse than a
single clip-global fit). The stronger variants also fail: near-field
(support-surface) gauge 16.2% median, disparity-space affine 8.1%, combined 17.1%.

## What the test was

The b6 branch ("GS-based bundle adjustment of camera + metric scale + background,
feeding unchanged HOI pipelines") rests on one load-bearing bet: *per-frame
monocular depth-gauge wobble dominates hand/object depth error, and the static
background alone pins the gauge*. A background-only GS-BA can only ever recover
what the background determines — this test measures that ceiling directly with GT,
no GS system needed.

Protocol per clip (`kill_test.py`, MoGe = `Ruicheng/moge-vitl` at half res with true
fov_x; Huber-IRLS affine fits; details in the script docstring):

- **bg pixels** = valid GT ∧ not-dilated(hand ∪ object) ∧ temporally-static by GT
- **fg pixels** = (hand ∪ object masks, eroded) ∧ valid GT
- gauges: **global** (one (a,b)/clip, bg), **per-frame bg** (the b6 mechanism),
  **near-field bg** (bg at the fg's own depth range ±15 cm), **disparity-space**
  variants, **oracle fg** (per-frame fit on fg itself = upper bound)
- metric: fg MAE reduction vs the global gauge, per clip

Machinery validated before use: synthetic-gauge selftest (recovery corr 1.0000,
R=0.91 at the injected noise floor). Masks: LangSAM + track-based manipulated-object
selection (sustained hand-contact run; validated IoU 0.962 vs reference masks on
kettle_N15, all 12 clips visually verified — see `make_masks.py`).

Data: 12 HOI4D clips with GT aligned depth — 5 kettle, 3 bottle, 2 mug, 2 bowl,
all distinct object instances, 4 cameras (`/workspace/hoi4d/clips/`,
`clips_manifest.json`).

## Per-clip results

| clip | fg MAE (global) | per-frame bg | near-field | disparity | **oracle fg** | corr(a_bg, a_fg) | wiggle g→o |
|---|---|---|---|---|---|---|---|
| bottle_N29 | 6.6 cm | +0.3% | −0.7% | −5.0% | 52.2% | 0.49 | 2.1 → 0.26 cm |
| bottle_N38 | 7.4 cm | −19.8% | −20.5% | −20.0% | 25.4% | 0.22 | 4.5 → 0.85 cm |
| bottle_N41 | 7.3 cm | −17.5% | −16.3% | −19.6% | 38.6% | 0.25 | 4.9 → 0.72 cm |
| bowl_N12 | 11.1 cm | +24.0% | +29.0% | +25.0% | 84.7% | 0.48 | 7.8 → 0.64 cm |
| bowl_N30 | 7.2 cm | +9.7% | +20.4% | +9.6% | 81.3% | 0.27 | 5.2 → 0.07 cm |
| kettle_N11 | 5.0 cm | +1.1% | +10.8% | +0.8% | 25.7% | 0.13 | 2.9 → 0.45 cm |
| kettle_N15 | 5.5 cm | +2.7% | +19.2% | +25.2% | 52.5% | 0.10 | 4.3 → 0.45 cm |
| kettle_N22 | 5.2 cm | +14.8% | +13.3% | +6.6% | 39.3% | 0.16 | 3.5 → 0.31 cm |
| kettle_N40 | 5.2 cm | +36.1% | +36.4% | +36.0% | 52.1% | 0.32 | 4.6 → 0.12 cm |
| kettle_N48 | 4.3 cm | +28.8% | +29.9% | +28.1% | 54.0% | 0.62 | 3.5 → 0.41 cm |
| mug_N27 | 5.2 cm | +3.9% | +4.5% | −1.1% | 65.3% | **−0.61** | 3.0 → 0.04 cm |
| mug_N44 | 3.2 cm | +7.3% | +22.6% | +26.5% | 42.9% | 0.76 | 2.0 → 0.02 cm |

Figure: `killtest_summary.png`. Raw per-frame data: `<clip>/kill_test/result.json`
(+ per-clip gauge/error traces in `<clip>/kill_test/figure.png`).

## The two scientific findings

1. **The b6 transfer premise is false.** MoGe's per-frame error is NOT a global
   affine gauge: the background's per-frame gauge barely correlates with the
   foreground's (median corr(a_bg, a_fg) = 0.26; one clip *anti*-correlates at
   −0.61). The depth net's error is spatially structured — near-field hand/object
   pixels wobble in their own mode that far-field (and even same-depth nearby)
   background cannot predict. Consequence: **any background-only substrate — GS-BA,
   ViPE, MegaSaM alike — cannot fix hand/object depth in this regime.** This also
   explains the earlier eval observation that swapping depth nets (MoGe→DA3-metric)
   changed nothing.

2. **The jitter IS per-frame gauge error — but it lives in the foreground.** A
   per-frame affine correction fitted on the fg itself (oracle) cuts fg MAE 51% on
   average and collapses temporal wiggle **4.0 cm → 0.36 cm (11×)** — with just 2
   parameters per frame. The signal that HOI methods are missing is a per-frame,
   foreground-local scale/shift. Whatever estimates those 2 parameters per frame
   without GT solves most of the jitter.

## Implication — the pivot (pre-planned: b5)

Foreground-anchored gauge estimation, not background reconstruction. Candidate
anchors that live where the gauge lives:
- **object rigidity**: the manipulated object's metric size is constant → per-frame
  scale is observable from apparent size (the same signal our `--gt-anchor` and
  temporal-smoothing hacks crudely exploited);
- **hand shape prior**: MANO hand size is constant (HaWoR's own canonical hand);
- **contact events**: while the object rests on the support surface, its depth =
  the (locally reliable) surface depth → absolute anchor at rest frames;
- optionally a GS/photometric refinement layer over these anchors — i.e. the b5
  "method-agnostic temporal refiner", now with a causally-validated motivation and
  a measured 11× stability headroom as its target.

The negative result is publishable ammunition: it kills the "just ground HOI in a
scene reconstruction" intuition (TRAM-style scale transfer does not extend from
bodies to near-field manipulation) with an oracle-controlled decomposition.
