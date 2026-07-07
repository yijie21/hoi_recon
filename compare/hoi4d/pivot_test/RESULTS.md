# b5-pivot kill test — RESULTS (2026-07-04)

## Verdict: **KILL** for GT-free *geometric* foreground anchors (both pre-registered primaries fail all four criteria)

> **Question under test** (pivot pre-planned in `../kill_test/RESULTS.md`): the b6 kill
> test proved the per-frame depth gauge that drives HOI foreground error/jitter lives in
> the foreground and is worth 51% MAE / 11x wiggle if recovered. **Can it be estimated
> WITHOUT GT from foreground-intrinsic geometric anchors** (object rigidity via visible
> extent, chained optical-flow relative scale, hand size, fusions)? Same 12 HOI4D clips,
> same baseline (clip-global gauge) and eval protocol as b6; anchors see GT only through
> the clip-global gauge. Machinery: `pivot_test.py` (self-tested), `aggregate.py`.

**Pre-registration trail** (full detail in the `pivot_test.py` docstring):
v1 design -> synthetic selftest only -> external design review (Opus) -> v2 frozen
(kappa-oracle ceiling, spectrum precheck, DC-decoupling, smooth control, pooled
aggregation + bootstrap, primary = eb_s) -> declared pilot clip kettle_N22 revealed
eb_s structurally degenerate (rig anchor anti-correlated under occlusion -> EB shrinks
to baseline) -> amended primary flow_s selected on the pilot ONLY, verdict computed on
the 11 held-out clips. Both verdicts reported; both are KILL.

| criterion (amended primary flow_s, 11 holdout clips) | result | bar |
|---|---|---|
| (a) pooled headroom ratio vs 1-DOF kappa-oracle | **-1.20** (95% CI [-2.55, -0.41]); median per-clip -0.23 | >= 0.5 |
| (b) median object wiggle reduction | **0.59x** (i.e. it *adds* jitter); ko oracle achieves 8.6x | >= 2.5x |
| (c) no clip degraded > 5% MAE (obj & union) | **violated on 6/11 clips** (worst: kettle_N48 4.3 -> 15.8 cm) | all |
| (d) beats the evidence-free smoothing control | **-42.9 cm pooled vs +0.2 cm** | > |

Registered primary eb_s on all 12: pooled ratio -0.55, median wiggle 1.00x (it mostly
does nothing by design), (c) violated on the 3 bottle clips. KILL.

## The three scientific findings

1. **The foreground gauge target is sharper than b6 stated: it is one parameter per
   frame, and it is slow.** The new 1-DOF kappa-oracle (best per-frame pure scale on
   object pixels) captures essentially the whole 2-DOF oracle (pooled 1.00 vs 1.04;
   shift-only 0.99 — scale and shift are degenerate at object distance, as predicted by
   the design review). Its trace is 93–100% low-frequency (med-5 band) on every clip.
   Median achievable wiggle reduction: 8.3x (1-DOF), 22x (2-DOF). So a temporally
   coherent, 1-DOF-per-frame estimator is not structurally excluded — the signal is
   there and it is smooth.

2. **Mask-geometry statistics cannot harvest it.** Two independent estimator families,
   each with the failure mode the diagnostics predicted:
   - *Visible-extent rigidity* (rig-o/-h/-u): the object's TRUE visible extent (same
     statistic on GT depth, same pixels) swings cv = 1–21% with grasp occlusion —
     larger than the few-% gauge signal on most clips. The anchor consequently
     *anti-correlates* with the true gauge on typical grasp clips (kettle_N22:
     corr −0.35; see figure.png middle panel: GT extent collapses 8.2 -> 6.2 cm at
     grasp onset while the gauge barely moves). Pooled headroom −1.15.
   - *Chained flow scale* (flow/-s): on textured objects it tracks the gauge SHAPE
     well (corr_lp 0.73–0.99 on kettles/bowl_N30) but the chain accumulates
     multiplicative bias — kettle_N48 drifts at −0.54 log/100fr vs the true −0.10,
     a ~2x amplitude error by clip end; on texture-poor bottles it is unusable.
     Pooled headroom −1.20.
   - *EB fusion safety* worked when the anchors disagree (shrinks to exactly the
     baseline, R = 0.000) but the bottles break its independence assumption: both
     anchors derive from the same mask/depth pixels, share the same wrong mode, agree,
     and the fusion follows them off the cliff. A GT-free agreement test cannot detect
     a *shared* systematic error.
   - The evidence-free *smoothing control* gains ~nothing (pooled +0.006), confirming
     (again, now on 12 clips) that the error is not a filtering problem.

3. **Where the gauge IS visible from images, it is visible through correspondence —
   which points at rendering, not masks.** Flow's high shape-correlation on textured
   objects is positive evidence that pixel correspondence carries the gauge; its
   failure is integration drift, an artifact of *chaining relative* measurements.
   A render-and-compare estimator against a FIXED canonical object model (the original
   b5 mechanism: freeze appearance, optimize per-frame pose+scale photometrically)
   makes every frame an *absolute* measurement against the same model — no chaining,
   no mask-extent dependence, sub-pixel evidence. None of that is testable in a
   CPU-day; it is exactly the part of b5 this test could not cheaply probe.

## Per-clip detail

See `aggregate.json` (both verdicts, per-clip rows, bootstrap CI) and per-clip
`<clip>/pivot_test/{result.json, figure.png}` (kappa traces vs the ko oracle, extent
pollution panel, MAE bars). Batch logs: `<clip>/pivot_test_log.txt`.

## Implication — what survives for the "GS helps HOI" program

Dead so far, all with oracle-controlled evidence: background-anchored substrates (b6),
output filtering (b6 two-band analysis + smooth control here), commodity depth-engine
swaps (b6 corollary), and now cheap geometric foreground anchors (this test).
Alive, with a sharpened, quantified target (per-frame 1-DOF foreground gauge; 51% MAE
/ 8–22x wiggle headroom; low-frequency):
- **photometric render-and-compare refinement against a frozen canonical object
  (GS or mesh) — the b5 core**, now motivated by three controlled negative results;
- model-based absolute anchors as priors inside it: MANO hand size (HaWoR canonical
  hand), object-at-rest support contact;
- a learned per-frame gauge estimator (HOI4D GT provides supervision) as a
  feed-forward alternative.
Practical note: any GPU instantiation needs Blackwell-compatible torch (cu128+) on
this box; the current `daid`/`hort` envs fail on the RTX 5090s.
