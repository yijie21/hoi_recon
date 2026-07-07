# Gate-2 — feed-forward 4D models as the depth substrate (2026-07-04)

## Verdict (pre-registered criterion, medians over 12 HOI4D clips, object region)

| source | MAE (global-gauged) | temporal wiggle | R_ko (per-frame gauge left) | raw-metric MAE | verdict |
|---|---:|---:|---:|---:|---|
| MoGe (b6-era baseline) | 4.90 cm | 3.47 cm | 0.465 | 27.8 cm | baseline |
| OpenD4RT 48CLIP | 10.67 cm | 1.47 cm | 0.675 | 144.8 cm | **FAIL — gauge persists, accuracy regresses** |
| VGGT-Omega 1B-512 | **4.18 cm** | **1.42 cm** | 0.350 | **9.0 cm** | **PARTIAL — halves the wiggle, does not solve the gauge** |

Criterion (frozen in `gate2_eval.py` before any model output was seen):
SOLVE = wiggle <= 1.0 cm AND R_ko <= 0.15 AND MAE <= MoGe's. PARTIAL = wiggle
<= 0.5x MoGe AND MAE <= MoGe. R_ko = fraction of object MAE a per-frame 1-DOF
scale fit (the kill tests' kappa-oracle) still removes — the "remaining
per-frame gauge error" readout.

## Protocol

48 evenly-spaced frames per clip; identical pixel populations for all sources
(eroded object/hand masks, static background — kill-test rules, full-res
sampling); per-source clip-global Huber affine on background normalises scale
conventions (same treatment MoGe always got). D4RT queried through its own
WorldTrack eval convention (256x256 video, uv normalised by original dims,
t_src=t_tgt=t_cam diagonal = per-frame depth in each frame's own camera);
VGGT-Omega through `load_and_preprocess_images` (balanced 512, pure resize at
this aspect). Extraction: `gate2_extract.py`; metrics: `gate2_eval.py`;
per-clip data: `<clip>/gate2/{moge,d4rt,vggt}_samples.npz`, full table in
`gate2_results.json`. Env: `gate2` conda env, torch 2.11+cu128 on RTX 5090.

## Findings

1. **The 2024-25-era conclusion "swapping depth backbones changes nothing"
   does NOT extend to the newest generation.** VGGT-Omega materially improves
   the foreground: 2.4x lower temporal wiggle at slightly better accuracy, and
   its RAW output is near-metric (9.0 cm median without any gauge fit — vs
   MoGe 27.8, and it beats gauged MoGe outright on several clips: bowl_N12
   2.9 vs 12.8, bowl_N30 3.4 vs 8.2 — the two clips that were MoGe's worst).
   The user-hypothesised "modern 4D models handle dynamic objects" is
   partially vindicated, and Gate-2's designed-in comparison did its job.

2. **But the per-frame foreground gauge problem survives.** VGGT-Omega's
   median R_ko = 0.35: a 1-DOF per-frame scale on the object still removes a
   third of its depth error; its wiggle floor (1.42 cm) sits ~3x above the
   kappa-oracle residual (~0.45 cm). Per-clip it is also uneven — on
   bottle_N38/N41 it is *worse* than MoGe (12.8/9.8 vs 8.2/6.9). Its error
   mode is largely uncorrelated with MoGe's (corr of kappa traces spans
   -0.53..0.87) — a different net, a different wobble, the same structural
   gap: monocular dynamic-foreground depth is a prior, not a measurement.

3. **OpenD4RT is not usable as a metric substrate** (median gauged MAE 10.7 cm,
   2x worse than MoGe; R_ko 0.675; raw scale ~1.5 m off). It is temporally
   smooth (wiggle 1.47) — smoothly wrong, the exact "consistency without
   correctness" failure mode. Honest caveats: this is the *unofficial*
   OpenD4RT reproduction (48-frame checkpoint, WorldTrack-tuned), its strength
   is tracking rather than dense metric depth, and two clips (kettle_N15,
   kettle_N40) show it CAN nail the gauge (R_ko 0.03-0.06) — the capability
   exists in the architecture but is wildly clip-inconsistent. Weak evidence
   about the true (unreleased) D4RT.

## Program update

- **Substrate**: VGGT-Omega replaces MoGe as the default depth prior going
  forward (better accuracy, 2.4x stability, near-metric scale — also
  simplifies the metric-anchoring story).
- **The b5 photometric-refiner target stands**, now on a better substrate with
  a smaller-but-real quantified headroom: R_ko 0.35 / wiggle 1.42 -> ~0.45 cm
  on VGGT-Omega outputs. The elimination chain now covers: background
  substrates (b6), filtering, mask-geometry anchors (b5-pivot), previous-gen
  depth swaps, AND the newest feed-forward 4D generation (this gate) — none
  solve it; one mechanism remains, and it has a stronger starting point.
- Rebuttal-grade control acquired: any reviewer's "did you try the latest 4D
  model?" is now answered with data.
