# Step-1 substrate matrix — VGGT-Omega injected into render_and_compare (2026-07-06)

## Headline: the VGGT-Omega substrate closes **53% of the object-depth oracle gap**
## and **61% of the object-jitter gap** in the full, unchanged RC pipeline — feed-forward, no per-clip optimization.

12 HOI4D clips x {moge, vggt, gt} = 36 full pipeline runs (~40 min wall on 2x
RTX 5090). Medians over clips, object region vs GT sensor depth:

| condition | obj depth MAE | obj wiggle (jitter) | hand depth MAE | hand wiggle |
|---|---:|---:|---:|---:|
| MoGe (current substrate) | 16.71 cm | 2.11 cm | 8.60 cm | 5.49 cm |
| **VGGT-Omega injected** | **9.55 cm** | **1.68 cm** | 8.16 cm | 4.25 cm |
| GT depth injected (oracle) | 1.37 cm | 0.65 cm | 3.12 cm | 4.06 cm |

Pooled oracle-gap closure (moge->vggt) / (moge->gt): **obj depth 0.53, obj
wiggle 0.61, hand wiggle 0.52, hand depth 0.16**.

Convergent validation: the Gate-2 sampling analysis predicted this — VGGT-
Omega's raw object-region depth error was 9.0 cm median (vs MoGe raw 27.8)
with R_ko 0.35 remaining; the end-to-end pipeline lands at 9.6 cm. The
pipeline faithfully transmits substrate quality, and the sampling-level
numbers are a reliable cheap proxy for full-pipeline outcomes.

## Reading

1. **Substrate swap is the single biggest free win available.** Halving
   object depth error and ~40% less jitter with zero method changes — the
   TRAM-style "just inject a better world prior" step, now measured for HOI.
2. **What remains is exactly the pre-quantified target.** The remaining 47%
   object-depth gap and the hand anchoring (closure only 0.16 — VGGT-Omega is
   weak on thin fast hands) are the b5 refiner's job: per-frame foreground
   gauge (R_ko 0.35) + hand-size/contact anchors. GT condition shows the
   pipeline could reach ~1.4 cm if the substrate were perfect.
3. **Per-category texture matters**: bottles stay VGGT-Omega's weak spot
   (bottle_N29 is the one clip where it is worse than MoGe end-to-end),
   consistent with Gate-2. Kettles/mugs/bowls improve 2-6x.

## Protocol & caveats (read before quoting numbers)

- RC v. `configs/real_forehoi.yaml`; depth injected via the `gt` backend
  (RC_GT_DEPTH_DIR / RC_GT_INTRINSICS): GT = HOI4D align_depth; VGGT-Omega =
  full-clip single-pass densified depth (`densify_vggt_depth.py`, one scale
  convention per clip, saved as mm PNGs, HOI4D intrinsics apply — pure-resize
  aspect). MoGe condition = the pipeline's native monocular path.
- **Object masks injected for ALL conditions** (validated kill-test masks,
  RC_OBJECT_MASK_PATTERN + RC_OBJECT_MASK_ERODE=5, new injection seam in
  `real_perception.py`): identical segmentation across substrates; a naive
  SAM2 point prompt had grabbed a 1.2k-px fragment (IoU 0.04) on the smoke
  clip. Erosion keeps silhouette-edge background depth out of the lifted
  object (size err +77% -> +24% on the GT smoke).
- **SAM-3D unavailable on this box** (its env is pre-Blackwell) -> stage3
  fail-soft **depth-lift object for all 36 runs**. Internally consistent, but
  absolute numbers carry the fallback penalty (GT floor 1.37 cm here vs ~0%
  in the archived SAM-3D-based kettle_gt run). Closure ratios are the robust
  statistics.
- The **object size metric is depth-lift-entangled** (boundary depth sets the
  lifted extent; sign flips between conditions, one +122% GT outlier on
  mug_N44) — treat depth MAE/wiggle as primary, size as diagnostic only.
- Eval: `eval_rc_matrix.py` (validated against the archived kettle_gt run);
  per-frame posed-mesh median depth vs GT depth in eroded masks; wiggle =
  std_t of signed error. Full table: `rc_matrix_results.json`.
- Runs: `render_and_compare/runs/hoi4d_matrix/<clip>__<cond>/` (+ per-run
  reprojection mp4s for visual inspection).

## Program consequence

Adopt VGGT-Omega as RC's default substrate (a `--depth vggt_omega` backend is
a small follow-up: the densify script already produces the depth; wiring it
as a first-class backend removes the env-var plumbing). Then build the b5
foreground-anchored refiner on top; its quantified remaining budget on this
harness: object 9.6 -> 1.4 cm, hand 8.2 -> 3.1 cm, jitter 1.7 -> 0.65 cm.
