# HOI-reconstruction on HOT3D — full experience log & method comparison

> **In plain terms.** This is the full retrospective of the project: building the hand-built
> registration pipeline (`icpjgr`) up through several improvements, then bake-off testing
> learned pose estimators (Any6D, FoundationPose) against it, then a combined approach. The
> conclusion: learned methods that use the calibrated depth data beat the hand-built pipeline on
> placing the object correctly, but the hand-built pipeline is more robust (it never drifts off
> or flips the object's rotation), so the best system combines both. Method codes decoded in
> [GLOSSARY.md](../../../GLOSSARY.md).

A reflective summary of the whole arc: from the first HOI4D sanity checks, through
building and tuning the optimization pipeline, to the learned-method bake-off and
the combined method. Written as a reference for what was tried, what worked, what
didn't, and why.

The north star throughout: **recover an object's 6DoF trajectory + shape from an
egocentric hand-object video**, evaluated against mocap-grade ground truth.

---

## 0. Why HOT3D became the benchmark

Early work was on HOI4D (kettle_N15). Its annotations turned out to be the
bottleneck — ±1.5 cm / ±20 px per-frame scatter, a 6.4 mm GT-vs-depth floor, and a
depth↔RGB wobble — so sub-cm method differences were undecidable. Three separate
convention traps there were caught only by *rendering and eyeballing*, never by
metrics: the pose annotates the CAD **bbox centre** (not origin); the only
`MANO_LEFT.pkl` on the box is **fabricated** (mirrored right hand); and HOT3D ships
two model sets whose canonicals disagree (**pose the eval GLBs, not the display
GLBs**).

HOT3D (Meta, via `bop-benchmark/hot3d`) fixed the evaluation problem: mocap-grade
object + hand poses and scanned CADs. It has **no depth sensor**, so the pipeline
input was synthesized by an adapter (`make_rc_input.py`): rectify the Aria
fisheye624 stream onto a virtual upright pinhole camera, and ray-cast GT depth from
the posed eval GLBs + UmeTrack hand meshes into that camera (16-bit mm PNGs, the
`--depth gt` convention). This gives a *calibrated RGB-D* benchmark with trustworthy
GT — the setting that later proved decisive.

**Frozen 6-clip bench** (one object interaction each, mesh-controlled by reusing
each incumbent's stage 0–3 so SAM-3D redraw noise can't masquerade as signal):

| clip | object | character |
|---|---|---|
| bottle_bbq 002034 | sauce bottle | near-revolution, textured label |
| mug_white 001970 | mug | handle → asymmetric, in-hand |
| vase 002500 | vase | revolution, fast in-hand rotation |
| potato_masher 002349 | masher | thin asymmetric tool |
| spatula_red 001990 | spatula | thin, hard to segment |
| puzzle_toy 001964 | Rubik's cube | 24-fold symmetric, wrong generated stickers |

Metric: mocap-GT eval (`gt_pose_eval_hot3d.py`) — posed-surface **chamfer** (mm,
placement), **centroid** (cm), **rot_traj** (deg, per-frame rotation after the
trajectory-optimal constant alignment; median + p90), plus an alignment-invariant
**canonical-shape ICP** (mm, pure shape quality).

---

## 1. The optimization pipeline campaign (icpj → icpjgr)

A gated, subagent-driven campaign built the pipeline up in stacked arms, each
accepted only if it passed a lexicographic gate (worst-clip chamfer, then mean
rot_traj; ±2 mm / ±5° noise floors) with no clip regressing >20%.

| arm | what it added | headline effect |
|---|---|---|
| **icpj** | baseline: SAM-3D mesh + depth+silhouette joint ICP + grasp closure | worst-clip chamfer 158.8 mm (mug/spatula catastrophic) |
| **icpjs** (T1) | hand-aware segmentation | **mug 60.7→7.0, spatula 158.8→20.5 mm** |
| **icpjp** (T2) | chroma-scored anchor attitude search | **bottle 19.9→9.9 mm**, centroid halved |
| **icpjgr** (T3) | speed-gated grasp-rigidity | mean rot_traj p90 74.9→72.9 (tail) |

Net baseline→best: **worst-clip chamfer 158.8→21.2 mm, mean chamfer 49.1→15.5 mm
(3.2×)**. Rotation *median* stayed ~flat (36°) — floored by the symmetric cube and
constant-attitude errors.

### What each tier taught

- **T1 — hand-aware segmentation (the big win).** Both catastrophic clips were
  stage-1 SAM2 failures on hand-held objects: the mug mask absorbed the forearm, the
  spatula mask leaked onto the table. Root cause the metrics hid: the frozen prompt
  pixel can land *on the occluding hand* (a tilted object's centroid projects onto
  the fingers). Fix: track hands as their own SAM2 objects and subtract them; prompt
  K≤5 candidate clicks vetoed against the hand mask; score each track
  (temporal-IoU − hand-overlap − area-jump − border) and select the best;
  **minimal-intervention** (leave an already-clean mask untouched) and **vanilla
  fallback** (object enclosed by both hands). *Lesson: the upstream mask decides
  everything; segmentation on hand-held objects is the real frontier, not
  registration.*
- **T2 — chroma attitude search (targeted win).** Depth is azimuth-blind on
  symmetric objects, so scoring rotation hypotheses on depth trivially returns
  identity. Scoring on **LAB-chroma** (does the mesh's texture line up with the
  image?) and **spread-gating at 18 LAB** (only act when the texture is actually
  discriminative) fixed the bottle and self-disabled elsewhere — zero regressions.
  *Lesson: azimuth on symmetric-but-textured objects is a colour problem; but it
  needs a mesh whose texture matches reality (fails on the cube's fabricated
  stickers).*
- **T3 — grasp-rigidity (marginal).** During fast in-hand rotation the object
  co-rotates with the wrist, which carries the azimuth signal geometry hides.
  Detection had to be **contact-based** (velocity is swamped by ICP jitter) and the
  rigidity term **speed-gated** (only where the wrist rotates fast — applying it
  everywhere leaked HaMeR wrist noise). Small but real tail improvement. *Lesson: a
  physically-motivated prior only helps if you apply it exactly where the base signal
  is missing.*

*Recurring lesson across the whole campaign: the acceptance gate (worst-chamfer,
median-rot_traj) systematically under-credits real wins — T2's win is in
chamfer/centroid, T3's in the rotation tail. And every load-bearing bug was caught
by looking at rendered output, not by a metric.*

---

## 2. The learned-method bake-off (T4)

Question: can a learned method beat the hand-built pipeline? Investigated 5 cloned
repos + 2 external methods. **"One environment for all" is infeasible** — they span
cu118/cu121/cu130 with conflicting custom CUDA extensions, none pre-built for
Blackwell sm_120. Feasibility split cleanly, and the *decisive variable was whether
the method consumes the calibrated RGB-D*.

### The full comparison — chamfer median (mm), mesh-controlled (same SAM-3D mesh)

| clip | icpjgr | HORT | ForeHOI | FoundationPose (track) | FP (register) | **Any6D** |
|---|---|---|---|---|---|---|
| bottle_bbq | 9.9 | ✗ wrong obj | 392 | 93 drift | **3.2** | **5.2** |
| mug_white | 7.0 | (unscaled) | 1147 | **2.3** | — | **3.4** |
| vase | 17.7 | ✗ wrong obj | 212 | **6.6** | — | **6.4** |
| spatula_red | 21.2 | ✗ wrong obj | 178 | **11.2** | 12.8 | **9.6** |
| potato_masher | 18.8 | ✗ wrong obj | 704 | 598 drift | **8.6** | **12.0** |
| puzzle_toy (cube) | 18.5 | (unscaled) | 391 | 18.9 | — | 21.3 |

### Group A — discard the calibrated depth → lose by 10–160×

- **HORT** (mono single-image, revived `hort5090`): no metric scale (~7–9× off), no
  temporal consistency, and LangSAM grabbed the *wrong* object on 4/6 clips.
- **ForeHOI** (feed-forward video, `forehoi5090`): canonical *shape* is excellent
  (2.7–16 mm shape-ICP) but its wild path self-estimates depth via DepthAnything3
  (off ~2.5× on egocentric HOT3D), so a well-shaped object lands at the wrong 3D
  location. **Shape good, placement bad.**

### Group B — consume the calibrated RGB-D → per-frame pose STRONGER than icpjgr

- **Any6D** (CVPR'25, render-and-compare): given the same mesh + depth + mask,
  **wins chamfer on 5/6** — often 2–3× lower than icpjgr. But per-frame independent →
  **symmetry-flip outliers** (rot_traj p90 153–173°).
- **FoundationPose** (CVPR'24): where its tracker holds (vase, mug) it *decisively*
  beats icpjgr — 3–8× lower chamfer and rotation down to **~3°**. But default `track`
  mode drifts catastrophically on 2/6; `register_each` recovers them (bottle 93→3.2)
  at ~7–8× compute. **Per-frame estimator stronger; default tracker's robustness
  weaker.**

### The verdict that reframed everything

A learned per-frame RGB-D estimator that *consumes* the calibrated depth is **more
accurate than the hand-built depth+silhouette registration** — the methods that lost
(HORT, ForeHOI) lost only because they discarded that depth. What the pipeline
contributes is **robustness through temporal optimization**: it never drifts (unlike
FP track) and never symmetry-flips (unlike Any6D), because it solves one smooth
trajectory constrained across all frames.

---

## 3. The combined method — best of both

Prototype (`combined_refine.py`): take the learned per-frame poses (Any6D) and wrap
them in the pipeline's temporal layer as a post-processor.

- **Symmetry-flip resolution**: discover the mesh's near-symmetry group, then per
  frame pick the symmetry-equivalent rotation temporally closest to its neighbours.
  Fixes the flips *without touching placement* (a symmetry op can't move the surface).
- **Translation jitter smoothing**: data-anchored acceleration smoother
  (`argmin ‖y−x‖² + λ‖D²y‖²`, closed form). Swept: **translation smoothing is a
  universal free win** (jitter p90 66→7 mm at <1 mm chamfer, rotation untouched);
  **rotation smoothing harms** (averages across residual flips: masher chamfer
  12→21, vase rot 8.5→14°) so it's off by default — it needs a flip-aware SO(3)
  formulation.

**Result on the bottle (the target case): beats icpjgr on *every* metric** —
chamfer 5.2 vs 9.9, rot_traj p90 17 vs 73 — by keeping Any6D's accuracy and fixing
its rotation. Generalizes to the isolated-flip failure mode; the sustained-basin
(masher) and non-symmetry (spatula) cases still need per-frame depth-anchored basin
selection.

---

## 4. What I'd build next (in expected-value order)

1. **Swap stage-4's registration core for a learned per-frame estimator**
   (FoundationPose-register / Any6D) and keep icpjgr's temporal layer — the combined
   method, done properly inside the pipeline rather than as a post-processor.
2. **Flip-aware SO(3) rotation smoother** + per-frame depth+silhouette basin
   selection — fixes the masher's sustained wrong-basin, which pure temporal
   smoothing can't.
3. **Hand-aware segmentation is still the highest-leverage upstream fix** for any
   method — every method's ceiling is set by the stage-1 mask on hand-held objects.
4. **Better SAM-3D texture fidelity** (or per-frame texture re-projection) so the
   photometric/attitude terms work on more objects (the cube needs the *real* sticker
   layout, unreachable from generation alone).

## 5. Meta-lessons

- **Calibration is a moat.** On calibrated RGB-D, the method that uses the
  calibration wins; monocular learned methods are strong at *shape* but can't place.
- **Accuracy and robustness are different axes.** Learned per-frame estimators win
  accuracy; temporal optimization wins robustness. The product is the combination.
- **Mesh-control your comparisons.** SAM-3D generation is nondeterministic; reusing
  the incumbent's stage 0–3 was the only way to attribute differences to the method
  and not to mesh luck.
- **Render and eyeball.** Every convention trap and load-bearing bug this project hit
  — HOI4D bbox-centre, fabricated MANO_LEFT, display-vs-eval GLBs, prompt-on-hand,
  mask leaks, the cube's fabricated texture — was caught by looking at output, not by
  a metric.
- **The gate metric is not the goal.** Worst-chamfer/median-rot_traj under-credited
  T2 (chamfer), T3 (tail), and the whole Any6D result (placement). Keep the raw
  per-clip numbers, not just the gate verdict.
