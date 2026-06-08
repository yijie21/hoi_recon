# Research Directions — picking up the HOI reconstruction work

A pick-up document. It captures (a) where the pipeline stands, (b) the limitations we
**measured** (not guessed), and (c) ranked, concrete directions for a follow-up paper.
The repo implements the **CHOIR** paradigm (arXiv:2605.20992): *coarse contact-agnostic
init → generative spatial rectification → contact-aware joint optimization*. Everything
below is about where that recipe breaks and what a strong successor would do.

> One-line thesis for a follow-up: **CHOIR closes the loop to a *grasp prior* — not to
> the *video*, and not to *time*. The next paper closes both.** Both gaps are measured
> below.

---

## 1. Where the pipeline stands (validated on `runs/grab`, 192 frames, left hand)

| component | method | state | measured quality |
|---|---|---|---|
| object 6D | render-and-compare (silhouette + photometric), occlusion-aware don't-care IoU, stage-7 rotation anchor | **good** | dc-IoU **0.952**, mask_cov 0.979, centroid ~7px (low-occ) |
| object temporal | velocity + acceleration smoothing; stage-7 object rotation prior | **improved** | reproj-accel median **4.52px** / p90 9.87px; transl-accel 2.27mm |
| hand | HaMeR→MANO, depth-anchored, chirality-corrected, kp2d + hand-silhouette + accel losses | **weak link** | precision **0.86**, IoU 0.71, centroid **35px** (median) |
| contact/grasp | joint differentiable MANO-articulation + object optim (contact, non-penetration) | working | gap median 0.4mm |

Run: `python -m hoi_recon.cli --video examples/grab.mp4 --out runs/grab --real --hand hamer --object sam3d --depth moge --config configs/new.yaml`
Diagnostics: `python scripts/object_confidence.py --run runs/grab` (confidence + jitter metrics, CSV + plot + video).

---

## 2. The limitations we MEASURED (this is the paper's motivation section)

These are diagnosed, with numbers and file pointers — they are the evidence a follow-up
should lead with.

1. **The hand is not registered to the video** — *the dominant gap.*
   The object is image-fit (silhouette+photometric); the hand had no image-space loss.
   Adding kp2d reprojection + hand-silhouette helped (precision 0.78→0.86, centroid
   44→35px, worst-frame tail 17→9px) but a **floor remains**: the metric-depth-anchored
   hand projects **~63px** from HaMeR's own keypoints at init (re-anchoring to MoGe depth
   places it at a z where the fixed-shape hand projects at the wrong 2D scale), so the
   median kp2d residual bottoms out **~30px**. This is the metric-grounding ⇄ image-fit
   tension. See `scripts/subprocess_entries/sam-3d-objects/joint_opt.py` (L_kp2d, L_hand_sil)
   and commit `5b1d604`.

2. **Temporal jitter is a paradigm signature, not a tunable bug.**
   Per-frame fitting + a smoothness prior that *fights* the data term → residual shake.
   We localized it (jitter metric in `object_confidence.py`): object enters stage 7
   smooth (0.69mm) and leaves shaky because the object **rotation had no temporal anchor**.
   Fixing that halved rotation jitter (accel p90 0.132→0.063) and cut worst-frame shake
   23% — but translation jitter (2.27mm vs the 0.69mm pre-grasp track) **persists**: the
   stage-7 object image terms still tug it off-track per-frame. Commit `b561bcd`.

3. **Monocular metric scale is unresolved.** `--depth vggt` is up-to-scale; the hand
   depth-anchor is a per-frame heuristic. Scale is an **observability** limit, not a
   capacity one — no amount of optimization or model size makes one viewpoint metric.

4. **Occlusion is the hardest regime.** Methods tie under low occlusion and diverge under
   heavy occlusion (dc-IoU 0.92 vs FoundationPose 0.88; reproj-accel p90 12 vs 16px). We
   handle it with hand-crafted don't-care masking — a learned occlusion model would be
   more general.

5. **FoundationPose is *not* the answer here (tested, ruled out).** RGB-D tracker; on
   monocular MoGe depth it tracks the depth noise — raw transl-accel 8.35 vs 3.48mm,
   drifts 4.8cm vs 0.2cm. Lesson: *match the method to your strongest evidence* (image,
   not depth). See report §7 and commit `b561bcd`.

---

## 3. Research directions, ranked

Each: the gap it closes, the idea, why it's novel vs CHOIR, a concrete first step, and risk.

### D1 — Image-grounded generative refinement *(highest value; the measured #1 gap)*
- **Gap:** CHOIR's rectification asks "is this a plausible grasp?" not "does this reproject
  onto the pixels?" — so a grasp prior can pull the hand off the video (the ~30px hand floor).
- **Idea:** a refiner that is **jointly grasp-prior-consistent AND reprojection-consistent**
  — render-and-compare *inside* the generative loop, so plausibility never overrides image
  evidence.
- **Novelty vs CHOIR:** closes the loop to the video, not just the prior.
- **First step:** add a differentiable reprojection (kp2d + silhouette) critic to the
  rectification objective; measure hand centroid/IoU vs the current optimizer baseline with
  `object_confidence.py`.
- **Risk:** low-medium. Most defensible single contribution.

### D2 — Temporally-native generative prior *(pairs with D1)*
- **Gap:** CHOIR uses a *static* grasp prior (DexGraspNet) applied per-frame; temporal
  consistency is only an optimization energy → the jitter we spent a session on.
- **Idea:** a prior **generative over the whole 4D trajectory** (a manipulation/motion prior,
  not a grasp prior) → amortized smoothness, the opt→feed-forward shift.
- **Novelty vs CHOIR:** smoothness becomes a property of the model, not a regularizer.
- **First step:** prototype the lightweight version first — dense object **point tracks**
  (CoTracker, already cloned in `third_party/co-tracker`) → fit a smooth rigid SE(3); the
  data is smooth by construction so jitter drops at the source. This is also a strong result
  on its own.
- **Risk:** medium. D1+D2 together = the recommended paper (see §4).

### D3 — Joint hand-object generation under occlusion
- **Gap:** reconstruct-then-couple; occlusion is the measured worst case.
- **Idea:** **jointly generate both bodies**, hallucinating occluded parts from the
  interaction prior, instead of correcting separately-estimated ones.
- **First step:** a joint latent over (MANO params, object 6D, contact) conditioned on
  per-frame visibility; train on synthetic grasps (§5).
- **Risk:** medium-high; likely concurrent work — differentiate on the occlusion benchmark.

### D4 — Dynamics consistency, not just kinematic contact
- **Gap:** CHOIR's contact is geometric; nothing enforces the object's *motion* is explicable
  by grasp forces (object can float despite a loose grip).
- **Idea:** a **differentiable-physics / contact-force** consistency term — object motion must
  follow from contact.
- **First step:** add a momentum/contact-force residual to stage 7; measure how many
  physically-impossible frames it removes.
- **Risk:** medium; strong if it yields a clean "physically impossible → impossible removed" story.

### D5 — Resolve metric scale *from the interaction*
- **Gap:** monocular up-to-scale; we fought the hand depth-anchor tension (limitation 3, 1).
- **Idea:** the MANO hand is a **metric ruler** and contact rigidly couples hand↔object, so
  the grasp can disambiguate object scale. Solve scale from the interaction, not from depth.
- **First step:** replace the per-frame depth anchor with a single global scale solved from
  hand size + contact; check if the 63px init gap (limitation 1) shrinks.
- **Risk:** low-medium; elegant, underexplored, directly attacks a measured floor.

### D6 — Distill to a single feed-forward spatio-temporal model *(the destination)*
- **Gap:** the whole pipeline is test-time optimization + off-the-shelf init.
- **Idea:** one spatio-temporal model trained on synthetic grasps + this pipeline's
  pseudo-GT (stage 8 already exports it). Smoothness/speed/robustness amortized.
- **First step:** the data engine (§5) + a video-native architecture; keep a thin
  optimization refinement for hard physics.
- **Risk:** high effort; clearest long-horizon paper. Caps at the teacher unless you add
  independent signal (self-supervision / multi-view / sensors).

---

## 4. Recommended single paper: D1 + D2

*"CHOIR closes the loop to a grasp prior; we close it to the video and to time."* A
**temporally-native, image-grounded generative refiner**, evaluated on **reprojection +
jitter** metrics where the static/per-frame recipe demonstrably fails.

- **Why it's defensible:** both gaps are measured here, not asserted; the diagnostic tooling
  (`scripts/object_confidence.py`) *is* the motivation and the benchmark.
- **Likely reviewer rejection — "this is just CHOIR + a video model":** preempt by (i) leading
  with the structural diagnosis (why per-frame + static-prior *must* jitter and mis-register),
  (ii) a benchmark/metric that exposes the gap and on which CHOIR-class methods score poorly,
  (iii) ablations isolating the image-grounding and the temporal-prior contributions.

---

## 5. Data strategy (you don't collect paired data — you render it)

How FoundationPose-style models get "perfect 3D GT": **synthetic rendering**, where GT pose is
free because you place the object. For HOI you need a *grasp generator* in front of the renderer.

| source | gives | role |
|---|---|---|
| **synthetic grasps** (DexGraspNet/GRAB + render + LLM/diffusion texture aug) | exact hand+object pose, contact, unlimited scale | bulk supervision (hand domain gap is the catch) |
| **real lab GT** (HO3D, DexYCB, ARCTIC, OakInk, HOI4D) | exact, real appearance | anchor metric scale + evaluate (narrow) |
| **this pipeline's pseudo-GT** (stage 8 export) | in-the-wild diversity | distillation (biased toward the optimizer) |
| **self-supervision** (photometric/temporal/contact on unlabeled video) | scale, unbiased | break the labeled-data ceiling — same losses, used as training signal |

Note: the pipeline is already **model-free at test time** (SAM-3D reconstructs the object mesh
from one image — no CAD needed), so only the *training* data problem remains, and §5 solves it.

---

## 6. Tooling already built that supports this work

- `scripts/object_confidence.py` — per-frame object **and** hand metrics: dc-IoU (occlusion-fair),
  mask coverage, centroid error, **jitter** (reproj-accel px, transl-accel mm), hand IoU/precision
  vs an auto-generated SAM2 hand-mask track. CSV + curve plot + worst-frame montage + dynamic video.
  This is the diagnosis/benchmark engine for any follow-up.
- Occlusion-robust **don't-care IoU** + SAM2 hand-occluder masking (`render_compare.py`, `joint_opt.py`).
- Hand image-registration losses: **kp2d reprojection** (HaMeR joints) + hand silhouette + acceleration.
- `third_party/co-tracker` is cloned (point tracks were scaffolded in stage 1/3 docstrings, never wired)
  — the D2 first step.
- `pipeline_report.html` — full self-contained write-up of the method, results, and the
  render-and-compare-vs-FoundationPose analysis.

## 7. Immediate, low-risk engineering next steps (double as research scaffolding)

1. **Wire CoTracker** on the SAM2 object mask → 2D tracks + visibility (download weights;
   `third_party/co-tracker`). Foundation for D2/D1.
2. **Point-track → SE(3) term:** anchor each track to the mesh at the reference frame, add a
   visibility-weighted reprojection loss to `render_compare`/`joint_opt`. Measure jitter +
   spin vs current best. (Distilled essence of the "Gaussian 4D" instinct, without the
   monocular ill-posedness.)
3. **Same term on the hand** (dense knuckle/skin tracks) → attack the 30px hand floor.
4. **Tighten residual object translation jitter** (limitation 2): stronger translation prior
   or down-weight the stage-3-redundant object image terms in stage 7. Quick win.

---

*Context: this file distills a working session that fixed left-hand chirality, added
occlusion-robust object losses, hand image-registration, and object temporal smoothing, then
A/B'd FoundationPose (lost) and reasoned through optimization-vs-feed-forward, data synthesis,
and the CHOIR follow-up space. See commits `5bced94`..`4f7b2c8` and `pipeline_report.html`.*
