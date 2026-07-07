# Contact-aware compositional 4DGS: hand + object + static-background Gaussians jointly optimized with contact and non-penetration constraints defined in Gaussian space

**Branch ID:** `b3-contact-coupled-compositional-gs`
**Overall score:** 4
**Verdict:** maybe, leaning drop (most crowded micro-niche, contact machinery targets the weaker-evidenced failure mode; pursue only under hard repositioning)

## Digest (card summary)

**TL;DR:** Rebuild hand, object, and background from one video as jointly optimized Gaussian layers, betting contact welds and a metric background anchor fix jitter and scale errors.

**Why:**
- All four harness baselines mis-scale objects +33-43% and jitter; per-frame heuristics never couple hand-object.
- Bet: make jitter unrepresentable — weld object to wrist per grasp, anchor scale to background.
- HOI4D GT-depth harness with four reproduced baselines exists; this exact constraint structure remains unclaimed.

**Pipeline:**
1. **Masked metric SLAM** — MASt3R-SLAM + UniDepth on hand/object-masked frames builds static background Gaussians, camera path, metric scale. — `video + SAM2 masks -> background 3DGS, camera SE(3), scale gauge`; reuses SAM2/SAM3 point-prompt mask protocol validated on harness
2. **Per-frame initialization** — WiLoR hand poses and HORT/SAM-3D object Gaussians per frame, fused into canonical object shape. — `frames -> MANO params + canonical object GS + rough SE(3)`; reuses WiLoR/HaMeR, HORT, SAM-3D wired in render_and_compare
3. **Spline trajectory fitting** — Fit wrist, articulation, and object SE(3) cubic B-splines (knots ~0.2s); jitter becomes unrepresentable. — `per-frame poses -> band-limited se(3) splines`
4. **Joint photometric refinement** — Whole-clip optimization: composite splat render vs. frames, plus mask, monodepth, and CoTracker track losses. — `3-layer GS + splines -> refined scene, no contact terms`
5. **EM contact coupling** — Alternate: infer contact phases (CHOIR fields + track rigidity), re-optimize with grasp weld and penetration energies. — `refined scene -> contact-consistent 4D scene (T_o = T_wrist * G_p * dT)`; reuses CHOIR contact fields in render_and_compare backend
6. **Harness evaluation, ablations** — Score scale/depth, jitter RMS, penetration, contact-persistence, chamfer against four baselines; peel off each term. — `4D scenes -> metric table on ~40 HOI4D clips`; reuses HOI4D GT-depth harness + 4 adapted baselines (compare/hoi4d)

**What's new:** Vs. CHOIR (arXiv'26, contact-aware joint HOI optimization): moves contact into splatting — structural wrist-weld reparametrization plus SLAM-anchored background layer giving metric scale, absent from all HOI-GS works.

**Top risks:**
- ⚠ Saturating niche: eight monocular HOI-GS papers in 14 months; GHOST/CHOIR already claim contact-awareness.
- ⚠ Wrist weld propagates monocular hand-depth error (HaWoR +28%) straight into the object.
- ⚠ New interaction metrics are satisfied by construction; motivating evidence is one kettle clip.

**Kill test:** 2-3 week pilot on ~10 HOI4D clips: best baseline + temporal smoothing + monodepth loss vs. full contact-coupled system. Go if structural coupling clearly wins scale/depth AND chamfer; kill if the cheap bolt-on closes most of the gap.

**If it works:** Erases the harness's +33-43% object-scale and -29 to -81% depth errors plus hand-object jitter, without per-frame chamfer/MPJPE loss, beating four reproduced baselines.

## Branch definition

- **Id:** b3-contact-coupled-compositional-gs
- **Title:** Contact-aware compositional 4DGS: hand + object + static-background Gaussians jointly optimized with contact and non-penetration constraints defined in Gaussian space
- **Angle:** Interaction physics as the differentiator: extend b2's compositional scene with (i) a static background Gaussian layer that anchors camera pose and metric scale, and (ii) contact energies coupling the hand and object Gaussian sets — attraction of contact-labeled hand Gaussians to the object surface during grasp phases, SDF-style non-penetration, and shared-motion constraints while in contact (object velocity tied to contacting-hand velocity, detected via TAPIR/CHOIR-style contact inference).
- **Core Hypothesis:** Relative hand-object jitter (floating/penetration flicker) is the perceptually and metrically dominant failure mode, and photometric losses alone under-constrain the interaction; contact coupling in the joint GS optimization removes it, directly improving the contact and relative-depth metrics in the HOI4D harness where all 4 baselines fail.
- **Why Distinct:** The novelty claim is contact-in-the-GS-loop plus background-anchored scale — a constraint structure absent from hand-avatar GS, MANUS (multi-view, known setup), and generic dynamic-GS works. b2 could exist without any contact term; b3 is specifically about the hand-object coupling and full-scene composition.
- **Builds On:**
  - b2-template-embedded-gs
  - MANUS (contact via Gaussians, but multiview)
  - CHOIR contact modeling (user's render_and_compare)
  - TAPIR (contact-phase detection)
  - SplaTAM-style background anchoring

## Clarification

- **Problem Statement:** Given a single monocular RGB video (moving/egocentric camera) of a hand manipulating an unknown rigid object, recover a metrically scaled 4D reconstruction — MANO hand trajectory, model-free object shape with its SE(3) trajectory, and camera path — that is simultaneously per-frame accurate AND temporally/physically consistent: no hand-object floating, no interpenetration flicker, no relative-depth or scale drift. All four pipelines evaluated on our HOI4D GT-depth harness (render_and_compare, do-as-i-do, ForeHOI, HORT) fail this: monocular methods mis-scale the object by +33–43% and misplace its depth by −29% to −81% (kettle clip C12/N15), and every method exhibits relative hand-object jitter that required ad-hoc temporal-smoothing patches — because they estimate frames (nearly) independently and couple hand and object only through per-frame heuristics, so monocular evidence never constrains the interaction itself. The problem is to reformulate 4D HOI reconstruction as one full-clip test-time optimization in which temporal smoothness and contact physics are structural properties of the parametrization, not post-hoc filters — within the multi-hour per-clip budget that existing pipelines (do-as-i-do: ~3 h per 5 s clip) already spend.
- **Technical Core:** A three-layer compositional 4D Gaussian scene rendered through one differentiable rasterizer with alpha compositing, optimized jointly over the whole clip. (1) BACKGROUND layer: static world-frame 3DGS initialized from hand/object-masked metric SLAM (DROID/MASt3R-SLAM + MoGe/UniDepth metric depth), with per-frame camera SE(3) refined SplaTAM-style; static-by-parametrization, it pins the camera trajectory and the metric scale gauge so hand-object relative-depth errors become photometrically visible misalignments (this is exactly the anchor that made render_and_compare GT-exact when GT depth was injected — here recovered, not given). (2) HAND layer: Gaussians rigged to MANO (barycentric face attachment + LBS); free parameters are shape beta, per-Gaussian appearance/opacity/scale, and a continuous-time pose trajectory — wrist SE(3) and PCA-articulation as cubic B-splines in se(3) with knots every ~0.2 s, so the trajectory is band-limited and jitter is unrepresentable. (3) OBJECT layer: canonical model-free Gaussians (initialized from SAM-3D/HORT lifted by clip-wide fusion, which also fills hand-occluded regions) plus a rigid SE(3) spline trajectory. CONTACT COUPLING, the differentiator, is one reparametrization plus two Gaussian-space energies, gated by an inferred contact posterior: (a) phase detection — CHOIR dense hand-contact fields (already in the render_and_compare backend) fused with a track test (TAPIR/CoTracker tracks on the object mask are marked in-contact when their motion is rigidly explained by wrist motion over a window), segmenting the clip into free/approach/grasp/release; (b) grasp-phase reparametrization — during each grasp phase p the object trajectory becomes T_o(t) = T_wrist(t) ∘ G_p ∘ ΔT(t) with G_p one constant grasp transform and ΔT(t) an identity-regularized slack spline, structurally deleting the relative-pose DOF that produce floating/flicker while still permitting evidence-driven in-hand slip; (c) attraction — contact-labeled hand Gaussians x_i are pulled to the object iso-surface using the object Gaussian mixture density F_o(x)=Σ_k o_k N(x; μ_k, Σ_k) as a soft SDF: E_att = Σ_i (F_o(x_i) − τ)²; (d) non-penetration, symmetric — E_pen = Σ_k ReLU(−sdf_MANO(μ_k^obj))² using the exact differentiable point-to-mesh SDF of the posed watertight MANO, plus Σ_{i∉contact} ReLU(F_o(x_i) − τ)² for hand Gaussians. Full objective: photometric L1 + D-SSIM on the composite render, per-layer SAM2/SAM3 video-mask supervision (point-prompt protocol already validated on the harness), rendered-depth vs. metric mono-depth consistency, long-range 2D track reprojection on hand and object Gaussians (disambiguates low-texture skin/plastic), E_att + E_pen, and spline acceleration priors. Optimization schedule fitting the multi-hour budget: Stage A masked background SLAM + camera; Stage B per-frame initialization (WiLoR/HaMeR hand, HORT/SAM-3D object) fit by splines; Stage C joint photometric refinement without contact; Stage D 2–3 EM rounds — E-step re-estimate contact posteriors/phase boundaries from current geometry and tracks, M-step re-optimize with (b)–(d) — making contact inference self-correcting rather than trusted upfront. Evaluation reuses the existing HOI4D GT-depth harness and the four adapted baselines in the common scene format: current size/depth-error metrics plus new interaction metrics (relative hand-object pose acceleration RMS, penetration depth/volume rate, contact-persistence IoU against contact derived from HOI4D GT hand+object poses) and per-frame chamfer/MPJPE to demonstrate no accuracy trade-off; ablations peel off contact terms (recovering the b2 baseline), background layer, splines, and grasp reparametrization to attribute the gains.
- **Novelty Claim:** The first monocular in-the-wild 4D hand-object reconstruction that puts interaction physics inside the splatting optimization itself — a hand-rigged + model-free-object + static-background compositional Gaussian scene whose object trajectory is reparametrized through per-grasp constant grasp transforms and regularized by Gaussian-density contact-attraction and non-penetration energies, so contact consistency and temporal smoothness hold by construction — a constraint structure absent from HOLD (per-frame implicit-SDF poses, no contact/background/scale anchor), MANUS (multi-view rig, known setup), hand-avatar GS (hand-only), and HUGS-style human-scene composition (no object, no contact).
- **Target Venue:** CVPR (primary: reconstruction-system contribution with a rigorous quantitative harness and 4 reproduced baselines; CoRL as fallback if repitched as a demonstration-data engine for manipulation policy learning)
- **Key Assumptions:**
  - The manipulated object is rigid (HOI4D-style manipulation); articulated/deformable objects are out of scope or a stated extension.
  - Contact-phase detection (CHOIR contact fields + rigid-track-correlation) is precise enough that grasp phases are correctly segmented; false-positive contact is the main risk, mitigated by soft posteriors, the identity-regularized slack spline, and EM re-estimation rather than hard upfront trust.
  - There is enough camera motion and background texture/parallax for masked metric SLAM to anchor scale and camera (true for egocentric HOI4D; degrades on static-tripod textureless scenes, where scale falls back to metric mono-depth priors).
  - Per-frame initializations (WiLoR/HaMeR hand, SAM-3D/HORT object, SAM2/SAM3 point-prompt masks — all already wired in the repo) land within the basin of convergence of the joint photometric refinement.
  - Accumulating object observations across the clip provides enough angular coverage to fuse a usable canonical Gaussian shape despite persistent hand occlusion.
  - Relative hand-object jitter, floating, and penetration flicker are the dominant residual failure mode (supported by the harness: +33-43% size / -29-81% depth errors for monocular baselines, plus the documented temporal-smoothing and scale-fix patches), and reviewers will accept new temporal/contact metrics alongside standard per-frame ones.
  - A multi-hour offline per-clip budget is acceptable to the community for this problem (precedent: do-as-i-do already spends ~3 h on a 5 s clip; HOLD-style per-clip optimization is an accepted paradigm).
  - Photometric splatting loss is informative on low-texture skin and objects once combined with mask and long-track losses; motion blur and specularities are handled by robust losses rather than breaking convergence.
  - GT contact for the contact-persistence metric can be derived from HOI4D's GT hand and object pose annotations (proximity thresholding), since HOI4D ships GT depth and poses but not explicit contact labels.

## Novelty check

- **Closest Works:**
  - **Title:** CHOIR: Contact-aware 4D Hand-Object Interaction Reconstruction
  - **Venue Year:** arXiv, May 2026
  - **Url:** https://arxiv.org/abs/2605.20992
  - **Relation:** Closest conceptual overlap: monocular open-world video, model-free object shape + 6D pose over time, and contact used as an explicit hand-object coupling signal via contact-aware joint optimization with dynamically updated contact constraints (essentially the branch's EM-style contact re-estimation idea). It validates the branch's core hypothesis but also preempts much of it; it does not appear to use Gaussian splatting as the optimization substrate, nor a static-background layer for metric-scale/camera anchoring, nor a grasp-transform trajectory reparametrization.

  - **Title:** GHOST: Fast Category-agnostic Hand-Object Interaction Reconstruction from RGB Videos using Gaussian Splatting
  - **Venue Year:** arXiv, Mar 2026
  - **Url:** https://arxiv.org/abs/2603.18912
  - **Relation:** Monocular RGB video, Gaussian (2DGS disc) hands+objects, category-agnostic, with grasp-aware alignment refining hand pose and OBJECT SCALE for physically consistent contact, plus hand-aware occlusion-robust losses; evaluated on ARCTIC/HO3D/in-the-wild. Directly contests the claim of 'first physically consistent contact inside a GS pipeline', though its contact handling is an alignment step rather than continuous contact/non-penetration energies in a whole-clip joint optimization, and it has no SLAM-anchored metric background layer.

  - **Title:** Interaction-Aware 4D Gaussian Splatting for Dynamic Hand-Object Interaction Reconstruction
  - **Venue Year:** arXiv, Nov 2025 (rev. Jun 2026)
  - **Url:** https://arxiv.org/abs/2511.14540
  - **Relation:** Monocular 4DGS for HOI without object priors: interaction-aware hand-object Gaussians with occlusion/edge parameters, hand-conditioned object deformation fields, physical-interaction and smooth-motion regularizers, and progressive dynamic-vs-static-background handling. Overlaps the compositional 4DGS + background + interaction-regularization story; lacks explicit contact attraction/non-penetration energies, MANO-SDF, grasp reparametrization, and metric-scale anchoring.

  - **Title:** Clay-to-Stone: Phase-wise 3D Gaussian Splatting for Monocular Articulated Hand-Object Manipulation Modeling
  - **Venue Year:** CVPR 2026 (Highlight)
  - **Url:** https://cvpr.thecvf.com/virtual/2026/poster/38611
  - **Relation:** Monocular GS-based hand-object manipulation with a phase-structured optimization (flexible CLAY phase then rigidity-enforcing STONE phase) — publishes the 'phase-wise structure inside GS optimization' pattern the branch uses for grasp phases, though aimed at articulated objects and without contact energies or background/scale anchoring. Code: https://github.com/ru1ven/ARGS.

  - **Title:** BIGS: Bimanual Category-agnostic Interaction Reconstruction from Monocular Videos via 3D Gaussian Splatting
  - **Venue Year:** CVPR 2025
  - **Url:** https://arxiv.org/abs/2504.09097
  - **Relation:** Monocular HOI 3DGS with MANO-prior hand Gaussians, model-free object Gaussians completed via SDS diffusion prior, and an interacting-subjects joint alignment step. Establishes the MANO-rigged-GS + model-free-object-GS composition; no explicit contact/non-penetration energies, background layer, metric scale, or trajectory splines.

  - **Title:** Grasp-and-Lift: Executable 3D Hand-Object Interaction Reconstruction via Physics-in-the-Loop Optimization
  - **Venue Year:** arXiv, Jan 2026
  - **Url:** https://arxiv.org/html/2601.18121v1
  - **Relation:** Occupies the 'interaction physics as the differentiator' position via a stronger mechanism: simulation-in-the-loop where the wrist is welded to a kinematic control body and the object's motion is computed from contact forces, with cubic-spline low-jerk trajectories. Functionally supersedes the branch's grasp-transform reparametrization + contact energies for physical plausibility, though it is mesh/simulator-based, not a differentiable-rendering GS scene with background.

  - **Title:** DP-DeGauss: Dynamic Probabilistic Gaussian Decomposition for Egocentric 4D Scene Reconstruction
  - **Venue Year:** arXiv, Apr 2026
  - **Url:** https://arxiv.org/abs/2604.07986
  - **Relation:** Egocentric 4DGS that probabilistically routes a unified Gaussian set into background / hand / object branches with category masks — publishes the three-layer compositional background+hand+object GS scene decomposition itself, without MANO rigging, contact physics, or metric-scale claims.

  - **Title:** Grasp in Gaussians (GraG): Fast Monocular Reconstruction of Dynamic Hand-Object Interactions
  - **Venue Year:** arXiv, Apr 2026
  - **Url:** https://arxiv.org/abs/2604.12929
  - **Relation:** Monocular dynamic HOI with a compact sum-of-Gaussians representation, SAM3D-video object initialization (same init the branch plans), joint/depth alignment losses, 6.4x faster tracking. Overlaps the pipeline plumbing and the dynamic-HOI-in-Gaussians framing; no contact energies or background anchoring.

  - **Title:** HOLD: Category-agnostic 3D Reconstruction of Interacting Hands and Objects from Video
  - **Venue Year:** CVPR 2024
  - **Url:** https://arxiv.org/abs/2311.18448
  - **Relation:** The acknowledged baseline paradigm: per-clip optimization of hand + model-free object from monocular video, already including a contact loss favoring physically plausible hand-object constellations — so 'contact in the per-clip HOI optimization' per se dates to 2024; the branch's delta must rest on the GS substrate, structural (not penalty-only) coupling, and scale/background anchoring.

  - **Title:** HaWoR: World-Space Hand Motion Reconstruction from Egocentric Videos
  - **Venue Year:** CVPR 2025
  - **Url:** https://arxiv.org/abs/2501.02973
  - **Relation:** Covers the background-anchoring ingredient: adaptive SLAM for egocentric video fused with a metric depth foundation model to recover metrically scaled world-space hand trajectories — the branch's Stage-A (masked metric SLAM anchoring camera + scale) is essentially this, extended to objects.

  - **Title:** MANUS: Markerless Grasp Capture using Articulated 3D Gaussians
  - **Venue Year:** CVPR 2024
  - **Url:** https://arxiv.org/pdf/2312.02137
  - **Relation:** Contact estimation via articulated hand Gaussians and object Gaussians (Gaussian-space proximity), but in a dense multi-view rig with known setup — the branch correctly cites it; it defines contact-in-Gaussian-space but not as an optimization constraint in a monocular setting.

- **Delta:** Not already done as a whole, but the headline claim needs rewriting. Every individual ingredient is now published: contact-as-coupling-signal joint optimization on monocular video (CHOIR '26, HOLD '24), grasp-aware contact alignment inside a monocular GS pipeline (GHOST '26), phase-structured GS optimization for manipulation (Clay-to-Stone CVPR'26 Highlight), MANO-rigged + model-free-object Gaussians (BIGS CVPR'25), background/hand/object compositional egocentric GS (DP-DeGauss '26, Interaction-Aware 4DGS '25), SLAM+metric-depth world anchoring (HaWoR CVPR'25), and simulator-grade interaction physics with wrist-welded object motion and spline trajectories (Grasp-and-Lift '26). The surviving delta is the specific constraint STRUCTURE and the metrology: (1) a metric SLAM-anchored static-background GS layer that turns hand-object relative-depth/scale errors into photometric residuals — absent from all HOI-GS works, which drop the background and leave scale unanchored (the +33-43% size errors the harness documents); (2) grasp-phase kinematic reparametrization T_o=T_wrist∘G_p∘ΔT plus band-limited se(3) splines, making jitter and floating unrepresentable by construction rather than penalized (GHOST/CHOIR/HOLD all use soft losses or one-shot alignment; Clay-to-Stone's phases target articulation rigidity, not hand-object coupling); (3) Gaussian-mixture-density attraction + posed-MANO-SDF non-penetration as energies inside the whole-clip splatting objective with EM contact re-estimation; and (4) an evaluation contribution: GT-depth harness with 4 reproduced baselines and new relative-jitter/penetration/contact-persistence metrics. To be defensible at CVPR the paper must position against and empirically beat GHOST, CHOIR, and BIGS (not just HOLD/MANUS), and the claim should be 'structural contact coupling + background-anchored metric scale in compositional 4DGS', not 'first contact in splatting'.
- **Crowdedness:** crowded
- **Crowdedness Reason:** Monocular hand-object reconstruction with Gaussian splatting has become one of the hottest micro-niches of 2025-2026: at least eight directly overlapping papers appeared in the last ~14 months (BIGS CVPR'25, Interaction-Aware 4DGS 11/25, Grasp-and-Lift 1/26, AGILE 2/26, GHOST 3/26, GraG 4/26, DP-DeGauss 4/26, CHOIR 5/26, Clay-to-Stone CVPR'26 Highlight), and two of them (GHOST, CHOIR) already claim contact-aware/physically-consistent monocular HOI. The exact intersection proposed (background-anchored metric scale + structural grasp coupling + contact energies in one 4DGS optimization) is still unclaimed, so it is 'crowded' rather than 'saturated' — but the window is closing fast and concurrent-work collisions by a CVPR 2027 deadline are likely.
- **Already Done:** False
- **Search Confidence:** verified-by-search

## Pressure test

- **Failure Modes:**
  - Concurrent-work collision is near-certain, not hypothetical: 8+ monocular HOI-GS papers in 14 months, two of which (GHOST 3/26, CHOIR 5/26) already claim contact-aware/physically-consistent monocular HOI, and Clay-to-Stone (CVPR'26 Highlight) already published phase-structured GS optimization for manipulation. By a Nov 2026 CVPR'27 deadline the surviving delta ('constraint structure + background scale anchor') will read to reviewers as a recombination of published ingredients, and 1-2 more collisions in the interim are likely.
  - The grasp weld T_o = T_wrist ∘ G_p ∘ ΔT structurally propagates hand errors into the object — the exact bug this team already diagnosed in its own harness: do-as-i-do's hand-anchored depth inherited HaWoR's +28% hand-depth error until they replaced it with a GT anchor. Monocular wrist SE(3) (WiLoR/HaMeR) is depth-ambiguous and jittery; welding the object to it trades hand-object relative jitter for correlated absolute error, and the joint optimization must fix the hand and object simultaneously for the weld to help — a circular dependency the proposal never resolves.
  - EM contact-phase estimation over discrete grasp boundaries inside a nonconvex photometric landscape is fragile: a mis-detected grasp onset welds the object to the wrist during approach (object drags/teleports), and once the M-step has satisfied the weld photometrically, the E-step has little gradient signal to undo it. Soft posteriors + slack splines mitigate but also reintroduce the very relative-pose DOF the reparametrization claims to delete — the method's headline ('jitter unrepresentable by construction') and its escape hatch (identity-regularized ΔT(t) slack) are in direct tension.
  - The claim that a background GS layer makes relative-depth errors 'photometrically visible' is optically weak: depth errors along the optical axis are precisely what photometric losses are least sensitive to under the small parallax of a 5 s handheld clip. The real depth signal is the rendered-depth vs. metric-monodepth loss — which any baseline could bolt on — so the ablation must disentangle 'contact structure' gains from plain depth supervision, and reviewers will suspect the latter explains most of the scale/depth improvement.
  - Self-serving evaluation circularity: the proposed new metrics (relative pose acceleration RMS, penetration rate, contact-persistence IoU) are exactly the quantities the method's constraints enforce by construction. A trivially wrist-welded object scores perfectly on all three while being geometrically wrong. Unless the method also wins (not just ties) on independent per-frame chamfer/MPJPE/ADD, an AC will read the metric suite as designed for the method.
  - The Gaussian-mixture density F_o(x) as a soft SDF has a known degenerate solution: the optimizer can satisfy E_att and E_pen by shrinking opacities/scales of Gaussians near the contact region instead of moving geometry, since o_k, Σ_k are co-optimized with μ_k. This requires detach schedules and per-term LR surgery — exactly the fragile-hyperparameter territory that kills reproducibility and rebuttals.
  - The motivating evidence is n=1: the entire quantitative case (+33-43% size, −29 to −81% depth) comes from ONE 5-second kettle clip (compare/hoi4d/depth_eval.md), and the 'jitter is dominant' claim from one commit-level flicker fix — which, tellingly, a cheap temporal depth-smoothing patch already fixed. If a 10-line smoothing filter removes the perceptual artifact, demonstrating that hours of structural contact machinery buys a visible margin over 'baseline + smoothing + monodepth loss' is a genuinely hard experimental bar.
  - Baseline reproduction gap: to be defensible the paper must beat GHOST, CHOIR, and BIGS, but GHOST and CHOIR are 2026 arXiv works with uncertain code release; comparing only against HOLD/HORT/ForeHOI (what the harness has) invites a one-line rejection for missing the closest published baselines, and re-implementing either is a multi-month project by itself.
  - Benchmark scale-up cost: a credible CVPR paper needs ~30-50 HOI4D clips plus HO3D/ARCTIC or in-the-wild results for comparability; at multi-hour whole-clip optimization × (method + 5 ablations) this is hundreds of GPU-hours of turnaround per experimental iteration, which slows the debug loop of an already fragile multi-stage system (masked SLAM → per-frame init → joint photometric → EM contact) where any stage failing sinks the clip.
  - Paradigm-timing risk: the field is visibly moving feed-forward (HORT ~1 min, GraG advertises 6.4x faster tracking); a multi-hour per-clip test-time optimization pitched at CVPR 2027 continues a 2024 paradigm (HOLD) and will draw 'expensive per-scene optimization, limited practical relevance' reviews unless the accuracy gap over fast methods is dramatic.
- **Unsound Assumptions:**
  - 'Relative hand-object jitter is the perceptually and metrically dominant failure mode' — the harness actually documents SCALE and DEPTH errors as dominant (addressed by the background/depth anchor, not by contact coupling); the jitter evidence is one flicker anecdote already fixed by trivial temporal smoothing, so the differentiating machinery targets the weaker-evidenced failure.
  - 'Per-frame initializations land within the basin of convergence of joint photometric refinement' — contradicted by the harness itself: HORT places the object at −81% depth; nothing photometric recovers from that, so convergence actually rests on the depth/track losses being strong enough to move the object ~0.7 m, which is asserted, not shown.
  - 'Background GS layer turns relative-depth/scale errors into photometric residuals' — under 5 s of handheld parallax, along-axis depth errors are nearly photometrically invisible; the anchor that was 'GT-exact' in render_and_compare used injected GT depth, and replacing GT depth with SLAM+monodepth (5-15% metric error typical) does not license extrapolating that exactness.
  - 'Reviewers will accept new temporal/contact metrics alongside standard ones' — risky when the new metrics are optimized by construction; acceptance hinges on winning independent metrics, which the proposal only promises to 'not trade off'.
  - 'GT contact can be derived from HOI4D GT poses by proximity thresholding' — HOI4D object-pose annotations have documented noise at the centimeter level, comparable to the contact threshold itself, so the contact-persistence GT is itself uncertain in exactly the regime being measured.
  - 'Soft posteriors + EM make contact inference self-correcting' — EM over discrete phase boundaries with a photometric M-step has no guarantee of escaping a wrong initial segmentation; CHOIR's dynamically-updated contact constraints already occupy this idea, so it is both fragile and non-novel.
  - 'Clip-wide fusion provides enough angular coverage for canonical object shape' — HOI4D manipulations are short and one-sided; the object back face is often never observed, forcing a diffusion/SDS completion prior (as BIGS needed), which the technical core omits.
- **Feasibility:**
  - **Data:** Good raw availability, thin current state. HOI4D (GT depth + poses) is in hand and partially wired, SAM2/SAM3 mask protocol validated, four baselines adapted — but the harness today is ONE kettle clip (compare/hoi4d/depth_eval.md). Scaling to 30-50 rigid-object clips across categories plus an external benchmark (HO3D/ARCTIC for comparability with GHOST/BIGS numbers) is mandatory and is weeks of curation; GT contact labels must be synthesized from noisy pose annotations. Rigid-only scope excludes several HOI4D categories.
  - **Compute:** Acceptable but painful. Single-GPU per clip is fine; the problem is iteration velocity: whole-clip 4DGS + per-iteration posed-MANO SDF + spline optimization + 2-3 EM rounds plausibly lands at 2-6 h/clip, and a full result table (method + 5 ablations + 4 baselines × ~30-50 clips) is 500-1500 GPU-hours per experimental sweep. On rented GPUs this is affordable in money but expensive in debug-loop latency for a system with 4 sequential fragile stages.
  - **Evaluation:** The weakest leg. (1) Closest baselines GHOST/CHOIR/BIGS likely lack usable code; without beating them head-to-head the novelty delta collapses to unverifiable prose. (2) The three new interaction metrics are enforced by the method's own constraints — circular unless per-frame chamfer/MPJPE/ADD also improve, which the mechanism gives no strong reason to expect. (3) The most damaging ablation ('baseline + temporal smoothing + monodepth loss', i.e., the cheap fixes already applied in this repo) is exactly the one reviewers will demand and the one the structural machinery may fail to clearly beat.
- **Scores:**
  - **Novelty:** 4
  - **Feasibility:** 5.5
  - **Crowdedness Inverse:** 2
  - **Venue Fit:** 5
  - **Overall:** 4
- **Verdict:** maybe
- **Verdict Reason:** As specified, this is a 2026-vintage recombination entering the single most crowded micro-niche in the area, with its headline mechanism (contact coupling) targeting the weaker-evidenced failure mode (n=1 flicker anecdote already fixed by cheap smoothing) while its stronger-evidenced problem (metric scale/depth, +33-43%/−81% on one clip) is addressed by an ingredient (SLAM+monodepth background anchoring) that is closest to a bolt-on depth loss any baseline could adopt. It escapes 'drop' only because two assets are real and rare: a GT-depth harness with four reproduced baselines (currently n=1 clip — must become n≈40 before anything else), and the genuinely unclaimed metric-background-anchored composition. Pursue only under a hard repositioning: lead with the benchmark/metrology contribution plus background-anchored metric scale, demote contact energies to one ablated component, budget for reproducing or numerically matching GHOST/CHOIR/BIGS, and pre-register the killer ablation (baseline + smoothing + monodepth loss) — if that ablation closes most of the gap in a 2-3 week pilot on ~10 clips, drop the branch.

## Scores

- **Id:** b3-contact-coupled-compositional-gs
- **Title:** Contact-aware compositional 4DGS with contact/non-penetration constraints in Gaussian space
- **Overall:** 4
- **Verdict:** maybe, leaning drop (most crowded micro-niche, contact machinery targets the weaker-evidenced failure mode; pursue only under hard repositioning)
