# Attack the root cause only: GS-based bundle adjustment of camera + metric scale + background to stabilize depth/scale, feeding existing HOI pipelines

**Branch ID:** `b6-gs-scale-depth-anchor`
**Overall score:** 5.5
**Verdict:** maybe (strongest: highest feasibility, empty niche, cheapest falsification; run the two gating pilots first)

## Digest (card summary)

**TL;DR:** Stabilize camera and metric depth scale per video with background Gaussians, then rerun four existing HOI pipelines unchanged — betting jitter comes from scale wobble, not pose error.

**Why:**
- All four pipelines jitter measurably: hands teleport 1.15 m in depth, apparent size flickers 1.75x.
- GT-depth injection zeroes object depth/size error; better depth nets change nothing — gauge is the culprit.
- Cheapest, most falsifiable branch; doubles as the ablation de-risking every sibling branch.

**Pipeline:**
1. **Gauge-transfer pilot** — Verify background-fitted affine depth corrections also fix hand/object-region depth error against GT. — `MoGe-2 depth + masks + HOI4D GT depth -> per-region residual error`; reuses HOI4D GT-depth harness, SAM2 masks
2. **Static-scene Gaussian bundle adjustment** — Jointly optimize background Gaussians, per-frame cameras, one global metric scale, per-frame affine gauges (a_t,b_t). — `video + dynamic masks + MoGe-2 depth -> cameras T_t, scale s, gauges (a_t,b_t)`; reuses SAM2/SAM3 masks all baselines already compute; GeoCalib
3. **Gauge transfer to dynamic pixels** — Apply corrected depth everywhere; render amodal support-surface depth and reliability behind dynamic masks. — `gauges + background Gaussians -> corrected metric depth D~_t per frame`
4. **Inject into unchanged pipelines** — Substrate depth replaces MoGe/anchors via existing interfaces; HORT/ForeHOI get closed-form output re-gauge. — `D~_t -> re-run HORT, ForeHOI, render_and_compare, do-as-i-do`; reuses real_perception.py sensor-depth path; do-as-i-do --gt-anchor
5. **Oracle-sandwich benchmark** — Compare monocular/+substrate/+GT-depth per method; ablate substrate against ViPE, Video-Depth-Anything, COLMAP. — `20-50 HOI4D clips -> gauge-vs-pose jitter decomposition + stability suite`; reuses HOI4D GT-depth ablation, repo stability diagnostics

**What's new:** TRAM (ECCV'24) grounds human bodies via masked-background SLAM scale; this transfers per-frame affine gauge corrections to hand/object pixels and causally decomposes jitter across four unchanged HOI pipelines.

**Top risks:**
- ⚠ ViPE/MegaSaM commodity engines may close the oracle gap too, demoting method to analysis paper.
- ⚠ Background-fitted affine gauge may not extrapolate to near-field hand/object pixels.
- ⚠ Whole diagnosis rests on one kettle clip; "gauge dominates" could invert at scale.

**Kill test:** Per-frame, fit affine (a_t,b_t) from MoGe to HOI4D GT depth on background pixels only; go if it cuts hand/object-pixel depth error >=50% across >=10 clips — no GS code needed; else kill.

**If it works:** Closes most of the GT-depth oracle gap with zero HOI-method changes; first gauge-vs-pose decomposition of 4D HOI jitter — CVPR analysis+system paper.

## Branch definition

- **Id:** b6-gs-scale-depth-anchor
- **Title:** Attack the root cause only: GS-based bundle adjustment of camera + metric scale + background to stabilize depth/scale, feeding existing HOI pipelines
- **Angle:** Minimal-intervention framing targeting the diagnosed root cause (per-frame depth/scale inconsistency and hand-depth wobble): use static-scene Gaussians + GS-SLAM-style joint camera/scale optimization (SplaTAM/MoGe-anchored) to produce a temporally consistent metric depth + camera trajectory, then re-run the UNCHANGED existing HOI methods on top of the stabilized depth/scale. No hand or object Gaussians at all.
- **Core Hypothesis:** A large fraction of the observed jitter is scale/depth wobble, not pose error; fixing the shared scale/camera/depth substrate via multi-frame GS consistency eliminates it upstream, and this can be quantified precisely with the HOI4D GT-depth ablation (GT depth injection already showed residual jitter, so this branch also stress-tests whether the root-cause diagnosis is complete).
- **Why Distinct:** It changes the INPUT to HOI methods rather than the HOI representation or optimizer — cheapest, most falsifiable branch, and doubles as the ablation that tells you how much of b1-b3's gain comes from scale stabilization versus pose coupling. Also the only branch that remains useful if template-embedded GS turns out to be crowded prior art.
- **Builds On:**
  - SplaTAM / GS-SLAM
  - MoGe / DA3-metric depth
  - GeoCalib
  - user's 4 baselines rerun unchanged
  - HOI4D GT-depth ablation result

## Clarification

- **Problem Statement:** Monocular 4D hand-object-interaction pipelines of every family — per-frame feed-forward (HORT), video feed-forward (ForeHOI), optimization-based (render_and_compare/CHOIR), and tracker-based (do-as-i-do) — produce temporally unstable metric reconstructions. On a common harness the failures are measured, not anecdotal: HORT's hand centroid teleports ~1.15 m in z frame-to-frame while x/y barely move; do-as-i-do's per-frame depth gauge swings 2x (translation_scale 0.52->1.01), producing 1.75x apparent-size flicker whose source is the HaWoR hand anchor (corr 0.775), and the hand's own depth wobbles 7.9x its true path length; render_and_compare's hand comes out at a uniform 0.77x scale; absolute object depth/size errors on HOI4D reach 29-81%. Yet injecting GT sensor depth into the same UNCHANGED pipelines makes object depth/size essentially exact (0% error for render_and_compare; 0% depth for do-as-i-do with a depth anchor), and swapping a better per-frame depth net (MoGe->DA3-metric) changes nothing (1.75x->1.76x flicker). The shared root cause is therefore not pose estimation but the per-frame, independently estimated metric gauge (depth scale/shift, camera pose, global scale). The problem: from the monocular video alone, recover the temporally consistent metric substrate — camera trajectory, one global scale, per-frame gauge-corrected dense depth with amodal background completion — that the oracle experiment shows is the missing input; feed it to the four existing HOI pipelines unchanged; and quantify precisely what fraction of the GT-depth oracle's stability and accuracy gains this substrate restores, thereby decomposing 4D HOI jitter into gauge error vs. pose error vs. shape-prior error.
- **Technical Core:** One mechanism — per-frame monocular depth-gauge re-anchoring against a static-scene 3D Gaussian bundle adjustment — wrapped in a 4-part system. (1) Dynamic-masked static-scene 3DGS BA (offline, all frames jointly; not incremental SLAM): reuse the SAM2/SAM3 hand+object masks every baseline already computes to exclude dynamic pixels; jointly optimize {background Gaussians G; per-frame camera T_t in SE(3); one global metric scale s; per-frame 2-parameter affine depth-gauge corrections (a_t, b_t) on the monocular metric-depth prior (MoGe-2/DA3-metric)} under an opacity/silhouette-gated photometric loss on background pixels + lambda*|D_GS,t - (a_t*D_mono,t + b_t)| on co-visible static pixels + GeoCalib gravity alignment + shrinkage of (a_t,b_t) to (1,0) + camera-trajectory smoothness. A single static model explaining all frames makes background depth consistent BY CONSTRUCTION; s is pinned by robust cross-frame consensus of the metric prior over static pixels, optionally sharpened by object-at-rest support-plane contact. Static-camera clips degrade gracefully to multi-frame depth consensus. (2) Gauge transfer to dynamic pixels: apply the background-fitted correction D~_t = a_t*D_mono,t + b_t to ALL pixels, including the hand/object regions the Gaussians never model — valid exactly because every diagnosed failure is a per-frame GLOBAL gauge error (uniform 0.77x hand scale; 2x scale swing; z-only teleports), not a local relative-structure error; the GS additionally renders amodal support-surface depth behind the dynamic masks plus per-pixel reliability from opacity/residuals. (3) Injection through the interface the HOI4D GT-depth harness already defines (16-bit metric depth + K at real_perception.py's sensor-depth path): render_and_compare consumes D~_t in place of MoGe (the path proven exact under GT injection); do-as-i-do switches its object anchor from the HaWoR hand to the substrate pointmap via its existing --gt-anchor code path, then transfers the object's stabilized depth back to the hand with the grasp-preserving common-delta rule — inverting the current hand->object error flow; HORT and ForeHOI keep their networks and get a closed-form per-frame re-gauge (per-frame translation + one global scale aligning their predicted points to D~_t inside their own masks). (4) Oracle-sandwich evaluation: every method x {monocular, +substrate, +GT-depth} on HOI4D GT align_depth, reporting accuracy (object size/depth/center error, hand MPJPE) and a stability suite (apparent-size flicker, depth wiggle ratio = z-path/net-displacement, acceleration jitter, grasp consistency), against substrate ablations (per-frame depth + temporal filtering, consistent-video-depth/pose engines like ViPE or Video-Depth-Anything, COLMAP+MVS) to isolate what GS BA specifically adds (joint camera+scale, amodal support depth, uncertainty). Headline number: fraction of the oracle gap closed with zero changes to the HOI methods.
- **Novelty Claim:** First causal, oracle-controlled demonstration that per-frame metric-gauge wobble — not pose estimation — dominates temporal instability in monocular 4D hand-object reconstruction, fixed upstream by a dynamic-masked static-scene Gaussian-splatting bundle adjustment whose background-fitted per-frame gauge corrections transfer to hand/object pixels and restore most of the GT-depth oracle's stability and accuracy in four unchanged SOTA pipelines — the hand-object analog of body-domain world grounding (TRAM/SLAHMR), which additionally requires amodal support-surface depth and grasp-preserving re-anchoring for the manipulated unknown object.
- **Target Venue:** CVPR (primary: diagnosis + system + oracle-sandwich benchmark on HOI4D is a 3D-vision analysis/systems paper; CoRL viable as backup only if a retargeting-to-robot downstream result is added, e.g. via the do-as-i-do/egoaero stack; ICLR is a poor fit — no learning contribution)
- **Key Assumptions:**
  - Gauge dominance transfers: monocular metric-depth error on dynamic hand/object pixels shares the per-frame global affine gauge of the static background, so background-fitted (a_t,b_t) corrects dynamic pixels too — directly verifiable per-pixel on HOI4D GT depth BEFORE building the system; fails if depth nets have region/category-dependent bias exceeding the gauge term.
  - Enough static, textured background is co-visible across frames (true for table-top/egocentric HOI like HOI4D and wild6) and hand+object masks do not dominate the frame; enough camera parallax for BA — a static camera degrades gracefully to multi-frame depth consensus, which already fixes gauge wobble.
  - The injection interface reaches each pipeline where it matters: proven for render_and_compare and do-as-i-do by the GT-depth harness (0% depth error, existing sensor-depth and --gt-anchor code paths); assumes output-side per-frame re-gauging of HORT/ForeHOI (no network changes) still counts as 'unchanged' and captures most of their placement error.
  - A single global metric scale exists and is recoverable (rigid background, constant pinhole intrinsics after GeoCalib); if the metric depth prior is globally biased, the substrate is consistent-but-uniformly-mis-scaled — stability gains persist but absolute accuracy is bounded by that bias unless object-at-rest support contact pins the scale.
  - Residual error under the GT-depth oracle (HaWoR hand-depth wobble; SAM-3D canonical-shape -21% size residual) is separable from gauge error and partly addressable by the object->hand anchor flip / grasp-preserving delta without touching MANO pose; if most jitter had survived the oracle, the causal claim inverts and the paper degrades to an analysis-only result.
  - Novelty survives the crowded-GS field: template-embedded HOI Gaussians (e.g. Interaction-Aware 4DGS, ICLR'26 submission) reconstruct hand+object WITH Gaussians, whereas this branch deliberately has none and contributes the substrate + causal decomposition; generic pose/depth engines (ViPE, Video-Depth-Anything, robust CVD) must be included as substrate baselines and matched/beaten for the GS-BA choice to be justified — otherwise the contribution reduces to the diagnosis + injection protocol, which remains the paper's core.
  - Compute fits the accepted budget: GS BA at roughly 10-40 min per 5s clip on one GPU is small against the ~3-8 h already spent per clip (do-as-i-do guided-diffusion tracker), making 'spend the per-clip budget on the substrate' credible.
  - Stability metrics (apparent-size flicker, wiggle ratio, acceleration jitter, grasp consistency) formalized from the repo's diagnostics will be accepted by reviewers as meaningful alongside standard accuracy metrics — they double as a contribution (a 4D-HOI stability protocol) but are not yet community-standard.

## Novelty check

- **Closest Works:**
  - **Title:** TRAM: Global Trajectory and Motion of 3D Humans from in-the-wild Videos
  - **Venue Year:** ECCV 2024
  - **Url:** https://arxiv.org/abs/2403.17346
  - **Relation:** The body-domain template this branch explicitly generalizes: masks dynamic humans out of SLAM/dense BA and derives metric motion scale from the static scene background, then feeds a (retrained) human regressor. Proves the 'background-derived gauge' mechanism works, but for bodies, no Gaussian splatting, no per-frame affine depth-gauge correction transferred to dynamic pixels, no unchanged-downstream-pipeline injection, no oracle decomposition.

  - **Title:** WHOLE: World-Grounded Hand-Object Lifted from Egocentric Videos
  - **Venue Year:** arXiv 2602.22209 (2026, venue TBD)
  - **Url:** https://arxiv.org/abs/2602.22209
  - **Relation:** Closest in problem statement (world-grounded hand+object from egocentric video), but it ASSUMES metric-SLAMed input (Aria HOT3D) rather than recovering the gauge from RGB, and it trains its own joint diffusion motion prior — i.e., it changes the HOI estimator instead of stabilizing the substrate under unchanged pipelines. Complementary, but reviewers will demand comparison/citation.

  - **Title:** ViPE: Video Pose Engine for 3D Geometric Perception (and MegaSaM, CVPR 2025)
  - **Venue Year:** NVIDIA tech report 2025 / MegaSaM CVPR 2025 (arXiv 2412.04463)
  - **Url:** https://research.nvidia.com/labs/toronto-ai/vipe/assets/paper.pdf
  - **Relation:** Generic engines that already output camera poses, intrinsics, and temporally consistent METRIC depth from dynamic in-the-wild video — they solve most of the proposed substrate without GS. This is the biggest novelty threat: if ViPE/MegaSaM-fed pipelines match the GS-BA substrate, the branch's system contribution collapses to the diagnosis + injection protocol. They must be included as baselines (the branch already plans this).

  - **Title:** DAS3R: Dynamics-Aware Gaussian Splatting for Static Scene Reconstruction
  - **Venue Year:** arXiv 2412.19584 (2024)
  - **Url:** https://arxiv.org/abs/2412.19584
  - **Relation:** Shows the core mechanism component already exists: static-scene 3DGS reconstructed from dynamic monocular video using dynamic masks (MonST3R confidence), with pose handling. Not metric, not HOI, no gauge transfer to dynamic pixels, no downstream task — but it means 'dynamic-masked static-scene GS BA' alone is not novel.

  - **Title:** Zero-shot Reconstruction of In-Scene Object Manipulation from Video
  - **Venue Year:** arXiv 2512.19684 (Dec 2025)
  - **Url:** https://arxiv.org/abs/2512.19684
  - **Relation:** Reconstructs hand-object manipulation consistent with a reconstructed scene point cloud in metric coordinates, explicitly tackling ambiguous hand-object depth via scene context. Overlaps the 'scene as metric anchor for HOI' idea, but builds one new pipeline rather than a substrate that upgrades four existing methods, and offers no causal gauge-vs-pose decomposition.

  - **Title:** HaWoR: World-Space Hand Motion Reconstruction from Egocentric Videos
  - **Venue Year:** CVPR 2025
  - **Url:** https://arxiv.org/abs/2501.02973
  - **Relation:** Uses adaptive dynamic-masked SLAM + a foundational metric depth model to fix camera scale for world-space HANDS — the hand-only precursor of the substrate idea, and simultaneously one of the jitter sources this branch diagnoses (the HaWoR anchor causing do-as-i-do's 2x scale swing). No object, no GS, no unchanged multi-pipeline injection.

  - **Title:** AGILE: Hand-Object Interaction Reconstruction from Video via Agentic Generation
  - **Venue Year:** SIGGRAPH 2026 (arXiv 2602.04672)
  - **Url:** https://arxiv.org/abs/2602.04672
  - **Relation:** Recent HOI system emphasizing consistent metric initialization and global object scale across the sequence via anchor-and-track (avoiding brittle SfM). Competes on the same failure mode (scale/metric inconsistency) but as yet another new end-to-end pipeline, not a substrate + causal analysis.

  - **Title:** Interaction-Aware 4D Gaussian Splatting for Dynamic Hand-Object Interaction Reconstruction
  - **Venue Year:** arXiv 2511.14540; withdrawn ICLR 2026 submission
  - **Url:** https://arxiv.org/abs/2511.14540
  - **Relation:** The crowded 'GS-for-HOI' direction the branch deliberately avoids: models hand+object WITH deformable Gaussians (plus a static-background stage). Confirms the branch's differentiation claim — this branch has no hand/object Gaussians and contributes the gauge substrate instead. Its withdrawal also hints the GS-HOI-reconstruction angle is contested.

  - **Title:** Video Depth Anything: Consistent Depth Estimation for Super-Long Videos
  - **Venue Year:** CVPR 2025 (arXiv 2501.12375)
  - **Url:** https://arxiv.org/abs/2501.12375
  - **Relation:** Representative of the consistent-video-depth family (also robust CVD, casualSAM) that fixes per-frame depth flicker generically. A mandatory substrate ablation: the branch must show GS BA's joint camera+global-scale+amodal-support depth beats temporally-smoothed monocular depth.

  - **Title:** Humans as Checkerboards: Calibrating Camera Motion Scale for World-Coordinate Human Mesh Recovery
  - **Venue Year:** arXiv 2407.00574 (2024)
  - **Url:** https://arxiv.org/abs/2407.00574
  - **Relation:** Body-domain evidence that camera/scale gauge calibration is a recognized, separable subproblem for world-coordinate reconstruction — supports the framing but also shows the 'gauge calibration as the fix' insight is published in the human-body domain.

- **Delta:** Every mechanism component exists separately — dynamic-masked SLAM/BA with background-derived metric scale (TRAM, HaWoR), static-scene 3DGS from dynamic video (DAS3R), metric camera+depth engines (MegaSaM, ViPE), per-frame affine depth alignment (standard in CVD work) — but no published work (1) recovers the full metric substrate (camera trajectory + one global scale + per-frame affine gauge corrections + amodal support depth) via static-scene GS bundle adjustment and transfers the background-fitted gauge to dynamic hand/object pixels, (2) injects it into FOUR unchanged SOTA HOI pipelines through their existing depth/anchor interfaces, or (3) runs the oracle-sandwich (monocular / +substrate / +GT-depth) causal decomposition showing gauge wobble — not pose error — dominates 4D HOI jitter, with a formalized stability protocol (apparent-size flicker, wiggle ratio, grasp consistency). The novelty therefore lives in the HOI-domain causal diagnosis + injection protocol + decomposition benchmark, not in the GS BA machinery itself. The load-bearing risk is exactly the one the branch names: if off-the-shelf ViPE or MegaSaM depth+pose fed through the same interfaces closes a similar fraction of the oracle gap, the paper's system contribution reduces to the analysis — still publishable as a CVPR analysis paper, but with a weaker method claim, so the GS-vs-ViPE ablation is make-or-break and should be run first.
- **Crowdedness:** moderate
- **Crowdedness Reason:** The surrounding rings are crowded and fast-moving: world-grounded HOI reconstruction saw WHOLE, AGILE, CHOIR, and Zero-shot In-Scene Manipulation all appear within roughly the last 8 months, and consistent metric depth/pose engines (MegaSaM, ViPE, Video Depth Anything) plus dynamic-masked static GS (DAS3R) are mature commodity infrastructure. But the exact niche — an oracle-controlled causal decomposition of HOI jitter into gauge vs pose vs shape error, fixed upstream and fed to unchanged existing pipelines — is empty; no analysis-style paper of this kind exists for hand-object reconstruction. Sparse niche inside a crowded neighborhood nets out to moderate, with the main erosion vector being that WHOLE-style world-grounded HOI systems could absorb the framing within one review cycle.
- **Already Done:** False
- **Search Confidence:** verified-by-search

## Pressure test

- **Failure Modes:**
  - MAKE-OR-BREAK BASELINE KILLS THE METHOD: ViPE / MegaSaM / Video-Depth-Anything already output temporally consistent metric depth + camera poses from dynamic monocular video. If feeding any of them through the same injection interface closes a similar fraction of the oracle gap (likely, since the substrate spec is exactly what they produce), the GS-BA machinery is dead weight and the paper collapses to 'we plugged an existing depth engine into existing pipelines' — an analysis paper with borderline-reject odds at CVPR. The branch names this risk itself but currently has zero evidence GS-BA wins.
  - GAUGE-TRANSFER EXTRAPOLATION FAILURE: the affine correction (a_t, b_t) is fitted only on static background pixels — in tabletop/egocentric HOI that is mostly a near-planar desk at a narrow depth range, making the 2-parameter regression ill-conditioned (a and b nearly collinear); applying it to near-field hand/object pixels is extrapolation outside the fitted depth range, and monocular depth nets have depth- and category-dependent bias (hands/skin are a known failure category). If the per-pixel verification pilot on HOI4D GT depth fails, the paper's one mechanism dies. This should be experiment #1, before any GS code is written.
  - TRIVIALITY ATTACK: the GT-depth oracle already proves 'better depth in → better output'; reviewers can argue the delta contribution is just monocular-video depth-estimation quality, which belongs to the video-depth literature, and the HOI part is plumbing. The causal gauge-vs-pose decomposition is the real novelty, but decomposition-only papers need breadth the current evidence lacks.
  - EVIDENCE BASE IS n=1: the entire root-cause diagnosis (0% error under GT depth, hand-anchor corr 0.775, 2x scale swing) comes from one kettle clip in compare/hoi4d/depth_eval.md. The oracle residuals already visible there (do-as-i-do -21% size from SAM-3D canonical shape; HaWoR hand-depth wobble surviving GT depth) mean the substrate ceiling is below 'fixed', and on other object categories the gauge fraction could be much smaller — the headline 'gauge dominates' claim could invert on a broader clip set.
  - 'UNCHANGED PIPELINES' FRAMING ERODES: 2 of 4 pipelines are not actually unchanged — HORT/ForeHOI need output-side per-frame re-gauging (post-hoc alignment, itself arguably a trivial baseline), and do-as-i-do needs the anchor flip + grasp-preserving delta rule (a real pipeline modification, already implemented as --gt-anchor). A careful reviewer will call the headline framing oversold, and two of the four 'SOTA baselines' (render_and_compare, do-as-i-do) are internal/reproduction stacks, not published SOTA.
  - EVAL MATRIX VS COMPUTE: 4 methods x (monocular / +substrate / +GT) x substrate ablations (ViPE, VDA, COLMAP+MVS, temporal filtering) x enough clips to be credible (>=20-50) explodes; do-as-i-do alone at ~3h per 5s clip means ~360+ GPU-hours for a 20-clip x 6-condition slice of one method on a single-GPU box. Pruning is possible but reviewers will notice do-as-i-do evaluated on 5 clips.
  - STABILITY METRICS + SMOOTHING BASELINE: the flicker/wiggle/grasp-consistency suite is homegrown; naive temporal smoothing of per-frame outputs can match the stability numbers cheaply, so differentiation must come from absolute accuracy — which is bounded by the metric prior's global bias unless the fragile support-plane-contact heuristic works. If MoGe-2/DA3 is globally biased on a scene, the substrate is consistent-but-wrong and accuracy gains vanish.
  - STATIC-SCENE GS BA FRAGILITY IN HOI: mask leakage is real (their own eval documents SAM3 locking onto the wrong kettle), the manipulated object transitions static->dynamic mid-clip, hands+object can dominate egocentric frames, and static-camera clips degenerate to multi-frame depth consensus where GS adds nothing — shrinking the regime where the proposed machinery matters.
  - FIELD VELOCITY: WHOLE, AGILE, and Zero-shot In-Scene Manipulation all appeared within ~8 months; a WHOLE-style world-grounded system with an RGB-only front end would absorb the framing within one review cycle, and by a Nov 2026 CVPR deadline reviewers will ask whether the substrate helps the newest SOTA rather than HORT/ForeHOI.
- **Scores:**
  - **Novelty:** 5.5
  - **Feasibility:** 7.5
  - **Crowdedness Inverse:** 4.5
  - **Venue Fit:** 6
  - **Overall:** 5.5
- **Unsound Assumptions:**
  - That a per-frame GLOBAL affine gauge fitted on background pixels transfers to dynamic near-field hand/object pixels — the cited evidence (uniform 0.77x scale, z-only teleports) is consistent with an affine gauge but does not prove affine sufficiency, and the fit is ill-conditioned on narrow-depth-range tabletop backgrounds; depth-net bias is region/depth-dependent.
  - That substrate depth behaves like sensor depth at the injection interface — GT align_depth is noisy-but-unbiased, GS-rendered/corrected depth is smooth-but-biased; pipelines proven exact under GT injection may respond very differently to correlated bias (e.g. render_and_compare anchoring to a uniformly shifted surface).
  - That the pipelines count as 'unchanged' — HORT/ForeHOI output re-gauging and the do-as-i-do anchor flip are modifications; the clean causal framing ('same methods, new input') is only literally true for render_and_compare.
  - That one global metric scale is recoverable from cross-frame consensus of the depth prior — if the prior is globally biased per-scene, absolute accuracy (the metric that distinguishes this from smoothing baselines) is unrecoverable; the object-at-rest support-plane fix is a heuristic with its own failure modes.
  - That the n=1 kettle-clip oracle result generalizes across objects, grasps, and scenes — the -21% shape residual and surviving HaWoR hand wobble already show gauge is not the whole story even on that clip.
  - That GS bundle adjustment specifically (vs commodity ViPE/MegaSaM/VDA) is needed — unproven, self-identified as make-or-break, and the prior expectation should be that commodity engines get most of the way.
  - That reviewers will accept the homegrown stability protocol (flicker, wiggle ratio, grasp consistency) as headline metrics without community precedent.
- **Feasibility:**
  - **Data:** Adequate for a pilot, thin for a paper. HOI4D align_depth + the injection harness exist and are validated in-repo (real_perception.py sensor-depth path, --gt-anchor), wild6 gives qualitative in-the-wild results. But quantitative GT currently = 1 clip, 1 dataset; a credible submission needs 20-50 HOI4D clips across categories plus ideally a second RGB-D HOI dataset (DexYCB/H2O) — obtainable, but the diagnosis must survive that scale-up.
  - **Compute:** GS BA at 10-40 min per 5s clip fits a single-GPU Vast box easily; the real bottleneck is the evaluation matrix — do-as-i-do costs ~3h per 5s clip, so 4 methods x ~6 conditions x 20+ clips demands aggressive pruning (run do-as-i-do on a subset only). Roughly 4-8 weeks single-GPU with careful scheduling; feasible but the matrix, not the method, is the cost.
  - **Evaluation:** The strongest aspect: the oracle-sandwich protocol is well-defined, the injection interfaces are already proven exact under GT depth, and the load-bearing assumption (gauge transfer to dynamic pixels) is directly verifiable per-pixel on GT depth BEFORE building anything — a rare property. Risks: homegrown stability metrics need smoothing baselines to be meaningful, absolute-accuracy claims hinge on global-scale recoverability, and the make-or-break GS-vs-ViPE/MegaSaM ablation determines whether there is a method contribution at all.
- **Verdict:** maybe
- **Verdict Reason:** As a research move this branch is excellent — cheap, maximally falsifiable, and it doubles as the ablation that de-risks every sibling branch. As a standalone CVPR paper it is fragile: the mechanism is assembled entirely from commodity parts, the most probable experimental outcome ('commodity metric depth+pose engines close most of the oracle gap too') demotes it to an analysis paper resting on homegrown metrics, the 'unchanged pipelines' framing is only literally true for 1 of 4 methods, and the entire causal diagnosis currently rests on a single 5-second clip. Verdict: run the two gating pilots first — (1) per-pixel verification on >=10 HOI4D clips that background-fitted affine gauges correct hand/object-region depth error, (2) ViPE/MegaSaM substrate fed through the same interfaces. If (1) holds and GS-BA measurably beats (2) — e.g. via amodal support depth or static-camera cases — pursue as a CVPR analysis+system paper; if (1) holds but (2) matches GS-BA, keep the substrate as infrastructure and fold the causal decomposition into a stronger sibling branch; if (1) fails, drop and be grateful it cost two weeks.

## Scores

- **Id:** b6-gs-scale-depth-anchor
- **Title:** Attack the root cause only: GS-based bundle adjustment of camera + metric scale + background to stabilize depth/scale, feeding existing HOI pipelines
- **Overall:** 5.5
- **Verdict:** maybe (strongest: highest feasibility, empty niche, cheapest falsification; run the two gating pilots first)
