# Method-agnostic GS temporal refinement layer: take ANY existing HOI method's per-frame output and re-optimize it under a joint multi-frame Gaussian rendering loss

**Branch ID:** `b5-gs-universal-refiner`
**Overall score:** 5.5
**Verdict:** maybe (unoccupied slot, but load-bearing claims at risk: HORT out of basin, monocular scale-depth null space; 2-3 week pilot required)

## Digest (card summary)

**TL;DR:** De-jitter any HOI method's output by lifting it into template-bound Gaussians and re-optimizing only pose trajectories against all frames, betting joint temporal photometric coupling is what's missing.

**Why:**
- All four baselines jitter identically: object depth slaved to hand (r=0.775), wiggle 7.9x, flicker 1.75x.
- Backbone swaps don't help (flicker 1.75x to 1.76x); smoothing never consults pixels. Bet: joint rendering does.
- Four wired baselines, HOI4D GT-depth harness, jitter metrics already built; the GS-refiner slot is unoccupied.

**Pipeline:**
1. **Canonicalize baseline outputs** — Lift any method's output into MANO trajectory plus one canonical rigid object with SE(3) track. — `any baseline's per-frame output -> MANO trajectory + object SE(3) track`; reuses compare/scenes/*.npz normalized scene contract (already prototyped)
2. **Register HORT clouds** — Multi-frame rigid registration merges per-frame point clouds into one canonical object, initializing its trajectory. — `HORT per-frame colored clouds -> canonical object mesh + SE(3) init`; reuses HORT baseline outputs
3. **Bind and freeze Gaussians** — Rig ~25k hand Gaussians via MANO LBS, bind object Gaussians rigidly; fit appearance briefly, freeze. — `canonical geometry + frame subsample -> frozen-appearance rigged Gaussians`; reuses differentiable GS rasterizer
4. **Pose-only joint refinement** — Optimize spline trajectories against all-frame composited photometric, silhouette, contact, and smoothness losses (5-10 min/clip). — `frozen Gaussians + full video -> refined smooth hand/object trajectories`; reuses SAM2/SAM3 masks already produced inside all four pipelines
5. **Temporal-stability benchmark** — Scale GT-depth harness to ~50 rigid HOI4D clips; add DexYCB/HO3D GT-motion acceleration error. — `clips + GT -> wiggle, flicker, depth-correlation, contact-persistence scores`; reuses HOI4D GT-depth harness + jitter diagnostics (depth corr 0.775, wiggle 7.9)
6. **Cross-baseline delta evaluation** — Report refiner as uniform delta on four baselines versus TOCH, GeneOH, SmoothNet, and BIGS. — `raw vs refined trajectories -> per-baseline stability + accuracy deltas`; reuses all four wired baselines (render_and_compare, do-as-i-do, ForeHOI, HORT)

**What's new:** Unlike TOCH (ECCV'22), which refines HOI sequences via a learned geometric prior without touching pixels, this re-optimizes any method's poses photometrically, so it can correct depth/scale error.

**Top risks:**
- ⚠ Monocular scale-depth null space: rigidity supervises only relative jitter, not absolute depth/scale.
- ⚠ HORT's -81% depth error sits outside the photometric basin; 4/4 claim degrades to 3/4.
- ⚠ Cheap smoothers (GeneOH, median+MA) may capture most jitter gains; accuracy deltas within noise.

**Kill test:** 2-3-week pilot: refine RC, ForeHOI, do-as-i-do on 5-10 rigid HOI4D clips; score GT depth/size error, MPJPE/Chamfer versus SmoothNet/GeneOH. Go only if non-circular accuracy improves, not just jitter; else fold into joint-optimization branch.

**If it works:** Every baseline's jitter metrics improve without accuracy regression on HOI4D/DexYCB, plus depth/scale corrections that evidence-free smoothers (SmoothNet, GeneOH) provably cannot make.

## Branch definition

- **Id:** b5-gs-universal-refiner
- **Title:** Method-agnostic GS temporal refinement layer: take ANY existing HOI method's per-frame output and re-optimize it under a joint multi-frame Gaussian rendering loss
- **Angle:** Plug-in module rather than new pipeline: initialize template-embedded Gaussians from the output of an arbitrary HOI method (render_and_compare, do-as-i-do, ForeHOI, HORT), freeze appearance quickly, then re-optimize only the pose/trajectory parameters against all frames' photometry + smoothness. Contribution = a universal test-time refiner evaluated across all 4 baselines, plus the jitter-metric suite (depth-correlation, wiggle ratio, acceleration energy, contact flicker) formalized as a benchmark.
- **Core Hypothesis:** A shared GS refinement stage recovers most of the stability of full joint optimization (b2/b3) at a fraction of the cost, and improves EVERY baseline's jitter metrics on the HOI4D harness without hurting accuracy — demonstrating that the missing ingredient across the field is joint photometric temporal coupling, not better per-frame estimators.
- **Why Distinct:** It is evaluated as a delta on N existing methods (a refiner + benchmark paper), not as a standalone reconstruction method; requires no reconstruction-from-scratch and directly weaponizes the user's diagnostic experiments and metrics as the paper's motivation section. Novelty search target: 'test-time Gaussian-splatting refinement of hand/object pose from video'.
- **Builds On:**
  - all 4 user baselines
  - user's jitter diagnostics (depth corr 0.775, wiggle 7.9)
  - HOI4D GT-depth harness
  - differentiable GS rasterizer

## Clarification

- **Problem Statement:** Monocular 4D hand-object-interaction reconstruction is temporally broken in a way per-frame metrics do not measure. Across four heterogeneous state-of-the-art pipelines run on a common harness — feed-forward per-image (HORT), feed-forward video (ForeHOI), compositional optimization (render_and_compare), and generative 6-DoF tracking (do-as-i-do) — hand and object trajectories are estimated per frame with only heuristic temporal coupling, producing the same failure family: object depth slaved to noisy hand depth (measured correlation 0.775), hand depth path-length 7.9x its true net displacement (wiggle ratio), 1.75x frame-to-frame apparent-size flicker of a provably rigid object, and contact that flickers on/off. Two controlled experiments show the fix is not better per-frame components: swapping the depth backbone (MoGe -> DA3-metric) leaves the flicker unchanged (1.75x -> 1.76x) because the error enters through the anchoring structure, and signal-space smoothing (median+moving-average) suppresses jitter (flicker 0.018 -> 0.005) but cannot correct systematic depth/scale error because it never consults the pixels. What is missing is (1) a method-agnostic mechanism that restores temporal coherence using the video itself as evidence — without redesigning, retraining, or re-running any estimator — and (2) an evaluation protocol that actually scores temporal stability, since HOI benchmarks report per-frame MPJPE/Chamfer/F-score that are blind to jitter. We propose both: a plug-in test-time Gaussian-splatting refinement layer that consumes any HOI method's output and re-optimizes only its pose trajectories under a joint all-frame rendering loss, and a temporal-stability benchmark (on the HOI4D GT-depth harness plus GT-motion datasets) that quantifies the field-wide problem and the refiner's uniform improvement of it.
- **Technical Core:** A bind-then-refine test-time optimizer with three components. (1) Canonicalization front-end: a normalized scene contract (already prototyped as compare/scenes/*.npz — hand verts/MANO params, canonical object mesh + per-frame SE(3), or per-frame colored point clouds, camera intrinsics, contact masks, all metric OpenCV camera frame) lifts any method's output to one state: MANO trajectory {theta_t, beta, R_t^h, tau_t^h, per-clip hand scale s_h} + one canonical rigid object with SE(3) trajectory {T_t^o} and global scale s_o. Mesh-based methods map directly; point-cloud-only methods (HORT) get a canonical object by robust multi-frame registration of the per-frame clouds, which also initializes T_t^o. (2) Template-embedded Gaussian binding, poses frozen (Stage A, ~2k iters, minutes): ~15-25k hand Gaussians parameterized in MANO face-tangent frames and driven by LBS so rendered pixels are differentiable w.r.t. (theta_t, beta, s_h); ~10-30k object Gaussians rigidly bound to the canonical geometry, transformed by s_o*T_t^o. Appearance (SH color, opacity, scale) is initialized from the method's own textured mesh / colored points and fit briefly over a frame subsample with a per-frame affine exposure code, then FROZEN so it cannot absorb pose error. (3) Pose-only joint refinement over all frames (Stage B, ~2-4k Adam iters on 8-16-frame minibatches, ~5-10 min/clip on one consumer GPU): trajectories parameterized on a low-pass cubic B-spline/DCT basis (knots every ~5 frames; smoothness partly by construction, variable count O(T/k)), optimized against L = masked L1+D-SSIM photometric on the composited hand+object rasterization (compositing gives occlusion-correct gradients through the grasp, exactly where mask-only losses fail) + per-part silhouette vs the pipelines' existing SAM2/SAM3 masks + optional rendered-depth consistency (GT depth on the harness, metric mono-depth in the wild) + acceleration energy on MANO joints and se(3) log of T_t^o + camera-invariant smoothness on the hand->object relative pose (grasp consistency under camera shake; world frame via HaWoR/HOI4D cameras when available) + contact persistence (hinge on contact-vertex distance to a precomputed object SDF in baseline-contact frames) and non-penetration + a robust Geman-McClure trust region to the baseline trajectory so weakly-observed frames revert to the initialization instead of drifting. The key observability mechanism: because ONE fixed-size canonical object must photometrically reproject in EVERY frame, the 1/z apparent-size flicker that the diagnostics measured becomes a rendering residual with a nonzero gradient w.r.t. depth — joint multi-frame rigidity converts the jitter diagnostic into supervision, which is precisely what evidence-free smoothing cannot do. Evaluation: refiner reported as a delta on all four baselines — temporal suite (accel error mm/s^2 vs GT motion on DexYCB/HO3D, wiggle ratio, depth-anchor correlation, apparent-size flicker, contact-persistence) plus accuracy suite (MPJPE, Chamfer, object size/depth error on the HOI4D GT-depth harness, scaled from 1 to ~50 clips) — against OneEuro/Savitzky-Golay/median-MA filters, a learned keypoint smoother, a silhouette-only ablation (no photometric term), and from-scratch per-scene GS reconstruction (BIGS/GHOST) as the cost/quality upper bound; hypothesis: jitter improves for every baseline with no accuracy regression, and monocular depth/scale error shrinks where smoothing provably cannot shrink it.
- **Novelty Claim:** The first method-agnostic test-time refinement layer for monocular 4D HOI — it lifts ANY existing method's per-frame hand/object output into template-embedded (MANO-rigged + rigid-bound) Gaussians, freezes appearance, and re-optimizes only the pose trajectories under a joint all-frame composited rendering loss, validated as a uniform stability-plus-accuracy delta across four heterogeneous pipelines together with a temporal-stability benchmark — distinct from from-scratch per-scene GS-HOI reconstruction (BIGS/GHOST/GraG), from evidence-free keypoint-space smoothers, and from static single-body GS pose estimation (iComMa/6DGS).
- **Target Venue:** CVPR (refiner + cross-method benchmark is a classic CVPR shape; fallback ICCV/3DV; a CoRL/ICRA spin-off exists if the stabilized trajectories are shown to make do-as-i-do-style robot retargeting executable)
- **Key Assumptions:**
  - Baseline outputs land inside the photometric basin of convergence: errors are mostly high-frequency depth/pose noise (as the diagnostics show for RC/daid/ForeHOI), not gross misplacement; for grossly wrong initializations (e.g., HORT's -81% depth) the refiner must rely on multi-frame rigidity + relaxed trust region, and pure monocular scale ambiguity may remain unresolvable without a metric anchor (hand-size prior beta/s_h or metric depth).
  - The object is rigid with constant size (the mechanism converts 1/z size-flicker into depth gradients ONLY under rigidity); articulated/deformable HOI4D categories are out of scope or need per-part extension.
  - Photometric constancy holds well enough inside masks after robust losses + per-frame exposure codes; heavily textureless or specular objects degrade the photometric term toward silhouette-only (an ablation must quantify this floor).
  - Reliable per-part hand/object masks exist (SAM2/SAM3 already run inside all four pipelines, including the point-prompt fix for multi-instance scenes) and mutual-occlusion regions are handled by compositing, not by mask trust.
  - Camera intrinsics are known per method and camera motion can be factored (HaWoR/HOI4D camera poses, or the camera-invariant grasp-relative smoothness term) so temporal priors are imposed in a frame where they are valid.
  - A short appearance fit initialized from the method's own textured geometry, then frozen, is sufficient to prevent appearance from absorbing pose error (the main entanglement risk; mitigated by fitting on a frame subsample and never co-optimizing appearance and pose).
  - Point-cloud-only outputs (HORT) can be canonicalized by multi-frame rigid registration well enough to define one object; failure here reduces the refiner's coverage claim from 4 baselines to 3.
  - The HOI4D GT-depth harness scales from the current single kettle clip to ~50 clips (plus DexYCB/HO3D for GT-motion acceleration error) so the benchmark claim is statistically defensible.
  - A minutes-per-clip test-time budget (~5-10 min on one RTX-4090-class GPU, vs the existing 3 h do-as-i-do budget) is acceptable for the 'refinement layer' framing and is achievable with ~50k Gaussians and spline-parameterized trajectories.
  - Reviewers accept the framing that from-scratch GS-HOI pipelines (BIGS/GHOST/GraG, CVPR'25-'26) are complementary upper-bound comparisons rather than prior art for a method-agnostic pose-only refiner evaluated as a delta on existing methods.

## Novelty check

- **Closest Works:**
  - **Title:** TOCH: Spatio-Temporal Object-to-Hand Correspondence for Motion Refinement
  - **Venue Year:** ECCV 2022
  - **Url:** https://arxiv.org/abs/2205.07982
  - **Relation:** Closest in framing: a method-agnostic refiner that takes noisy hand-object interaction sequences from any tracker and corrects them. But it is a learned data prior operating in geometry/correspondence space (TOCH fields + temporal autoencoder) — it never consults the video pixels, refines the hand only (object pose assumed given/clean), and cannot fix systematic depth/scale error, which is exactly the gap this branch attacks.

  - **Title:** GeneOH Diffusion: Towards Generalizable Hand-Object Interaction Denoising via Denoising Diffusion
  - **Venue Year:** ICLR 2024
  - **Url:** https://arxiv.org/abs/2402.14810
  - **Relation:** Generalizable HOI trajectory denoiser applicable to arbitrary noisy inputs — same 'refine any method's output' ambition. Again evidence-free: a contact-centric learned prior with no photometric/image term, so it regularizes toward plausibility rather than toward what the video actually shows. Strong baseline the paper must compare against.

  - **Title:** SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos
  - **Venue Year:** ECCV 2022
  - **Url:** https://arxiv.org/abs/2112.13715
  - **Relation:** Establishes the exact paper shape (plug-in temporal refiner evaluated as a uniform delta across many backbones, with jitter/accel metrics) — but for human keypoints, in signal space, with no image evidence. This branch is essentially 'SmoothNet-shape paper, but photometric and for HOI'; SmoothNet is the canonical evidence-free baseline the clarification already positions against.

  - **Title:** GS-CPR: Efficient Camera Pose Refinement via 3D Gaussian Splatting
  - **Venue Year:** ICLR 2025
  - **Url:** https://arxiv.org/abs/2408.11085
  - **Relation:** Proves the core mechanism is publishable: test-time pose refinement of other methods' (APR/SCR) coarse estimates by rendering a frozen 3DGS model, incl. exposure-adaptive module (paralleling the branch's frozen-appearance + exposure codes). But it refines a static camera pose per query image — no articulated hands, no rigid-object trajectory, no multi-frame joint coupling.

  - **Title:** 6DOPE-GS: Online 6D Object Pose Estimation using Gaussian Splatting
  - **Venue Year:** ICCV 2025
  - **Url:** https://openaccess.thecvf.com/content/ICCV2025/papers/Jin_6DOPE-GS_Online_6D_Object_Pose_Estimation_using_Gaussian_Splatting_ICCV_2025_paper.pdf
  - **Relation:** GS-based 6-DoF object pose tracking with joint reconstruction — covers the rigid-object half of the mechanism (photometric pose gradients through a GS model over video), but online single-object tracking from scratch, no hand, no MANO/LBS, and not a refiner of other methods' outputs.

  - **Title:** BIGS: Bimanual Category-agnostic Interaction Reconstruction from Monocular Videos via 3D Gaussian Splatting
  - **Venue Year:** CVPR 2025
  - **Url:** https://arxiv.org/abs/2504.09097
  - **Relation:** From-scratch per-scene GS-HOI reconstruction (MANO-anchored hand Gaussians + object Gaussians + SDS). Overlaps in representation (template-embedded hand GS, rigid object GS, photometric optimization of poses) but is a standalone hours-scale reconstruction pipeline, not a fast method-agnostic pose-only refinement layer; the branch correctly frames it as upper-bound comparison, and reviewers will demand that comparison.

  - **Title:** GHOST: Fast Category-agnostic Hand-Object Interaction Reconstruction from RGB Videos using Gaussian Splatting
  - **Venue Year:** arXiv 2026 (2603.18912)
  - **Url:** https://arxiv.org/pdf/2603.18912
  - **Relation:** Fast GS-based HOI reconstruction from RGB video — narrows the speed advantage the refiner claims over from-scratch GS pipelines. Still reconstruction-from-scratch, not a delta-on-N-baselines refiner, but its existence pressures the '5-10 min refinement vs hours reconstruction' selling point.

  - **Title:** Interaction-Aware 4D Gaussian Splatting for Dynamic Hand-Object Interaction Reconstruction
  - **Venue Year:** arXiv Nov 2025 (2511.14540)
  - **Url:** https://arxiv.org/abs/2511.14540
  - **Relation:** Verified by fetch: from-scratch 4DGS HOI reconstruction with deformation fields and smoothness regularizers, no object priors, multi-view input. Shares 'temporal coherence via joint GS optimization' but does not consume or refine existing HOI methods' outputs and is not template/MANO-parameterized.

  - **Title:** EasyHOI: Unleashing the Power of Large Models for Reconstructing Hand-Object Interactions in the Wild
  - **Venue Year:** CVPR 2025
  - **Url:** https://arxiv.org/abs/2411.14280
  - **Relation:** Prior-guided test-time optimization on top of off-the-shelf foundation-model outputs — same 'optimize the outputs of existing components' philosophy, but single-image, per-frame, mask/physics-based (no GS, no multi-frame photometric coupling, no temporal claim).

  - **Title:** Hand-Centric Motion Refinement for 3D Hand-Object Interaction via Hierarchical Spatial-Temporal Modeling
  - **Venue Year:** AAAI 2024
  - **Url:** https://arxiv.org/abs/2401.15987
  - **Relation:** Learned refinement of perturbed HOI motions (explicitly motivated by jitter and inconsistent contact from trackers) — another member of the evidence-free learned-refiner family; overlaps in motivation and metrics, not in mechanism.

  - **Title:** DeepSimHO: Stable Pose Estimation for Hand-Object Interaction via Physics Simulation
  - **Venue Year:** NeurIPS 2023
  - **Url:** https://arxiv.org/pdf/2310.07206
  - **Relation:** Improves stability of HOI pose estimates via a physics-simulation feedback stage — a different (physics, per-frame-grasp) notion of 'stability' than temporal jitter, but a related add-on-refinement-stage precedent for HOI.

  - **Title:** DyTact: Capturing Dynamic Contacts in Hand-Object Manipulation
  - **Venue Year:** arXiv 2025 (2506.03103)
  - **Url:** https://arxiv.org/pdf/2506.03103
  - **Relation:** GS/2DGS-based dynamic hand-object contact capture with MANO-anchored dynamic articulation — further evidence the template-embedded-GS-for-HOI representation itself is not novel; again a per-scene capture method, not a method-agnostic refiner or benchmark.

- **Delta:** No found work occupies the intersection this branch claims: (1) a canonicalization layer that lifts ANY monocular HOI method's per-frame output (mesh- or point-cloud-based) into template-embedded Gaussians, (2) appearance fit-then-frozen, pose-trajectory-only joint ALL-frame composited photometric re-optimization at test time, and (3) evaluation as a uniform stability+accuracy delta across 4 heterogeneous baselines plus a formalized HOI temporal-stability benchmark. Existing method-agnostic HOI refiners (TOCH, GeneOH, AAAI'24 hand-centric) are learned priors that never consult the pixels — they cannot correct systematic depth/scale error, which is the branch's central mechanism (multi-frame rigidity turning 1/z size-flicker into depth gradients). Existing GS test-time pose refinement (GS-CPR, iComMa, 6DGS, 6DOPE-GS) handles a static camera or a single rigid object, not an articulated MANO hand + object composite with occlusion-correct gradients, and does not refine other methods' outputs across baselines. Existing GS-HOI video work (BIGS, GHOST, DyTact, Interaction-aware 4DGS) reconstructs per-scene from scratch and is not a plug-in delta on other estimators. The benchmark half is thinner novelty: accel/jitter metrics are standard (VIBE/SmoothNet lineage) and some HOI video papers already report Jitter/Accel/MPJVE; the genuinely new metrics are the HOI-specific ones (hand-object depth-anchor correlation, apparent-size flicker, wiggle ratio, contact persistence). Net: the mechanism is a novel composition of individually established pieces — the paper's defensibility rests on the cross-method uniform-improvement result and the depth/scale-correction claim that evidence-free smoothers provably cannot match, both of which must be shown against TOCH/GeneOH/SmoothNet baselines, not just filters."
- **Crowdedness:** moderate
- **Crowdedness Reason:** The two surrounding neighborhoods are individually busy — GS-based HOI/hand reconstruction (BIGS CVPR'25, GHOST'26, DyTact'25, Interaction-aware 4DGS'25, HOLD-descendants) and HOI motion refinement/denoising (TOCH, GeneOH, AAAI'24, DeepSimHO, plug-and-play jitter networks) — and GS-as-pose-refiner is established for cameras/objects (GS-CPR, iComMa, 6DGS, 6DOPE-GS). But the specific slot (method-agnostic test-time photometric GS refiner evaluated as a delta across HOI pipelines, plus temporal-stability benchmark) appears unoccupied as of this search. Risk is convergence: all ingredients are published and popular, so the window is real but likely short; a reviewer could also frame it as SmoothNet-shape + GS-CPR-mechanism applied to HOI, so the cross-baseline empirical result must carry the paper.
- **Already Done:** False
- **Search Confidence:** verified-by-search

## Pressure test

- **Failure Modes:**
  - Monocular scale-depth null space guts the headline claim: a uniform object rescale plus depth shift renders (near-)identically, so joint multi-frame rigidity only supervises RELATIVE depth jitter, not systematic depth/scale error. The 'corrects what smoothing provably cannot' claim then holds only when GT/metric depth enters the loss — at which point simple depth-fitting baselines compete and the mechanism story collapses to 'photometric smoothing'.
  - The universal (4/4 baselines) claim likely fails on HORT: -81% depth error and per-frame point clouds with 1.75x size flicker are outside the photometric basin; multi-frame rigid registration of those clouds is itself a hard unsolved subproblem. Non-convex photometric pose landscapes mean the refiner either diverges or the Geman-McClure trust region makes it a no-op — either way the paper's core selling point ('improves EVERY baseline') degrades to 3/4, and reviewers will notice the strongest test case is the one that fails.
  - Jitter metrics are gamed by trivial/learned smoothers — the user's own diagnostic shows median+MA already cuts flicker 0.018->0.005. If GeneOH/TOCH/SmoothNet capture most of the temporal gains at ~zero cost, the 5-10 min/clip optimizer is an expensive smoother, and the paper must win on accuracy deltas that may be within noise at ~50 clips x 4 baselines.
  - Evaluation circularity: the method optimizes acceleration energy, rigidity, and contact persistence, then reports acceleration, size-flicker, and contact-persistence metrics. Only GT-motion accel error (DexYCB/HO3D) and MPJPE/Chamfer/depth error are non-circular — the paper lives or dies on those, and the branch currently has zero evidence it improves them.
  - Appearance bake-in: Stage A fits texture/opacity at the baseline's WRONG jittered poses, then freezes it — pose error is baked into appearance and Stage B refines poses against a corrupted photometric target. Combined with low-texture skin, specular/textureless objects, and motion blur, photometric gradients may be too weak or biased exactly where refinement is needed (ablation floor risk acknowledged but not resolved).
  - Occlusion + symmetry unobservability: during manipulation the object is mostly hand-occluded, and common HOI4D objects (kettle, bottle, mug) are near rotationally symmetric — silhouette and photometric terms then carry almost no object-pose information in the very frames the method claims to stabilize; refined trajectories revert to the trust region (no measurable gain) or drift in the symmetry direction.
  - Benchmark half is largely pre-empted: Jitter/Accel/MPJVE are already standard in video pose papers (e.g., DanceHMR 2026), and HOI benchmarks with trajectory-quality metrics are appearing (DynaHOI). The genuinely new metrics (depth-anchor correlation, apparent-size flicker, wiggle ratio) are diagnostics, not a benchmark contribution reviewers weight heavily; the rigidity assumption also excludes HOI4D's articulated categories, shrinking scope and inviting cherry-picking critique.
  - Scaling compute is dominated by producing INPUTS, not the method: running 4 fragile third-party pipelines (do-as-i-do at ~3h/clip) over ~50 HOI4D clips plus DexYCB/HO3D means hundreds of GPU-hours and pipeline-babysitting before a single refinement result exists; a 1-clip harness to 50-clip benchmark is a large engineering leap the plan treats as an assumption.
  - Over-smoothing true dynamics: spline bases (knots every ~5 frames) plus acceleration penalties imposed in a possibly wrong frame (camera vs world; HaWoR factoring adds its own errors on egocentric HOI4D) can erase genuine fast manipulation motion, HURTING GT-motion accel error — the one non-circular temporal metric.
  - Reviewer framing risk and short window: the paper is compressible to 'SmoothNet paper shape + GS-CPR mechanism + BIGS representation, applied to HOI'. All ingredients are published and popular (HOIGS, GHOST, DyTact, 6DOPE-GS all 2025-26); a from-scratch-fast method like GHOST erodes the speed argument, and convergence within 6-12 months is plausible.
  - Baseline-fairness trap: TOCH refines hands only, GeneOH is trained on different noise distributions, SmoothNet is keypoint-space — every adaptation choice invites 'strawman baseline' or 'unfair comparison' rebuttal ammunition in either direction.
- **Unsound Assumptions:**
  - That multi-frame rigidity converts 1/z size flicker into supervision for ABSOLUTE depth/scale — it only constrains relative/temporal variation; the constant scale-depth offset lies in the photometric null space of a monocular rasterizer, so the central 'fixes what smoothing cannot' differentiator is overstated for in-the-wild (no metric anchor) settings.
  - That baseline outputs land inside the photometric basin of convergence — directly contradicted for HORT by the branch's own diagnostics (-81% depth), and untested for gross rotation errors on symmetric objects.
  - That appearance fit at noisy poses and then frozen cannot absorb pose error — fitting appearance FROM wrong-pose renders bakes the error in before the freeze; the mitigation (frame subsample, no co-optimization) does not address this initialization bias.
  - That reviewers will accept BIGS/GHOST/DyTact as 'complementary upper bounds' rather than prior art — the hand representation (MANO-anchored Gaussians, rigid object Gaussians, photometric pose optimization) is essentially identical; the only defensible delta is the method-agnostic-refiner protocol, which is an evaluation framing, not a mechanism.
  - That 'jitter improves with no accuracy regression' clears the CVPR bar — against learned denoisers that also improve jitter cheaply, the paper needs demonstrated accuracy GAINS (depth/scale/MPJPE) on non-circular metrics, a strictly stronger and unproven claim.
  - That the HOI4D harness scales from 1 kettle clip to ~50 clips (and to DexYCB/HO3D) with all 4 baselines runnable end-to-end — the largest unbudgeted engineering assumption in the plan.
  - That SAM2/SAM3 per-part masks are reliable in mutual-occlusion regions — compositing gives occlusion-correct gradients only if the underlying geometry is roughly right, which is circular for badly initialized frames.
- **Feasibility:**
  - **Data:** Public datasets (HOI4D, DexYCB, HO3D) suffice and the GT-depth harness exists — but only for 1 clip today. The real data cost is generating inputs: 4 heterogeneous third-party pipelines must be run at scale (do-as-i-do ~3h/clip), and the rigidity assumption restricts HOI4D to its rigid-object subset. Scaling to ~50 clips x 4 baselines x 2-3 datasets is doable but is weeks of pipeline-wrangling, not a script.
  - **Compute:** The refiner itself (5-10 min/clip, ~50k Gaussians, spline-parameterized poses on one 4090-class GPU) is realistic and the cheapest part. Dominant cost is baseline-output generation plus ablations and competitor baselines (TOCH/GeneOH/SmoothNet/BIGS-or-GHOST): plausibly 300-800 GPU-hours total — feasible on the user's Vast setup but nontrivial.
  - **Evaluation:** Weakest leg. Proposed temporal metrics are partly circular (the loss optimizes them) and partly pre-empted (Jitter/Accel/MPJVE standard). Non-circular evidence requires GT-motion accel error and accuracy metrics on DexYCB/HO3D + GT-depth harness, plus fair adaptations of TOCH/GeneOH/SmoothNet and a from-scratch GS upper bound whose code may not be released (GHOST is arXiv'26). Statistical defensibility across only 4 baselines and ~50 clips is marginal; the depth/scale-correction claim needs a carefully designed wild-vs-metric-anchor ablation to survive the scale-ambiguity attack.
- **Scores:**
  - **Novelty:** 5.5
  - **Feasibility:** 6
  - **Crowdedness Inverse:** 4
  - **Venue Fit:** 6
  - **Overall:** 5.5
- **Verdict:** maybe
- **Verdict Reason:** The slot (method-agnostic photometric GS refiner + cross-baseline HOI temporal evaluation) is genuinely unoccupied and the diagnostics-driven motivation is strong, but the paper's two load-bearing claims are both at risk: 'improves EVERY baseline' likely breaks on HORT, and 'corrects depth/scale error that smoothing cannot' collides with the monocular scale-depth null space unless a metric anchor is in the loss — reducing the contribution to an expensive smoother compressible to SmoothNet-shape + GS-CPR-mechanism. Pursue only after a cheap 2-3 week pilot on 5-10 rigid HOI4D clips (RC + ForeHOI + do-as-i-do) that must show: (a) non-circular accuracy gains (GT depth/size error, MPJPE/Chamfer) beyond GeneOH/SmoothNet, not just jitter reduction, and (b) a demonstrated (even partial) depth/scale correction with the mechanism ablated (photometric vs silhouette-only). If the pilot shows only jitter gains, drop or fold into the stronger joint-optimization branch; if it shows accuracy gains, this is a solid CVPR submission with a short window — move fast and pre-plan the HORT failure as an analyzed limitation rather than a hidden one.

## Scores

- **Id:** b5-gs-universal-refiner
- **Title:** Method-agnostic GS temporal refinement layer over any HOI method's per-frame output
- **Overall:** 5.5
- **Verdict:** maybe (unoccupied slot, but load-bearing claims at risk: HORT out of basin, monocular scale-depth null space; 2-3 week pilot required)
