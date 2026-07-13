# T4 — learned model-free tracker bake-off: literature pass + deferral

> **In plain terms.** This document is a literature survey and feasibility triage, not a set of
> results. It surveys candidate learned object-tracking methods (FoundationPose, Any6D,
> BundleSDF, and others) and checks which ones could actually be built and run on this project's
> GPUs. Most candidates needed incompatible or unbuildable software; only two (ForeHOI and HORT)
> were judged feasible within budget and picked for the real comparison, whose results live in
> [T4_RESULTS.md](T4_RESULTS.md). No method is integrated here — this is reconnaissance only.
> Method codes decoded in [GLOSSARY.md](../../../GLOSSARY.md).

Ran the literature pass (2026-07); **did not integrate** any method — deferred
by user decision after the T1–T3 checkpoint (the campaign already delivered its
main win on surface accuracy; a learned bake-off needs multi-hour Blackwell
sm_120 conda-env builds with custom CUDA ops, historically failure-prone on
this box, for a method that solves a slightly different problem and is not built
for this degree of egocentric hand occlusion).

## Candidate table

| method | venue | inputs | code | fit / risk on this task |
|---|---|---|---|---|
| **FoundationPose** | CVPR'24 (NVlabs) | RGB-D + reference recon OR CAD | github NVlabs/FoundationPose | most mature; model-free path needs a reference-view neural recon; heavy custom CUDA ops → sm_120 build risk high |
| **Any6D** | CVPR'25 | single RGB-D anchor → 6D pose + size | github taeyeopl/Any6D | closest to our single-anchor setup; newest; build cost/risk unknown |
| **6DOPE-GS** | Dec'24 | RGB-D online | — | Gaussian-splatting online track+recon; real-time; GS on sm_120 buildable but untested |
| **GSGTrack** | Dec'24 | RGB video | arxiv 2412.02267 | RGB-only GS pose track |
| **RGBTrack** | 2025 | RGB (depth-free) | arxiv 2506.17119 | builds on FoundationPose, depth-free |
| **BundleSDF** | CVPR'23 | RGB-D | github | original plan target; neural online track+recon; known-heavy deps |
| **Point2Pose** | 2026 | RGB-D | arxiv 2604.10415 | 2D point-tracker + online TSDF, explicitly targets OCCLUSION recovery — most relevant to hand occlusion if pursued |

## Feasibility investigation (2026-07-09, resumed at user request)

Deep-investigated the 5 cloned repos + externals. **"One environment for all"
is infeasible** — they span cu118/cu121/cu130 with conflicting custom CUDA
extensions, none pre-built for sm_120. Per-method verdict:

| method | I/O | Blackwell revival | verdict |
|---|---|---|---|
| **HORT** | mono single-image → object cloud (hand-relative, no metric scale) | needs consistent Blackwell torch+torchvision+pytorch3d (its model imports pytorch3d); reuse sam3d5090 stack | **FEASIBLE** (~2 h, qualitative) |
| **ForeHOI** | HOI video → mesh + FoundationPose pose; built-in HOT3D shape-eval | reuse sam3d5090 (spconv/kaolin/pytorch3d ready) + rebuild FP `mycuda` for sm_120 + HF weight DL | **FEASIBLE** (~3 h; most comparable) |
| **do-as-i-do** | RGB video → mesh + per-frame pose (maps to our format; reusable wild6 runner) | 6–8 custom CUDA extensions rebuilt for sm_120 (pytorch3d, kaolin, nvdiffrast, spconv, torch-scatter, diff-gaussian-raster + DROID-SLAM/lietorch) — none pre-built, incl. in rc5090 | HARD (multi-day) |
| **HOLD** | video → per-video VolSDF optim (hours/clip) | kaolin has no Blackwell wheel; MPI-walled weights (HTTP 401); unbuilt COLMAP/SAM-Track preprocessing | infeasible-in-budget |
| **EasyHOI** | single image only | 5 custom-CUDA packages + unbuilt LISA (26 GB) + affordance-diffusion env | infeasible-in-budget |
| **FoundationPose** (standalone) | RGB-D + reference → per-frame pose | `mycpp`/`mycuda` CUDA sm_120 rebuild | HARD |

Key resource that makes HORT/ForeHOI feasible: the **sam3d5090** env already has
working Blackwell builds of torch 2.8+cu128, pytorch3d 0.7.8, spconv 2.3.8,
kaolin 0.18.0, gsplat — the hard rebuild others need is already done there.
Selected scope (user, 2026-07-09): **revive ForeHOI + HORT**, compare to
icpjgr; skip HOLD/EasyHOI/do-as-i-do. Detailed per-method reports:
`.superpowers/sdd/investigate-{forehoi,hort,hold-easyhoi,daid-fp}.md`.

## If resumed

Integration pattern: follow the SAM-3D subprocess wiring
(`run_object_sam3d` in `backends/real_perception.py` + a
`backend.object_pose: <method>` branch in stage 4) — a new env + a CLI entry
reading frames/masks/depth/K from the run dir and writing `poses[T,4,4]`,
evaluated with `run_batch.py --arm <method>` against the frozen bench6 + gate.
Best first bets by fit: **Any6D** (single-anchor RGB-D matches our setup) or
**Point2Pose** (occlusion-native). Budget several hours per method; time-box
the env build and fall back if it exceeds it.
