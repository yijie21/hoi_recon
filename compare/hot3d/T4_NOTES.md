# T4 — learned model-free tracker bake-off: literature pass + deferral

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

## If resumed

Integration pattern: follow the SAM-3D subprocess wiring
(`run_object_sam3d` in `backends/real_perception.py` + a
`backend.object_pose: <method>` branch in stage 4) — a new env + a CLI entry
reading frames/masks/depth/K from the run dir and writing `poses[T,4,4]`,
evaluated with `run_batch.py --arm <method>` against the frozen bench6 + gate.
Best first bets by fit: **Any6D** (single-anchor RGB-D matches our setup) or
**Point2Pose** (occlusion-native). Budget several hours per method; time-box
the env build and fall back if it exceeds it.
