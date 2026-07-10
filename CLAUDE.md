## Orchestration workflow
You (Fable) are the orchestrator. Plan, decompose, synthesize.
Reasoning-heavy phases → deep-reasoner
Mechanical work → fast-worker
High-stakes decisions: task deep-reasoner (Opus) with the problem, think it through thoroughly, and synthesize a concise conclusion you can act on. Keep your own context lean.

## Where things stand (start here)
Object HOI reconstruction on HOT3D. Two best arms — a **placement-vs-rotation Pareto pair**:
`icpjgr` (rotation-robust `BEST_ARM`, `configs/real_forehoi_icp_joint_grasp.yaml`) and
`any6dp` (placement-optimal learned core, `configs/real_any6d.yaml`, wins chamfer 9/11).
The rotation/attitude/texture axis is a **proven dead end** — do not re-attempt the temporal /
grasp-rigidity / anchor-attitude / texture-baking fixes (all tested negative). Read, in order:
[`BEST_STRATEGY.md`](BEST_STRATEGY.md) (workflow + roadmap outcomes + **Open directions**),
[`compare/hot3d/docs/T5_NOTES.md`](compare/hot3d/docs/T5_NOTES.md) (full campaign), then
[`README.md`](README.md) (nav). Numbers: `compare/hot3d/scores/LEADERBOARD.md`. Envs +
box facts are in the recalled `hoi-recon-*` memories.
