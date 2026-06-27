# egoaero — EgoAERO Part A (Sec 2.1) Reconstruction

Self-contained implementation of the **EgoAERO egocentric hand-object reconstruction
pipeline** (Part A, Section 2.1 of the EgoAERO paper).  Runs fully in **mock mode**
today (no weights, no real cameras); real-backend stubs are wired and raise
`NotImplementedError` where documented.

---

## What it reproduces

**EgoAERO Part A, Sec 2.1** — asset-free, egocentric 4D hand-object reconstruction:

> Given a monocular egocentric RGB-D stream, reconstruct the per-frame MANO hand,
> object 6-DoF trajectory, and contact maps — without any per-object asset or template.

The 8-stage pipeline follows the paper's section structure and Appendices A–C.

---

## The 8 stages

| # | Name | Faithful to paper | Notes |
|---|------|-------------------|-------|
| 0 | `stage0_ego_io` | ✅ | Loads/generates ego frames, depth, GT scene (mock) |
| 1 | `stage1_semantic` | ✅ documented-default | MLLM prompt + seed-frame selection (§2.1.1); mock uses GT masks + area–occlusion score |
| 2 | `stage2_track` | ✅ documented-default | RANSAC coarse init + memory-pool pose-graph opt (App A); weights from BundleSDF/FoundationPose defaults |
| 3 | `stage3_mesh` | ✅ documented-default | Coarse-to-fine neural field (App B); SP1 mock uses tracked geometry + Umeyama alignment |
| 4 | `stage4_hand` | ✅ documented-default | HaWoR MANO + depth-residual global-translation correction (§2.1.3) |
| 5 | `stage5_ego_comp` | ✅ documented-default | Ego-motion compensation to table frame + temporal smoothing (§2.1.4) |
| 6 | `stage6_contact` | ✅ **faithful** | Adaptive contact optimisation (App C): App C constants verbatim, three-region iterative push-back |
| 7 | `stage7_eval` | ✅ | Before/after penetration, contact-gap, hand-jitter report |

"Documented-default" means the paper specifies the algorithm structure but omits
numeric constants or hyperparameters; values used are logged in [`ASSUMPTIONS.md`](ASSUMPTIONS.md).

---

## Quickstart

```bash
# from egoaero/ directory (no install required)
python -m egoaero.cli --out runs/demo --mock

# run only specific stages (e.g. stages 0-3)
python -m egoaero.cli --out runs/demo2 --mock --stages 0-3

# override config
python -m egoaero.cli --out runs/demo3 --mock --set num_frames=32 seed=42
```

Results land in `runs/<name>/`:
- `stage0_ego_io/`, `stage1_semantic/`, … `stage7_eval/` — per-stage bundles
- `contract/` — workbench contract output (see below)
- `config.yaml` — frozen run config

---

## Contract output layout

After a successful full-pipeline run the `contract/` folder holds:

```
contract/
  manifest.json        # lists all files + frame count
  hand_mano.npz        # verts (T, 778, 3) + joints (T, 21, 3) in table frame
  object_traj.npz      # obj_poses_t (T, 4, 4) SE3 trajectory in table frame
  object_mesh.obj      # reconstructed object mesh (OBJ, 1-indexed)
  contact.npz          # contact_mask (T, 778) binary per-vertex contact map
```

Validated by `egoaero.contract.validate(run_dir)` → `True` when all five files are present.

---

## Shared evaluation metrics

The workbench contract requires reporting:

- hand MPJPE (mm) and jitter
- object translation error (mm)
- penetration depth (mm)
- contact F1 and contact-frame gap (mm)

Stage 7 prints a before/after table for penetration, contact gap, and hand jitter.
Full MPJPE and contact-F1 require real GT annotations (planned for SP3 — see roadmap).

---

## Further reading

- [`ASSUMPTIONS.md`](ASSUMPTIONS.md) — every unspecified value: what was assumed, why, and where
- [`docs/specs/2026-06-27-egoaero-sp1-reconstruction-design.md`](docs/specs/2026-06-27-egoaero-sp1-reconstruction-design.md) — SP1 design spec
- [`docs/plans/2026-06-27-egoaero-sp1-reconstruction-plan.md`](docs/plans/2026-06-27-egoaero-sp1-reconstruction-plan.md) — SP1 implementation plan

---

## SP2 / SP3 / SP4 roadmap

| Sprint | Topic | Status |
|--------|-------|--------|
| SP2 | RL contact policy (real App C backend, learned push-back) | Planned |
| SP3 | Quality assessment (MPJPE, contact-F1, dataset eval) | Planned |
| SP4 | Dataset integration (Ego4D, EPIC-Kitchens egocentric clips) | Planned |

---

## Environment

This method shares the repository workbench conda/pip environment.  All runtime
dependencies (`numpy`, `scipy`, `pyyaml`, `trimesh`) are listed in
[`pyproject.toml`](pyproject.toml) and are available in the standard workbench env.
No separate `environment.yml` is needed — install with:

```bash
pip install -e .   # from egoaero/ directory
```
