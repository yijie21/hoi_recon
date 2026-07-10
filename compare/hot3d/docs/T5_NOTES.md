# T5 — Learned pose core *inside* the pipeline (item 1) + the rotation Pareto finding (item 2)

Roadmap items 1–2 from `BEST_STRATEGY.md` "What's still to improve". Item 1 shipped a
real win; item 2 is a comprehensively-established negative result. Both are recorded
here because the negative result is the more important scientific outcome.

## Item 1 — the learned core is now integrated (arm `any6dp`)

The old "combined" method was a 3-step **stitched** flow (pipeline → `run_any6d_hot3d.py`
→ `combined_refine.py`) whose learned poses never touched the grasp stages. Item 1 makes
the learned estimator a selectable **stage-4 pose core**: one pipeline run does
hand-aware seg → SAM-3D mesh → **Any6D per-frame RGB-D pose (forehoi5090 subprocess)** →
in-process temporal layer (symmetry-flip resolution + translation jitter smoothing) →
object **frozen** through the grasp stages → eval. Config `configs/real_any6d.yaml`
(`object_icp.pose_core: learned`). The object freeze (`smoothing.window: 1` +
`optim.joint.freeze_object: true`) guarantees eval scores exactly the learned+temporal
track while the hand still closes the grasp.

**Result — `any6dp` vs `icpjgr` (chamfer mm / rot_traj p90°, mesh-controlled):**

| clip | icpjgr | any6dp | |
|---|---|---|---|
| bottle_bbq | 9.9 / 72.9 | **5.5 / 21.5** | placement + rotation both win |
| mug_white | 7.0 / 76.5 | **3.4** / 72.9 | placement win |
| vase | 17.7 / 75.7 | **6.6** / 95.7 | placement win, rotation −20 |
| spatula_red | 21.2 / 42.6 | **12.8** / 142.5 | placement win, rotation −100 |
| potato_masher | 18.8 / 42.3 | 19.7 / 167.0 | ~, rotation −125 |
| puzzle_toy | 18.5 / 127.2 | 21.2 / 170.0 | ~, rotation −43 |

`any6dp` **wins chamfer on 4/6** (often 2–3×) but **fails the acceptance gate**
(`potato_masher rot_traj 24.1→46.4`): the learned core skips the ICP block where
icpjgr's rotation machinery (temporal-rotation + T2 chroma + T3 grasp-rigidity) lives, so
it loses rotation on symmetric / in-hand-rotated objects. `icpjgr` stays `BEST_ARM`;
`any6dp` is the documented **placement-optimal** alternative.

## Item 2 — the learned core's rotation is not recoverable (four negatives)

Goal: give the learned core icpjgr's rotation without losing its placement. Every avenue
failed, and the failures triangulate one conclusion.

1. **Depth-anchored basin selection** (`temporal_basin.py`, removed). Score the symmetry
   candidates per frame by depth-cloud fit, pick the best via Viterbi. → `basin_changed=0`
   on vase/spatula/masher (16 masher candidates). **Redundant**: Any6D already registers
   to depth, so re-scoring by depth just confirms its basin.
2. **Grasp-rigidity rotation prior** (`rotation_prior.py`, removed). Pull `dR_obj` toward
   `dR_wrist` on stable-grasp + fast-wrist frame pairs (T3 ported onto the learned poses).
   → **hurts** rot_traj (masher 46→67, vase 7.6→14): the HaMeR wrist is noisier than
   Any6D's already-depth-optimal per-frame rotation, and flip-resolution/smoothing disturb
   the many correct frames. (The old T3 lesson, sharper: marginal even for the *worse* ICP
   rotation, a net loss against the *better* learned rotation.)
3. **Surgical isolated-flip fix.** Replace only isolated-outlier frames (slerp neighbours),
   leave the rest. → ~neutral (vase p90 96→93, masher 46→73). The bad frames are
   **sustained azimuthal ambiguity**, not isolated flips.
4. **Hybrid transplant** (icpjgr rotation + Any6D translation/mesh). → hybrid rot_traj
   matches icpjgr **exactly** (transplant is coherent — shared stage-3 canonical frame),
   but chamfer collapses to icpjgr's level or worse (vase 6.6→18.9, spatula 12.8→35.9).

**The finding:** Any6D's chamfer advantage is *inseparable* from its rotation — both are
the same per-frame depth fit. Take icpjgr's rotation and you get icpjgr's chamfer back.
`any6dp` and `icpjgr` sit on a genuine **placement-vs-rotation Pareto frontier**; you can
have best placement XOR best rotation, not both, because each core's rotation and
translation are jointly fit to its own objective. For symmetric objects the two metrics
even *measure different things* — chamfer is azimuth-invariant (surface fit), rot_traj is
not — so the learned core scores a superb chamfer while its azimuth flickers.

**Kept from item 2:** the temporal layer's flip-aware infrastructure
(`temporal_pose.resolve_flips` + `smooth_traj` with an optional `lam_rot`), off by default
because even flip-aware rotation smoothing is a small net loss on the learned poses. The
depth-basin and grasp-rigidity experiment modules were removed (findings recorded here).

## Files
- Item 1 (kept): `render_and_compare/hoi_recon/{object_any6d.py, temporal_pose.py}`,
  `stages/stage4_align.py` (learned branch), `joint_grasp.py` (`freeze_object`),
  `configs/real_any6d.yaml`; runs `render_and_compare/runs/hot3d_*_any6dp`; scores
  `scores/batch_summary_any6dp.json`; overlays `overlays/rc_vs_gt_*_any6dp.mp4`.
- Item 2 (removed after negative): `temporal_basin.py`, `rotation_prior.py`,
  `configs/real_any6d_rot.yaml`.
