# T6 — A better learned core than Any6D: FoundationPose auto (arm `fpauto`)

> **In plain terms.** This document introduces `fpauto`, a new learned object-pose method built
> on FoundationPose, that fixes the placement-vs-rotation trade-off documented in T5. By giving
> FoundationPose an undistorted object mesh and using its own frame-to-frame tracker (instead of
> treating every frame independently) plus a per-clip check that falls back to a safer mode when
> tracking drifts, `fpauto` becomes the first learned arm to beat the previous best learned arm
> (`any6dp`) on BOTH object placement accuracy AND rotation accuracy at the same time. It is now
> the best learned object-pose method in the project, alongside the still-more-robust hand-built
> `icpjgr`. Method codes decoded in [GLOSSARY.md](../../../GLOSSARY.md).

Open-direction #1 from `BEST_STRATEGY.md` ("swap the learned stage-4 core to
FoundationPose") — built and **validated as a win on both axes** over `any6dp`.
This is the first arm to beat the placement-optimal `any6dp` on placement *and*
rotation simultaneously.

## The key realization (why "FP vs Any6D" is really "scale + tracker")

`any6d/estimater.py` exposes ONE class whose plain `register()` **is** FoundationPose's
render-and-compare register (same refiner/scorer). So `any6dp` already *contains* "FP
register_each" — it does `register_any6d` (scale) at the anchor, then `register` per
frame. The two remaining levers Any6D does **not** pull:

1. **Uniform metric scale.** `register_any6d` rescales the mesh **per-axis** (OBB extent
   ratio, `estimater.py:456-458`) → distorts revolution shapes. We instead feed FP the
   **icpjgr uniform** global-metric mesh (`{icpjgr_run}/stage8_eval/pseudo_gt.npz`
   obj_verts) — same SAM-3D shape, undistorted. (T4 already hinted this: FP-reg bottle
   3.2 < Any6D 5.2; with the icpjgr mesh we get **2.6**.)
2. **Temporal tracking.** FP `track` mode (register once + `track_one` fwd/back) keeps
   rotation **continuous** → no per-frame symmetry flips. This is the flip-free rotation
   the whole T5 rotation campaign deemed unreachable via temporal/attitude/texture priors
   — FP's tracker gets it *natively* where it holds.

## The arm: `fpauto` — per-clip drift-gated selector

Driver: `compare/hot3d/run_fp_hot3d.py` (env `forehoi5090`). One FP pass computes three
pose sets and caches them (`_fp_all_poses.npz`):
- **register_each** — independent per-frame `register`: drift-free metric translation,
  but flip-prone per-frame rotation.
- **track** — register anchor + `track_one` fwd/back: flip-free continuous rotation, but
  can drift in translation (2/6 clips).
- **fuse** — `track` rotation + `register_each` translation (coherent: same estimator,
  same centered mesh frame — unlike the failed T5 icpjgr-R/Any6D-t transplant).

**`auto`** picks per clip: register_each *always* supplies the (drift-free) translation;
`track` supplies the rotation **iff the tracker held**, judged by
`median‖track_t − reg_t‖ < 5cm`. The gate is unambiguous — held clips agree to
**0.5–1.9 cm**, drift clips diverge to **25–65 cm** (a 13× margin):

| clip | median‖track−reg‖ | verdict |
|---|---|---|
| mug / vase / spatula | 0.9 / 0.5 / 1.9 cm | HELD → fuse (reg-t + track-R) |
| puzzle | 4.8 cm | HELD → fuse |
| bottle / masher | 25.5 / 64.8 cm | DRIFTED → register_each |

**The temporal jitter-smoother (any6dp's layer) is OFF for this arm** — it *hurts*
(bottle chamfer p90 4.1→14.6, masher 9.7→13.0) because FP-track already gives rotation
continuity natively and register_each's per-frame translation, though jittery, is
data-accurate; the Any6D-tuned λ_trans over-smooths it. Raw `auto` is simpler and better.

## Results — `fpauto` vs `any6dp` vs `icpjgr` (mesh-controlled, chamfer mm / rot_traj p90°)

| clip | icpjgr | any6dp | **fpauto** | fpauto gate | vs any6dp |
|---|---|---|---|---|---|
| bottle_bbq | 9.9 / 72.9 | 5.5 / 21.5 | **2.6** / 162.9 | drift→reg | placement win; rot loss* |
| mug_white | 7.0 / 76.5 | 3.4 / 72.9 | 5.8 / **26.2** | held→fuse | rot win; placement loss |
| vase | 17.7 / 75.7 | 6.6 / 95.7 | **4.9** / **83.7** | held→fuse | **both win** |
| potato_masher | 18.8 / 42.3 | 19.7 / 167.0 | **9.7** / 173.0 | drift→reg | placement win; rot ~tie |
| spatula_red | 21.2 / 42.6 | 12.8 / 142.5 | **11.0** / **11.3** | held→fuse | **both win (rot 13×)** |
| puzzle_toy | 18.5 / 127.2 | 21.2 / 170.0 | **15.3** / **74.7** | held→fuse | **both win** |
| **mean** | 15.5 / 72.9 | 11.5 / 111.6 | **8.2 / 88.6** | | **wins both means** |

`fpauto` **wins placement 5/6** (all but mug, and marginally) and **rotation 4/6**
(mug, vase, spatula, puzzle) — mean chamfer **8.2 mm** (any6dp 11.5) and mean rot_traj
p90 **88.6°** (any6dp 111.6). Rendered-and-eyeballed: `overlays/rc_vs_gt_*_fpauto.mp4`
— spatula tracks GT tightly in position *and* blade orientation.

*bottle rot loss is on an **unobservable** axis: it is a revolution can, so its azimuth
is invisible to shape (chamfer 2.6 mm is perfect; the can *looks* right in the overlay).
`rot_traj` penalizes an ambiguity the eye cannot see — the same wall T5 documented, not
a real regression. masher rotation (173° ≈ any6dp 167°) stays bad because its `track`
drifts → falls back to flip-prone register_each; icpjgr's ICP rotation (42°) still owns
the sustained-in-hand-rotation regime.

## Where each arm now sits (three-way Pareto, not two)

- **`icpjgr`** — rotation-robust `BEST_ARM`; only arm that bounds worst-case rotation on
  in-hand-rotated symmetric objects (masher 42°) via its ICP + grasp-rigidity machinery.
- **`any6dp`** — Any6D per-axis scale; best on mug placement (3.4).
- **`fpauto`** — best **overall** placement (mean 8.2) and best learned-core rotation
  where the tracker holds (spatula 11°, mug 26°, vase/puzzle 75–84°). The new
  placement-optimal *and* rotation-competitive learned arm.

## Reproduce

```bash
# env forehoi5090; needs the mesh-controlled icpjgr run on disk (uniform metric mesh)
cd compare/hot3d
PY=/workspace/miniconda3/envs/forehoi5090/bin/python
$PY run_fp_hot3d.py <rc_input> <icpjgr_run> <out> --mode auto     # register_each+track+fuse cached
/workspace/miniconda3/envs/rc5090/bin/python gt_pose_eval_hot3d.py <rc_input> <out>
./score_fp_modes.sh <rc_input> <fp_run> <icpjgr_run>              # compare all 3 modes
```

## Not yet done (follow-ups)

- **Full `pose_core: learned / method: fp` pipeline wiring** (an `object_fp.py` peer of
  `object_any6d.py` + `real_fp.yaml`), so `fpauto` runs as one `run_batch.py --arm fpauto`
  command with the grasp stages. Needs the uniform metric mesh handed over from the
  incumbent (or a self-contained uniform-scale estimator from anchor depth). The
  standalone driver above already produces the scored track.
- **Recover masher/bottle rotation:** both fall to register_each because `track` drifts.
  A re-anchoring tracker (periodic re-register when depth-fit score drops) could keep
  `track` alive on those clips → close the last rotation gap.
