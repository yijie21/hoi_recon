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

## Item 3 — geometry anchor-attitude search (built, tested — NOT robustly achievable)

Target: the ICP core's *constant* anchor-attitude error on shape-asymmetric objects
(the mug's absolute attitude `rot_abs` is 138.9° despite a good `rot_traj` of 22.6° — a
wrong-basin constant offset). A go/no-go over a 400-hypothesis SO(3) grid, scoring each
by depth-cloud fit and evaluating its `rot_abs` vs GT, showed:

- **The errors are correctable in principle** — a constant rotation exists that takes the
  mug to `rot_abs` 12.2° and the spatula to 42.9° (from 158.4°).
- **Multi-frame depth+silhouette only *partially* finds it** where the shape is
  discriminative: the mug's handle drives depth to `rot_abs` 45.6° (a large real fix from
  138.9°), but depth is **fooled** on the thin spatula (best 154.6° vs 42.9° achievable)
  and on the near-symmetric masher (item 2's `basin_changed=0`), and is fundamentally
  blind on revolution objects (vase/bottle).
- **The whole fix is gate-invisible** — `rot_traj` (the acceptance metric) removes exactly
  the constant offset item 3 fixes, so the gate never moves; item 3 improves *absolute*
  reconstruction quality (the object faces the right way) but not the leaderboard.

**Then built and tested it** (`attitude_fix.py`, since removed): a constant SO(3)
correction found by a multi-frame depth+silhouette search over a 300-hypothesis grid,
applied only past a relative-gain gate. It does **not** work robustly:

- **The mug (the target) never fires.** Its `rel_gain` stays 0.03–0.04 across silhouette
  weights (`w_sil` 0.5→2.0), i.e. the corrected fit is no better than the current pose —
  the go/no-go's "45.6°" was a noise-level argmin, not a real signal.
- **Root cause — the cue is occluded.** The mug's only attitude feature is the handle, and
  the mug's object-mask area swings **4× (22.9k→95.5k px)** across the clip: it is grasped
  and heavily/variably occluded by the hand, so neither depth-chamfer (handle is a tiny
  surface fraction) nor silhouette (handle often hidden) carries an attitude signal. This
  is a *data* limitation — the discriminative feature is exactly where the hand is.
- **The search false-fires on revolution objects.** The bottle triggers a 116° correction
  (`rel_gain` 0.24) — a partial-view depth cloud makes azimuthal rotation look better by
  chance. This is precisely the wrong-basin trap the ICP core's `rotation: init` + prior
  deliberately avoids.

So the item-3 fix is confirmed unachievable on this data, for a principled reason, not a
tuning failure — the module was removed. Consistent with items 2 and 4: the
rotation/attitude axis is a hard wall.

## Item 4 — texture re-projection (attempted, negative)

Idea: bake the object's REAL texture from the clip (project each frame's RGB onto the
SAM-3D mesh via the icpjgr poses, aggregate per-vertex) so the T2 chroma-attitude term
works on objects whose SAM-3D-*generated* texture is wrong (the cube's stickers). A
facing-weighted multi-frame baking prototype (4 clips):

| clip | seen | cross-frame color-std | baked-vs-SAM3D dLAB | surface chroma-spread baked / SAM3D |
|---|---|---|---|---|
| mug | 79% | 0.18 | 5.7 | 2.4 / 0.7 |
| masher | 50% | 0.30 | 14.7 | 3.9 / 1.8 |
| cube | 47% | 0.30 | 13.3 | **6.3 / 2.3** |
| bottle | 73% | 0.24 | 20.7 | 6.9 / **9.2** |

Negative on three grounds. **(1) Smeared, not crisp:** the per-vertex cross-frame colour
std is 0.18–0.30 (on [0,1]) — a blurry texture, because a clean bake needs the accurate
rotations the bake is meant to help produce (chicken-egg). **(2) Circular:** baking with
the icpjgr poses makes the texture self-consistent with them, so re-running T2 just
re-confirms the baking pose (no attitude gain vs GT). **(3) Defeated where it helps:** only
the cube's real stickers are markedly more discriminative than SAM-3D's (6.3 vs 2.3), but
the cube is 24-fold symmetric and the bake is smeared, so the stickers can't pin azimuth;
and where T2 already works (bottle) the duller baked texture (6.9) is *worse* than SAM-3D's
vivid generated label (9.2), so baking would HURT it. A non-circular variant (search the
anchor azimuth for the one maximizing cross-frame texture consistency) needs crisp relative
tracking and still breaks on symmetry — the same wall as items 2–3. Not built; prototype
scratch-only.

## Item 6 — scale validation

Selected 6 more single-object HIT clips (same categories, different sequences/instances)
via motion+FOV probing → `selection_scale.json` (12-clip `selection_all.json`). Ran
`icpjgr` + `any6dp` on all 12. Two clips are non-comparisons: spatula 003024 skipped
(target off-screen at frame 0, can't seed SAM2); masher 002334 is a bad SAM-3D mesh
(canonical-ICP 65 mm — a stage-1/3 failure) so both arms fail (icpjgr 171.5, any6dp 164.7).
The remaining **11 scored clips** (5 new + 6 frozen):

| new clip | icpjgr chamfer | any6dp chamfer | factor |
|---|---|---|---|
| bottle_bbq 002044 | 7.1 | **2.8** | 2.5× |
| mug_white 002107 | 6.9 | **1.9** | 3.6× |
| mug_white 002113 | 6.9 | **2.3** | 3.0× |
| spatula_red 003010 | 23.4 | **3.9** | 6.0× |

**The placement win generalizes and strengthens.** any6dp wins chamfer on **9/11** clips
(the 2 losses are the cube and the near-symmetric masher 002349). New-clip median chamfer
**7.1 → 2.8**; all-clip mean (excluding the shared-mesh failure) **13.7 → 8.0 mm**. On the
fresh clips the win is *larger* than on the frozen bench (up to 6× on spatula 003010).

**The Pareto trade-off holds.** any6dp regresses rotation (`rot_traj` p90) on **6/11** —
exactly the symmetric / in-hand-rotated objects where icpjgr's rotation priors do the
work. So the 6-clip finding is confirmed at 2× scale across 4 object categories and 6
sequences: the learned core is decisively **placement-superior**, at a **rotation** cost.

## Overlays — objects + HANDS (the full-HOI visualization)

The overlays now splat the **hand(s)** as well as the object, from two sources (they must
differ):
- **GT hands = UmeTrack** (mocap-grade, toolkit-native forward kinematics), NOT MANO —
  HOT3D's MANO thetas are unreliable and the only `MANO_LEFT.pkl` on the box is a fabricated
  right-hand mirror (reverses the left palm). `make_rc_input.py` already computes these in
  the rectified pinhole frame to ray-cast depth but discards them, so `extract_gt_hands.py`
  re-derives + saves them per clip (`<rc_input>/gt_hands.npz`: per-frame verts, camera frame).
- **Estimated hands = the pipeline's HaMeR/HaWoR MANO hand**, grasp-optimized in stage 7 —
  already stored (`hand_verts` in `pseudo_gt.npz`, `hand_faces` in `stage7`). Left-hand clips
  are unreliable (fabricated MANO_LEFT; `stage2_hand.hand_side` flags them); the GT UmeTrack
  hand stays correct either way. The pipeline reconstructs only the **interacting** hand,
  whereas GT shows both — a faithful difference, not a bug.

3-way HOI overlay: `make_best_overlay.py <cat> <num>` → `overlays/best/hoi3_<cat>_<num>.mp4`
(GT | icpjgr | any6dp; object in the panel colour, hands in tan). Confirmed by eyeball: the
UmeTrack GT hands overlay the real hands finger-perfect.

## Files
- Item 1 (kept): `render_and_compare/hoi_recon/{object_any6d.py, temporal_pose.py}`,
  `stages/stage4_align.py` (learned branch), `joint_grasp.py` (`freeze_object`),
  `configs/real_any6d.yaml`; runs `render_and_compare/runs/hot3d_*_any6dp`; scores
  `scores/batch_summary_any6dp.json`; overlays `overlays/rc_vs_gt_*_any6dp.mp4`.
- Item 2 (removed after negative): `temporal_basin.py`, `rotation_prior.py`,
  `configs/real_any6d_rot.yaml`.
- HOI overlays: `extract_gt_hands.py` (GT UmeTrack hand extractor), `make_best_overlay.py`
  (3-way object + hand overlay).
