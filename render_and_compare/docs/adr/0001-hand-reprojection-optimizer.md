# Hand-reprojection optimizer: image-first, object frozen, reuse joint_opt.py

Status: accepted (2026-07-12)

## Context

The best arms (icpjgr / any6dp / fpauto) run the stage-7 **`joint_grasp`** optimizer,
which moves the hand *rigidly toward the object* to close the grasp and anchors it only
weakly to HaMeR's 3D hand (`E_prior`). It has **no image term**, so HaMeR's per-frame
error — dominated by wrist translation (HaMeR's weak-perspective depth is discarded and
replaced by a depth-lift, still noisy) plus some articulation — is never corrected, and
the grasp closure can drag the hand *further* off the observed pixels. Result: the MANO
hand does not backproject onto the hand in the RGB (visible in `hand_reproj.mp4`).

## Decision

Add a **hand-reprojection optimization** that aligns the MANO hand to the image, reusing
the already-written (but dormant) `third_party/sam-3d-objects/joint_opt.py`, which already
implements every piece we want (`L_kp2d`, `L_hand_sil`, wrist6 + 15-joint articulation +
betas, `L_prior` to HaMeR, contact + non-penetration). Concretely:

1. **Evidence = kp2d + hand silhouette** (`L_kp2d` primary, `L_hand_sil` secondary). HaMeR's
   21 2D keypoints (`kp2d`, present for all frames) directly pin the hand to observed pixels;
   the SAM2 hand mask constrains the boundary.
2. **DOF = wrist 6D + finger articulation**, regularized strongly toward HaMeR (`L_prior`);
   **betas a single global value**, not per-frame.
3. **Object frozen.** Add `--freeze_object` to `joint_opt.py` (drop `o_r6`/`o_t` from the
   optimizer) so on frozen-object arms it optimizes *only the hand* against the fixed object
   track. Enabled as the stage-7 optimizer for those arms.
4. **Image-first weighting.** `w_kp2d`, `w_hsil` dominate; `w_contact`, `w_pen` are soft and
   confidence-gated (contact leads only where keypoints are occluded/low-confidence). The
   hand follows the observed pixels; contact only closes occluded fingers.

## Considered options

- **GT-raycast hand depth** (fit the hand to the hand-inclusive HOT3D depth, like the object
  ICP core): strongest signal on HOT3D but the depth is *derived from the GT UmeTrack hand*,
  so it fits to GT and **does not generalize** to datasets without GT hands. Rejected for the
  optimizer (kept as the independent *evaluation* signal — see below).
- **Extend numpy `joint_grasp`** with image terms: it is rigid/numpy with no differentiable
  renderer or articulation — a rewrite in the wrong framework. Rejected.
- **New standalone hand driver**: duplicates `joint_opt.py`'s MANO layer + renderer. Rejected
  in favour of reuse.
- **Contact-first / balanced weighting**: lets the hand sit off the observed pixels to seat on
  the object — contradicts the reprojection goal. Rejected.

## Consequences

- **Success is measured against GT, not the loss.** kp2d and the hand mask are optimization
  *targets*, so scoring against them is circular. Evaluate with an **independent** metric vs
  the HOT3D GT UmeTrack hand (hand chamfer + 2D-joint reprojection error), mirroring the
  object's chamfer-vs-GT, plus always eyeballing `hand_reproj.mp4`.
- **Needs SAM2 hand masks persisted.** Stage 1 saves hand *boxes* (`hand_boxes[T,2,4]`) but
  not per-pixel hand masks; `L_hand_sil` needs the masks SAM2 already computes for the hands.
  Either persist them from stage 1 or generate from the boxes. If unavailable, `w_hsil=0` and
  kp2d alone still runs.
- **Applies to the full-pipeline arms** (icpjgr, any6dp) that carry a hand. fpauto is currently
  an object-only standalone driver; wiring a hand through it is a separate step.

## Validation (2026-07-12)

Built as: `joint_opt.py --freeze_object` + `--w_contact`/`--w_pen` args (image-first: raise
`w_kp2d`, keep `w_pen`, soften `w_contact`), driven standalone by
`compare/hot3d/run_hand_reproj.py` on an existing icpjgr run; scored by
`compare/hot3d/gt_hand_eval_hot3d.py` vs the GT UmeTrack hand; before/after overlay by
`compare/hot3d/make_hand_reproj_compare.py`. (MANO had to be re-pickled chumpy-free —
`checkpoints/mano/MANO_RIGHT_np.pkl` — since sam3d5090 has no chumpy.)

All 6 clips, kp2d-only (no hand masks yet), vs GT UmeTrack hand — chamfer mm / 2D reproj px, med (p90):

| clip | BEFORE (joint_grasp) chamfer / reproj | AFTER (kp2d-aligned) chamfer / reproj |
|---|---|---|
| bottle | 45.5 (111) / 42.6 (83) | **9.0 (14) / 2.3 (2.7)** |
| mug | 101.0 (185) / 17.4 (531) | **25.8 (139) / 3.8 (4.1)** |
| vase | 45.6 (295) / 39.4 (273) | **14.6 (75) / 2.8 (3.6)** |
| potato_masher | 198.5 (447) / 107.5 (236) | **56.8 (156) / 1.9 (2.6)** |
| spatula | 145.9 (251) / 68.0 (140) | **7.1 (18) / 2.6 (3.5)** |
| puzzle | 19.4 (43) / 5.4 (41) | **7.5 (15) / 3.2 (4.0)** |

**2D reprojection collapses to 1.9–3.8 px on every clip** (from 5–108 px median, 40–530 px p90 — the
mis-placed-frame tail is eliminated); hand chamfer drops 3–20×. Before/after overlays:
`compare/hot3d/overlays/hand_cmp_<clip>.mp4`. The residual
~9 mm chamfer is almost entirely depth (kp2d + silhouette are both z-blind; the hand z stays
on HaMeR's depth-lift anchor) — an image-only floor, since correcting z would need the GT-hand
depth we deliberately excluded for generality. Visually (overlays) the AFTER hand sits on the
observed hand at both fine articulation and gross placement, where BEFORE it floated off.

Open follow-ups: (1) `L_hand_sil` refinement — needs SAM2 hand masks persisted (stage 1 keeps
only `hand_boxes`); kp2d-only already lands 2.3 px so this is polish. (2) Wire
`pose_core`-style into stage 7 (`differentiable` + `--freeze_object`) as a one-command arm.
