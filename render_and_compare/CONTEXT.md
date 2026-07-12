# render_and_compare — glossary

Project-specific vocabulary for the HOI reconstruction pipeline. Definitions only —
what a term *is*, not how it is implemented. Decisions live in [`docs/adr/`](docs/adr/).

## Hand

**Hand reprojection**:
The projection of the estimated MANO hand into an RGB frame. "Correct" reprojection means
the projected hand lands on the *observed* hand pixels. Improving it is the subject of
[ADR-0001](docs/adr/0001-hand-reprojection-optimizer.md).
_Avoid_: hand overlay, hand alignment (both overloaded).

**kp2d**:
HaMeR's 21 per-frame 2D hand keypoints, in full-image pixels, OpenPose joint order,
already un-mirrored for left hands. The hand's primary image-space evidence.
_Avoid_: joints2d, keypoints (ambiguous with the 3D `joints`).

**Hand-reprojection optimizer**:
The frozen-object `joint_opt.py` pass that aligns the MANO hand to `kp2d` + hand silhouette
(image-first), correcting wrist 6D + finger articulation while contact stays soft.
_Avoid_: hand refiner.

**Frozen-object arm**:
An arm whose object trajectory is held fixed downstream of stage 4 (e.g. any6dp, fpauto), so
stage-7 optimization moves only the hand. Contrast with `joint_grasp`, which moves both.

## Object (existing, for cross-reference)

**Arm**:
One end-to-end pipeline configuration + pose core, named and benchmarked (icpjgr, any6dp,
fpauto, ...). The unit of comparison on the leaderboard.

**Mesh-controlled**:
A comparison in which every arm registers/places the *same* SAM-3D stage-3 mesh, so only the
pose/placement method differs (removes SAM-3D GPU nondeterminism as a confound).
