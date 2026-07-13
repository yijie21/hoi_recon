# HOT3D results — best 4D hand-object reconstruction

*Reproduced 2026-07-13 on 6 HOT3D clips with mocap-grade ground truth. Lower is better on every metric.* Method codes are decoded in [`GLOSSARY.md`](../../../GLOSSARY.md).

## What the numbers mean
- **Placement (mm)** — average 3D gap between the reconstructed object and the true object, both placed in the scene. Under ~5 mm is a tight fit.
- **Rotation (deg)** — how well the object's turning matches truth, as median / 90th-percentile frame error. Large values mean the orientation is ambiguous (round/symmetric objects).
- **Hand fit (px)** — how far the reconstructed hand lands from the real hand in the image. 2–4 px is pixel-accurate.

## The result we ship (best object + best hand)
| clip | object method | placement (mm) | rotation med/p90 (deg) | hand fit (px) |
|---|---|---|---|---|
| bottle_bbq | learned core | 2.9 | 53.5/161.9 | 2.3 |
| mug_white | learned core | 4.1 | 12.2/33.5 | 3.8 |
| vase | learned core | 5.4 | 6.4/58.6 | 2.7 |
| spatula_red | learned core | 12.1 | 5.4/10.4 | 2.6 |
| puzzle_toy | learned core | 15.3 | 21.1/93.7 | 3.2 |
| potato_masher | registration pipeline | 22.0 | 62.4/81.8 | 1.9 |

## Object placement: registration pipeline vs learned core (mm)
| clip | registration pipeline | learned core | 
|---|---|---|
| bottle_bbq | 18.6 | 2.9 |
| mug_white | 6.9 | 4.1 |
| vase | 17.1 | 5.4 |
| spatula_red | 29.9 | 12.1 |
| puzzle_toy | 18.4 | 15.3 |
| potato_masher | 22.0 | (uses pipeline) |

## Hand fit: before vs after the hand optimizer (image reprojection, px)
| clip | before | after |
|---|---|---|
| bottle_bbq | 56.6 | 2.3 |
| mug_white | 23.3 | 3.8 |
| vase | 35.4 | 2.7 |
| spatula_red | 20.9 | 2.6 |
| puzzle_toy | 5.4 | 3.2 |
| potato_masher | 8.5 | 1.9 |

*Older experimental methods (Any6D, ForeHOI, FoundationPose-standalone, and the registration-pipeline variants) are compared in the campaign notes under `docs/`.*
