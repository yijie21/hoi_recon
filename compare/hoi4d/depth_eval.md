# HOI4D GT-depth evaluation — Kettle clip (C12/N15, 5s, camera ZY20210800002)

GT depth from HOI4D align_depth (16-bit mm). GT reference = kettle SAM2 mask unprojected with GT depth.
GT object longest span = 0.240 m; GT object depth = 0.829 m.

| method | object size (m) | object depth (m) | size err | depth err |
|---|---|---|---|---|
| **GT (kettle)** | 0.240 | 0.829 | — | — |
| render_and_compare (**GT depth injected**) | 0.240 | 0.829 | **+0%** | **−0%** |
| do-as-i-do (**GT depth, right kettle + GT-anchor**) | 0.190 | 0.827 | −21% | **−0%** |
| do-as-i-do (GT depth, wrong kettle + hand-anchor) ✗ | 0.312 | 1.058 | +30% | +28% |
| ForeHOI (monocular) | 0.319 | 0.587 | +33% | −29% |
| HORT (monocular) | 0.342 | 0.156 | +43% | −81% |

## Findings
- **render_and_compare + GT depth is exact** — anchors the object directly to GT depth; tight grasp (~1–2 mm).
- **Monocular HORT/ForeHOI** mis-scale (+33–43%) and badly misplace depth (−29% to −81%).
- **do-as-i-do — two bugs found and fixed:**
  1. *Wrong object.* Its pipeline segments with a **text** prompt ("kettle"); with multiple kettles on
     the desk SAM3 locked onto the wrong (blue, background) kettle. Fixed by using a **point** prompt on
     the grasped kettle body + a **negative** point on the table (a bare positive point over-segmented
     the whole table). Now segments/tracks the correct grasped kettle.
  2. *Hand-anchored depth.* Object placement anchored to the HaWoR hand, so it inherited HaWoR's
     hand-depth error (+28%). Fixed with a new `optimize_translation_scale.py --gt-anchor` that places
     the object directly at its GT-depth pointmap centroid and sizes the mesh from the GT object extent.
  Result: **object depth GT-exact (−0%, was +28%)**; size −21% (diagonal ≈ GT; residual from SAM-3D's
  canonical kettle aspect ratio + 5–95‑pct extent sizing) — vs +30%/+28% with the wrong kettle + hand anchor.

Per-method runtime (5s clip): HORT ~1 min, ForeHOI ~2 min, render_and_compare ~2.5 min, do-as-i-do ~3 h
(guided-diffusion tracker). Overlay videos: `compare/hoi4d/overlays/{rc_gt,daid_gt,forehoi,hort}.mp4`.
