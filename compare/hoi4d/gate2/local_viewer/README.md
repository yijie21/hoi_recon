# Local 4D viewer — VGGT-Omega reconstructions of the HOI4D kill-test clips

Self-contained package: an interactive point-cloud player for the VGGT-Omega
4D reconstructions (48 frames per clip), with hand/object tinting, camera
trajectory, confidence filtering, and an optional GT sensor-depth reference.
No GPU, no dataset, no repo — just Python.

## Setup (any OS, Python >= 3.9)

```bash
pip install -r requirements.txt      # numpy + viser
python viewer.py                     # then open http://localhost:8080
```

Options: `--clip <name>` initial clip, `--port <p>` (default 8080).
The "clip" dropdown in the UI switches clips live.

## Controls

- **Playback**: frame slider, autoplay, fps.
- **Points**: stride (1 = densest), max depth clip, per-frame confidence
  percentile filter, point size; color = plain RGB, hand/object tint
  (hand green / object red, rest dimmed), or confidence heatmap.
- **Reference**: camera trajectory (grey frusta + orange centres; the current
  frame's frustum is highlighted with its image), and a GT sensor-depth cloud
  drawn at a +1.5 m x-offset. The GT cloud is in its OWN (real camera)
  coordinate frame — a side-by-side reference, not an aligned overlay.

## What to look for

Switch color to *hand/object tint* and play: the background stays put while
the red object (and green hand) subtly breathe in depth during the grasp —
that per-frame foreground depth-gauge wobble is exactly what the kill tests
quantified (median residual gauge error R_ko = 0.35 for VGGT-Omega; see
`compare/hoi4d/gate2/RESULTS.md` in the repo).

## Data

`data/<clip>.npz`: VGGT-Omega depth + confidence (f16, 384x688), preprocessed
RGB (u8), per-frame world-to-camera extrinsics (world = frame-0 camera,
OpenCV) + intrinsics, hand/object masks, GT align_depth resampled to model
resolution with matching intrinsics, and the source frame indices.
Regenerate on the server with `pack_local_viewer.py` (one dir up).

`data/<clip>__gtdepth.npz`: the SAME clip rebuilt from GT sensor depth
(HOI4D align_depth) — static camera at the origin, true intrinsics, same 48
frames / resolution / masks. Each clip therefore appears twice in the
dropdown for direct A/B: pick `<clip>`, note a view/frame, switch to
`<clip>__gtdepth`. Expect the GT clouds to show sensor holes (zero-depth
pixels are dropped), and the VGGT clouds to be hole-free but with the
foreground depth breathing the harness quantified (obj depth ~9.6 vs 1.4 cm,
jitter 1.68 vs 0.65 cm on the RC matrix). The `confidence` color mode is
meaningless for GT entries (uniform confidence).
Regenerate with `make_gt_demo_data.py` (one dir up).
