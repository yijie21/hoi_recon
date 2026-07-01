# EgoAERO visual outputs — the two products of Figure 1

These reproduce the two halves of the paper's Fig. 1 from a single (mock) egocentric clip:

1. **3D HOI reconstruction** — contact-consistent hand-object trajectory (SP1).
2. **Simulated dexterous-hand execution** — that trajectory retargeted to a MuJoCo
   Shadow Hand and rolled out under the two-stage policy (SP2).

> Mock data, run on the base box. The trained policy is a short (40k-step, CPU) demo run
> that — honestly — does not beat the zero-action baseline (see `egoaero/README.md` SP2),
> so `sim_trained` and `sim_zero` look nearly identical. The point here is the *pipeline*:
> input clip → reconstruction → simulated hand, end to end.

## Files

| file | what it is |
|---|---|
| **`figure1_composite.png`** | **side-by-side Fig-1 analog: one clip → reconstruction → simulated hand** |
| `recon_run32_recapture.png` / `.gif` | 3D reconstruction of a loose-contact clip → **recapture** |
| `seq0000_repairable_accept.png` / `.gif` | 3D reconstruction of an accepted dataset sequence → **repairable_accept** |
| `sim_trained/sim_keyframes.png` / `sim_rollout.gif` | Shadow Hand executing under the trained two-stage policy (39-step realrun clip) |
| `sim_zero/sim_keyframes.png` / `sim_rollout.gif` | Shadow Hand under zero policy (baseline) |
| `sim_accept/sim_keyframes.png` / `sim_rollout.gif` | Shadow Hand on the accepted clip used in the composite |
| `dataset_summary.json` | SP4 collection summary (3 accepted / 11 attempts) |

The composite uses one accepted clip (`repairable_accept`, Q=0.49) for both halves so the
reconstruction and the simulated execution correspond frame-for-frame.

In the reconstruction panels: red = MANO hand skeleton, faint grey = hand mesh, blue = object
mesh, green = contact vertices; the bottom row plots contact count, object translation/speed,
and the quality verdict. In the sim panels: the grey Shadow Hand (orange = fingertip pads)
follows the reconstructed wrist trajectory while the sphere = manipulated object.

## Environments

Two envs are needed (the base `/venv/main` lacks numpy):

- **`/workspace/miniconda3/envs/forehoi/bin/python`** — numpy + matplotlib → the **reconstruction** figures.
- **`/workspace/miniconda3/bin/python`** (conda base) — numpy + mujoco + stable-baselines3 → the **sim** rollout.
  Needs OSMesa (`sudo apt-get install -y libosmesa6`) and `MUJOCO_GL=osmesa`.

## Reproduce

### (1) 3D HOI reconstruction
```bash
PY=/workspace/miniconda3/envs/forehoi/bin/python
# run the full pipeline (stages 0-8) on a mock clip, then write the contract
$PY -c "from egoaero import config,contract; from egoaero.pipeline import run_pipeline; \
        ctx=run_pipeline(config.load_config(overrides={'mock':True,'num_frames':32}),'/tmp/run','all'); contract.write(ctx)"
# render it
$PY viz_output/visualize.py /tmp/run viz_output/recon.png viz_output/recon.gif
```
Or build a whole accepted dataset via the SP4 closed loop, then visualize a sequence:
```bash
$PY -m egoaero.dataset.cli --out /tmp/dataset --n 3 --seed 0
$PY viz_output/visualize.py /tmp/dataset/seq_0000 viz_output/seq0000.png viz_output/seq0000.gif
```

### (2) Simulated Shadow-Hand execution
torch + OSMesa segfault in one process, so it is split into record (torch) → replay (GL):
```bash
PY=/workspace/miniconda3/bin/python
# need a reconstruction run dir with a contract/ (reuse one from step 1, e.g. /tmp/run)
# trained policy (omit the 3rd arg for the zero-policy baseline):
$PY viz_output/sim_record.py /tmp/run /tmp/rec.npz /tmp/egoaero_realrun/policy
MUJOCO_GL=osmesa $PY viz_output/sim_replay.py /tmp/run /tmp/rec.npz viz_output/sim trained-policy
```
`sim_record.py` rolls out `StageIIEnv` (mocap wrist follows the reconstructed hand; fingers from
the policy) and records per-step `qpos`/`mocap`; `sim_replay.py` replays and renders it offscreen.

---

## Real video: `wild6.mp4` (partial perception front-end)

`assets/wild6.mp4` (a real monocular RGB clip — a hand grasping an eye-drop bottle) **cannot** be
run through the EgoAERO pipeline: stages 1–6 are mock-only (`if not cfg.mock: raise`) and consume
the synthetic scene's ground-truth 3D arrays, and `backends/real.py` (HaWoR/SAM3/ORB-SLAM3/
BundleSDF/SAM3D) is a stub. So there is no asset-free object mesh, 6-DoF object track, or ego-SLAM.

What **is** genuinely recoverable from the real pixels with installed models is a perception
front-end, run by `perceive.py`:

| component | model | output |
|---|---|---|
| hand | **WiLoR** (`wilor_mini`, weights cached in `forehoi`) | per-frame **MANO** hand (778 verts + 21 joints + cam) |
| object | **YOLOv8** (COCO `bottle`) | manipulated-object bbox |
| depth | **Depth-Anything-V2-Small** (`transformers`) | monocular relative depth |

Result on wild6.mp4 (98 frames @ 10 fps): hand in **98/98**, bottle in **50/98**, depth on 20.

| file | what it is |
|---|---|
| `wild6_figure1.png` | Fig-1-style composite: real frames → perception → 3D MANO hand (honestly labeled partial) |
| `wild6_perception.png` | 4 keyframes: RGB+hand/bottle overlay (top) and 3D MANO hand (bottom) |
| `wild6_depth.png` | monocular depth maps |
| `wild6_overlay.gif` | hand+bottle tracking across the whole clip |

> Caveat: WiLoR's 3D hand is accurate within its crop, but the **2D overlay** uses an assumed
> focal length so it is only approximately aligned in the full frame; the 3D hand is the real product.

### Reproduce
```bash
P=/workspace/miniconda3/envs/forehoi/bin/python    # has wilor_mini, ultralytics, transformers, torch
ffmpeg -y -i assets/wild6.mp4 -vf "fps=10,scale=405:720" /tmp/wild6/frames/f%04d.png
CUDA_VISIBLE_DEVICES=1 $P viz_output/perceive.py        # -> /tmp/wild6/perception/*.npz
$P viz_output/viz_real.py                               # -> wild6_perception.png, wild6_depth.png, wild6_overlay.gif
$P viz_output/composite_real.py                         # -> wild6_figure1.png
```
(`perceive.py` auto-downloads YOLO + Depth-Anything weights on first run; WiLoR weights are cached.)

---

## Interactive viser viewer (`egoaero-view`)

Like `render_and_compare`'s viewer, the repo now ships an interactive 4D-HOI viser app
(`egoaero/egoaero/viz/viser_app.py`): object + hand played over a timeline with play/pause,
per-frame contact highlighting (in-contact hand verts turn red), optional contact lines, and
live contact / surface-gap readouts. Run it with the `forehoi` env (which has `viser`):

```bash
scripts/view_demo.sh                                    # mock run (auto-generated) — object mesh + hand point cloud
scripts/view_demo.sh /tmp/egoaero_viz/dataset/seq_0000  # an accepted dataset sequence
scripts/view_demo.sh viz_output/wild6_viser_scene.npz   # the REAL wild6 recon — MANO mesh hand + bottle proxy
# or directly:
/workspace/miniconda3/envs/forehoi/bin/python -m egoaero.viz.viser_app --run <run_dir> [--stage stage6_contact]
```

It serves a browser app on port 8080. To reach it from your laptop, SSH-forward the port:
```bash
ssh -p $VAST_TCP_PORT_22 -L 8080:127.0.0.1:8080 root@$PUBLIC_IPADDR   # then open http://localhost:8080
```

Two object modes are supported:
- **canonical mesh + per-frame 6-DoF pose** — the mock/dataset runs (full SP1 contract).
- **per-frame point cloud** — for exported real recons. `wild6_viser_scene.npz` (built by
  `export_viser.py`) holds the **real WiLoR MANO hand mesh** (778 verts + 1538 MANO faces) plus
  the **bottle bbox back-projected to the grasp-depth plane** — an honest detection proxy, *not* a
  reconstructed object (no asset / 6-DoF track / mesh). `wild6_viser_check.png` is a static preview.

### (3) Side-by-side Figure-1 composite
Needs a single accepted run dir (with `contract/` + `quality.json`) and its rendered sim strip:
```bash
RUN=/tmp/dataset/_work/attempt_0007          # any run with decision != recapture
PY=/workspace/miniconda3/bin/python
$PY viz_output/sim_record.py $RUN /tmp/rec.npz /tmp/egoaero_realrun/policy
MUJOCO_GL=osmesa $PY viz_output/sim_replay.py $RUN /tmp/rec.npz /tmp/sim accepted-clip
/workspace/miniconda3/envs/forehoi/bin/python viz_output/composite.py \
    $RUN /tmp/sim/sim_keyframes.png viz_output/figure1_composite.png
```
