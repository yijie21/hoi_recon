# Running the REAL render_and_compare HOI pipeline on this machine (wild6)

This documents the exact, reproduced-and-verified real (GPU) run of the
`render_and_compare` pipeline on `egoaero/assets/wild6.mp4` (a hand grasping a
white bottle). Every perception backend below ran with real weights — no mock,
no fabricated output.

## TL;DR — the working command

```bash
conda activate forehoi                       # single env holds everything (see below)
cd /workspace/code/hoi_recon/render_and_compare
export HF_HOME=/workspace/huggingface_cache/
export CUDA_VISIBLE_DEVICES=1                 # pick the GPU with the most free VRAM (>=20GB)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m hoi_recon.cli --video examples/wild6_trim.mp4 \
    --out runs/wild6_real --real --config configs/real_forehoi.yaml
```

Outputs land in `runs/wild6_real/` (see "Outputs" below).

## Environment

This box is an **RTX 4090 (sm_89)**, not Blackwell — so the cu121 stack works and
the two-env design collapses into **one existing conda env, `forehoi`**
(`/workspace/miniconda3/envs/forehoi`, torch 2.5.1+cu121, python 3.11). It already
holds MoGe, SAM2, ultralytics, all HaMeR runtime deps, chumpy, pytorch3d, kaolin,
and the SAM-3D-Objects fork. The `hoi_recon` package is installed editable there.

- **Main pipeline env AND the SAM-3D subprocess env are both `forehoi`**
  (`backend.sam3d_env: forehoi` in the config; no separate `sam3d-objects` env).
- Python: `/workspace/miniconda3/envs/forehoi/bin/python`.

### One-time setup that was done (already in place, do not repeat)

1. **MoGe-v2** installed into forehoi (`pip install --no-deps` MoGe main +
   its pinned `utils3d` commit `3fab839…`, which provides the `utils3d.pt` API).
   Weights downloaded to `checkpoints/moge/moge-2-vitl-normal/model.pt`.
2. **`checkpoints/` tree** (symlinks to weights already on the machine):
   - `moge/moge-2-vitl-normal/model.pt`   (downloaded, real file)
   - `sam2/sam2.1-hiera-large/sam2.1_hiera_large.pt` -> `/workspace/code/ForeHOI/weights/…`
   - `wilor/detector.pt` -> `hort/pretrained_models/detector.pt`
   - `hamer/hamer_ckpts/checkpoints/hamer.ckpt` + `model_config.yaml`
     -> `easyhoi/third_party/hamer/_DATA/hamer_ckpts/…`
   - `hamer/data/mano_mean_params.npz` -> easyhoi `_DATA/data/…`
   - `mano/MANO_RIGHT.pkl` -> `hort/mano_data/mano/MANO_RIGHT.pkl`
     (no MANO_LEFT anywhere on the box; HaMeR mirrors the right model, fine for wild6)
3. **`third_party/` tree**:
   - `third_party/hamer` -> `easyhoi/third_party/hamer` (its `ViTDetDataset` needs an
     extra `valid` arg; `backends/real_perception.py` was patched to pass it only when
     the constructor asks for it, so both HaMeR forks work).
   - `third_party/sam-3d-objects/` = per-item symlinks into
     `/workspace/code/ForeHOI/wheels/MV_SAM3D` (the SAM-3D-Objects "pointmap" fork,
     which has `notebook/inference.py`, `sam3d_objects/`, and `checkpoints/hf/pipeline.yaml`),
     **plus** this repo's subprocess entry scripts copied in
     (`sam3d_infer.py`, `render_compare.py`, `joint_opt.py`, `choir_*_opt.py`),
     **plus** two local package shadows (see the SAM-3D note below).

### The SAM-3D / utils3d conflict and how it is solved (important)

MoGe-v2 (main pipeline, stage 0) needs the **new** `utils3d` (`utils3d.pt`), while
`sam3d_objects` needs the **old** `utils3d` (`utils3d.numpy.depth_edge`) AND its
internal depth model imports **moge-1.0.0** (which uses `utils3d.torch`). These are
mutually exclusive in one site-packages. Because the SAM-3D subprocess runs with
`cwd = third_party/sam-3d-objects`, we drop **local package shadows** there so the
subprocess (only) sees the old stack, while the main forehoi process keeps the new one:

- `third_party/sam-3d-objects/utils3d/`  = utils3d 0.0.2 (from the cached wheel)
- `third_party/sam-3d-objects/moge/`     = moge 1.0.0 (from the cached wheel)

`sys.path[0]` (the script dir) wins over site-packages, so no env is polluted. This
is what makes SAM-3D run real in the same forehoi env.

## Which backends ran REAL vs fell back (config `real_forehoi.yaml`, `runs/wild6_real`)

| stage | backend | status |
|------|---------|--------|
| 0 depth + intrinsics | **MoGe-v2** | REAL |
| 0 camera | identity (static-cam assumption) | not estimated (no VIPE/DA3 on box) |
| 1 hand boxes | **WiLoR YOLO** (`detector.pt`) | REAL — detected a **left** hand, 74/74 frames |
| 1 object mask | **SAM 2.1** (point-prompt + propagate) | REAL |
| 2 hand -> MANO | **HaMeR** (metric-depth-anchored) | REAL (full 778-vert MANO + params) |
| 3 object shape | **SAM-3D-Objects** textured mesh (4000 verts, 8000 faces, vertex colors) | REAL |
| 3 object 6D | in-process silhouette rotation tracker on depth-lift translation | REAL |
| 7 grasp optim | numpy rigid joint hand+object contact optimization | REAL |
| 8 eval + overlays | this repo | REAL |

Self-consistency (stage 8): **gap median 1.8 mm**, 215/778 hand verts within 1 cm of
the object over 74 frames — i.e. a real, in-contact grasp (with the depth-lift-only
convex-hull object it was 75 mm / 1 vert, so SAM-3D matters).

### Notes / deviations from the "target" combined.yaml command
- The clip was **temporally subsampled to 74 frames** (`examples/wild6_trim.mp4`,
  stride 4 over the 294-frame 720x1280 original) to keep the multi-model pipeline
  tractable on the shared GPU. Full temporal extent is preserved. To run the full
  294 frames, point `--video` at the original `egoaero/assets/wild6.mp4`.
- `real_forehoi.yaml` uses the **in-process** silhouette object-pose tracker and the
  **numpy** grasp optimizer (robust, no heavy PyTorch3D subprocess). The target
  `combined.yaml` additionally runs the differentiable PyTorch3D **render-compare**
  object pose + **joint MANO-articulation** optimizer as forehoi subprocesses; a
  forehoi-wired copy is `configs/combined_forehoi.yaml` (only diff from the repo's
  combined.yaml: `sam3d_env: forehoi`). Run:
  `python -m hoi_recon.cli --video examples/wild6_trim.mp4 --out runs/wild6_combined \
   --real --config configs/combined_forehoi.yaml`

## Combined config (`configs/combined_forehoi.yaml`, `runs/wild6_combined`) — also REAL

The full best-performance path was also run and every heavier component was
confirmed to run REAL in the single forehoi env:
- Stage 2: HaMeR **+ CHOIR isolated fit (Eq 1)** applied (Dyn-HaMR gracefully skipped —
  no `dynhamr` env; it is optional).
- Stage 3: SAM-3D mesh + **differentiable render-and-compare** object 6D
  (`render_compare.py`, PyTorch3D subprocess in forehoi) — completed.
- Stage 7: **differentiable joint MANO-articulation + object optimizer**
  (`joint_opt.py`, PyTorch3D subprocess in forehoi) — runs and converges (2D-keypoint
  reprojection error fell 28.5px -> ~9px over its iterations).

IMPORTANT operational note (GPU memory ordering): the SAM-3D / render-compare /
joint-opt subprocesses each need ~13 GB. If stages 0-2 run **in the same process**
first, MoGe+SAM2+HaMeR stay resident and SAM-3D can OOM (it then falls back to the
depth-lift hull and the differentiable path is silently skipped). The robust recipe
is to run **stages 0-2 once, then re-invoke with `--stages 3-8`** so the main process
is light and the GPU subprocesses get the full card:
```bash
python -m hoi_recon.cli --video examples/wild6_trim.mp4 --out runs/wild6_combined \
    --real --config configs/combined_forehoi.yaml --stages 0-2
python -m hoi_recon.cli --video examples/wild6_trim.mp4 --out runs/wild6_combined \
    --real --config configs/combined_forehoi.yaml --stages 3-8
```
(Or pin `CUDA_VISIBLE_DEVICES` to the emptier GPU.) The combined run is slower
(PyTorch3D render at 720x1280 over 74 frames, 200+400 iters) and more sensitive to
the shared-GPU contention; `runs/wild6_real` (the silhouette + numpy-grasp config
above) is the fast, robust deliverable and already uses the real SAM-3D mesh.

## Outputs (`runs/wild6_real/`)

- **`stage8_eval/pseudo_gt.npz`** — keys: `hand_verts[74,778,3]`, `hand_joints[74,21,3]`,
  `obj_verts[4000,3]`, `obj_faces[8000,3]`, `obj_poses[74,4,4]`, `contact_map[74,285]`,
  `rectify_delta`, `object_delta`.
- **`stage7_contact_optim/arrays.npz`** — the same 4D HOI plus `obj_colors[4000,3]`
  and `hand_faces[1538,3]` (textured object + MANO topology).
- **`stage8_eval/report.json`** — self-consistency diagnostics.
- Reprojection-overlay validation videos + grids: `hand_reproj.mp4`,
  `object_reproj.mp4`, `hoi_reproj.mp4` (+ `*_grid.png`).
- Per-stage bundles under `stage0_preprocess/ … stage8_eval/` (frames, depth, masks,
  SAM-3D mesh cache at `stage3_object/sam3d/object.npz`).

View it: `conda run -n forehoi hoi-recon-view --run runs/wild6_real`.

## Reproducing from scratch on a fresh checkout
Re-run the setup once (weights + envs already on this box are reused via symlink):
the `checkpoints/` and `third_party/` symlink trees above, the MoGe-v2 + pinned
utils3d install into forehoi, and the two local shadows under
`third_party/sam-3d-objects/`. Then run the TL;DR command.
