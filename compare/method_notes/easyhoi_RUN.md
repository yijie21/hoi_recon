# EasyHOI — How to Run (single-view hand–object reconstruction)

Status on this machine (set up 2026-06-30):

| Stage | Env | State |
|-------|-----|-------|
| 1. Hand pose + mask (HaMeR + ViTPose + detectron2) | `easyhoi` | ✅ RUNS, verified |
| 2. Hand/object segmentation (LISA-13B) | `lisa` | ❌ NOT set up (heavy, see Blockers) |
| 3. Inpainting / remove hand (affordance_diffusion / GLIDE) | `afford_diff` | ❌ NOT set up (heavy, see Blockers) |
| 4. Segment inpainted object (SAM) + hoi_box | `easyhoi` | ⚠️ env ready, needs stage-3 output + SAM ckpt |
| 5. Object 3D shape (InstantMesh, local/free path) | `instantmesh` | ✅ RUNS, verified |
| 6. Watertight mesh (resample_mesh) | `easyhoi` | ⚠️ runs but meshfix no-ops (trimesh API bug); optim falls back to `full.obj` |
| 7. Prior-guided optimization (final HOI) | `easyhoi` | ⚠️ env READS & IMPORTS fine, but needs the masks from stages 2–4 → BLOCKED |

**Runnable end-to-end? NO** — the full pipeline needs 2 more heavy, unintegrated
environments (LISA + affordance_diffusion). The repo itself states this is unfinished
("Integrate the code execution environments into one" / "Complete a one-click demo" are
open TODOs).

**What IS produced here:** a real **hand mesh** (HaMeR) and a real **object mesh**
(InstantMesh) for the test image — i.e. the "initial reconstruction" floor. They are in
**different, un-aligned coordinate frames**; the prior-guided optimization (stage 7) is
what fuses + aligns them, and it is blocked on the segmentation/inpaint masks.

---

## Environments & exact Python paths

- **easyhoi** — `/workspace/miniconda3/envs/easyhoi/bin/python`
  - Python 3.9, torch 2.1.0+cu121, torchvision 0.16.0
  - pytorch3d 0.7.8, nvdiffrast 0.4.0, detectron2 0.6, mmcv-full 1.7.2, mmpose 0.24.0 (ViTPose fork), manotorch, chamfer_distance, mesh-to-sdf, pymeshfix, open3d, geomloss, libigl, numpy 1.23.5
  - Used for: stage 1 (hand), stage 4 (SAM seg), stage 6 (resample), stage 7 (optimization)
- **instantmesh** — `/workspace/miniconda3/envs/instantmesh/bin/python`
  - Python 3.10, torch 2.1.0+cu121, diffusers 0.20.2, transformers 4.34.1, accelerate 0.23.0, nvdiffrast 0.4.0, onnxruntime, numpy 1.26.4
  - Used for: stage 5 (object shape)
- **NOT created:** `lisa` (LISA-13B segmentation) and `afford_diff` (affordance_diffusion inpainting). See Blockers.

Always set before any GPU stage:
```bash
export CUDA_VISIBLE_DEVICES=0
export PYOPENGL_PLATFORM=egl
```

### Build notes (gotchas already solved, for re-creating the env)
- nvcc must be CUDA 12.1 (installed into the env via conda `nvidia/label/cuda-12.1.1::cuda-toolkit`); system nvcc is 13.2 and rejects gcc>12.
- All CUDA source builds (nvdiffrast / detectron2 / pytorch3d) need conda gcc-11:
  `conda install -c conda-forge gxx_linux-64=11 gcc_linux-64=11`, then
  `export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc CXX=...g++` and build with `--no-build-isolation`.
- HaMeR / ViTPose are installed **editable, `--no-deps`** to avoid their `mmcv==1.3.9` pin; we use prebuilt `mmcv-full==1.7.2` (cu121/torch2.1.0) instead, and patched `third_party/ViTPose/mmpose/__init__.py` (`mmcv_maximum_version` 1.5.0 → 1.8.0).
- Keep `numpy<2` (chumpy/mmcv/detectron2). `deprecation` is required by manotorch.

---

## Weights (already downloaded)

- **HaMeR demo bundle** → `third_party/hamer/_DATA/` (≈6 GB): `hamer_ckpts/checkpoints/hamer.ckpt`, `vitpose_ckpts/vitpose+_huge/wholebody.pth`, `data/mano_mean_params.npz`.
- **detectron2 ViTDet** body detector: auto-downloaded to torch cache on first run.
- **MANO** (reused, no new license): `MANO_RIGHT.pkl` copied to **both**
  `third_party/hamer/_DATA/data/mano/MANO_RIGHT.pkl` and `assets/mano/models/MANO_RIGHT.pkl`
  (manotorch expects `assets/mano/models/`). Source: `/workspace/code/hoi_recon/hort/mano_data/mano/MANO_RIGHT.pkl`.
- **InstantMesh** → `ckpts/diffusion_pytorch_model.bin` (1.7 GB), `ckpts/instant_mesh_large.ckpt` (1.5 GB); `sudo-ai/zero123plus-v1.2` and `facebook/dino-vitb16` in HF cache (`/workspace/huggingface_cache`).

---

## Input expected

A data directory containing an `images/` subfolder with RGB images (`.png`/`.jpg`).
Demo dir used here: `/workspace/code/hoi_recon/easyhoi/data_run/` with `images/f0030.png`
(405×720, a hand grasping an eye-drop bottle).

```bash
export DATA_DIR=/workspace/code/hoi_recon/easyhoi/data_run
```

---

## Step-by-step commands

Run everything from the repo root: `cd /workspace/code/hoi_recon/easyhoi`.

### Stage 1 — Hand pose + hand mask (HaMeR)  ✅
```bash
/workspace/miniconda3/envs/easyhoi/bin/python preprocess/recon_hand.py --data_dir $DATA_DIR
```
Produces `$DATA_DIR/hamer/{name}.pt`, `{name}_cam.json`, render PNGs, and
`$DATA_DIR/obj_recon/hamer_mask/{name}.png`.
To also dump the **hand mesh** as `.obj`, call HaMeR's demo directly with `--save_mesh`
(recon_hand.py omits it):
```bash
cd third_party/hamer && /workspace/miniconda3/envs/easyhoi/bin/python demo.py \
  --img_folder $DATA_DIR/images --out_folder $DATA_DIR/hamer \
  --hand_mask_path $DATA_DIR/obj_recon/hamer_mask \
  --batch_size=1 --full_frame --save_mesh ; cd ../..
```

### Stage 2 — LISA segmentation (hand_mask + obj_mask)  ❌ NOT set up
```bash
# conda activate lisa   # env not created here
CUDA_VISIBLE_DEVICES=0 python preprocess/lisa_ho_detect.py --seg_hand --skip --load_in_8bit --data_dir $DATA_DIR
CUDA_VISIBLE_DEVICES=0 python preprocess/lisa_ho_detect.py            --skip --load_in_8bit --data_dir $DATA_DIR
```
Would write `$DATA_DIR/obj_recon/hand_mask/` and `.../obj_mask/`.

### Stage 3 — Inpaint (remove hand → clean object image)  ❌ NOT set up
```bash
# conda activate afford_diff   # env not created here
python preprocess/inpaint.py --data_dir $DATA_DIR --save_dir $DATA_DIR/obj_recon/ --img_folder images --inpaint --skip
```
Would write `$DATA_DIR/obj_recon/input_for_lrm/{name}/full.png` (object image fed to InstantMesh)
and `$DATA_DIR/obj_recon/inpaint/hoi_box/{name}.json` (needed by stage 7).

### Stage 4 — Segment inpainted object + inpaint_mask (SAM)  ⚠️
```bash
/workspace/miniconda3/envs/easyhoi/bin/python preprocess/seg_image.py --data_dir $DATA_DIR
```
Needs a Segment-Anything checkpoint and the stage-3 inpainted image. Writes
`$DATA_DIR/obj_recon/obj_mask/` and `.../inpaint_mask/`.

### Stage 5 — Object 3D shape (InstantMesh, local/free)  ✅
Input: `$DATA_DIR/obj_recon/input_for_lrm/{name}/full.png` (normally from stage 3).
```bash
HF_HOME=/workspace/huggingface_cache HUGGINGFACE_HUB_CACHE=/workspace/huggingface_cache/hub \
/workspace/miniconda3/envs/instantmesh/bin/python preprocess/instantmesh_gen.py \
  preprocess/configs/instant-mesh-large.yaml $DATA_DIR
```
Produces `$DATA_DIR/obj_recon/results/instantmesh/instant-mesh-large/meshes/{name}/full.obj`.
(Tripo3D alternative `preprocess/tripo3d_gen.py` needs an API key — not available, skip.)

> In this demo, stages 2–3 were not run, so `full.png` was prepared by cropping the
> bottle region from the input frame (InstantMesh's own `rembg` removes background).
> This is a legitimate InstantMesh run, but with the hand not inpainted out the mesh is
> noisier than the full pipeline would give. Nothing was faked.

### Stage 6 — Watertight mesh  ⚠️
```bash
/workspace/miniconda3/envs/easyhoi/bin/python preprocess/resample_mesh.py --data_dir $DATA_DIR
```
Intended to write `fixed.obj` next to `full.obj`. Currently no-ops because `meshfix` calls
`trimesh.simplify_quadratic_decimation`, renamed to `simplify_quadric_decimation` in
trimesh 4.x (the exception is swallowed). **Non-blocking:** the optimizer loads `fixed.obj`
if present else falls back to `full.obj`.

### Stage 7 — Prior-guided optimization (final HOI)  ⚠️ BLOCKED on masks
```bash
/workspace/miniconda3/envs/easyhoi/bin/python src/optim_easyhoi.py -cn optim_teaser \
  out_dir=$DATA_DIR/easyhoi_out log_dir=$DATA_DIR/easyhoi_out/log
# (-cn optim_teaser_tripo when using Tripo3D meshes)
```
The env imports and MANO load cleanly. It cannot complete for the demo image because
`src/configs/data/teaser.yaml` requires, per image:
`obj_recon/hand_mask/{name}.png`, `obj_recon/obj_mask/{name}.png`,
`obj_recon/inpaint_mask/{name}.png`, `obj_recon/inpaint/hoi_box/{name}.json`
— all outputs of stages 2–4, which are not set up. Also note the config's `base_dir`
defaults to `./data`; point it at the demo dir by editing `src/configs/data/teaser.yaml`
(`base_dir: ./data_run`) or adding a Hydra override.

---

## OUTPUTS — locations, formats, frames, units  (important for the viewer task)

### Hand (stage 1) — REAL, produced
- `data_run/hamer/f0030_0.obj` — **HAND MESH**. MANO topology: **778 vertices, 1552
  triangular faces**, per-vertex RGBA colors. Wavefront OBJ.
  **Frame:** HaMeR perspective-camera space, **OpenGL/pyrender convention** (+X right,
  +Y up, +Z toward camera) after `vertices_to_trimesh` adds `camera_translation` and a
  180° rotation about X. **Units: meters.** The mesh sits in front of the camera at
  z ≈ −4.17 m (bounds x∈[−0.095,0.024], y∈[−0.089,0.056], z∈[−4.218,−4.123]).
  The `_0` suffix is the hand-detection index.
- `data_run/hamer/f0030.pt` — torch dict:
  `mano_params{ global_orient [1,1,3,3] rotmat, hand_pose [1,15,3,3] rotmats, betas [1,10] }`,
  `cam_transl [1,3]`, `is_right [1]` (**0.0 = LEFT hand for this image**),
  `keypts [1,21,2]`, `boxes` (list of xyxy), `valid`, `batch_size`.
- `data_run/hamer/f0030_cam.json` — `{ extrinsics 3×4, fx, fy, cx, cy }`, normalized
  intrinsics (cx=cy=0.5; fx,fy are in normalized units, not pixels).
- `data_run/obj_recon/hamer_mask/f0030.png` — binary hand mask rendered by HaMeR.
- `data_run/hamer/f0030_0.png`, `f0030_all.jpg` — overlay visualizations.

### Object (stage 5) — REAL, produced
- `data_run/obj_recon/results/instantmesh/instant-mesh-large/meshes/f0030/full.obj`
  — **OBJECT MESH**. **≈32,222 vertices, ≈64,424 faces**, per-vertex colors. OBJ.
  **Frame:** InstantMesh canonical object space, normalized to roughly a **[−1,1] cube**
  (bounds ≈ x[−0.97,0.97], y[−0.41,0.53], z[−0.95,0.89]). **Not metric, not aligned to
  the hand** — alignment/scale to the hand is exactly what stage 7 solves.
- `.../images/f0030/full.png` — zero123plus 6-view grid; `input.png` — rembg'd input.

### Final HOI (stage 7) — NOT produced here; format from the code, for reference
When stages 2–4 are completed the optimizer writes to `out_dir`:
- `out_dir/after_{name}.ply` — **COMBINED hand+object mesh** (final result). It is a
  trimesh concatenation: **hand first (778 MANO verts / 1552 faces), then object verts/
  faces with per-vertex colors**. Frame: HaMeR camera space (`hamer_process` applied), with
  the optimizer's `hand_scale=10` and solved object alignment. Intermediate
  `init_{name}.ply` and `after_global_{name}.ply` use the same combined format.
- `out_dir/render/{prefix}_{name}.png` — rendered overlay of the combined mesh.
- `out_dir/eval_final/{name}.pkl` — torch dict `{ wTh [4×4], hTo [4×4], obj_mesh
  (trimesh), hA (MANO pose, axis-angle) }`.
- `out_dir/eval_final/{name}_hand_in_objcam.pkl` — torch dict `{ name, img_path, is_right,
  cam_projection, cam_extrinsics, mano_params, hand_scale, hand_transl, hand_rot }`.
  (Default config `out_dir` is an absolute `/storage/...` path — always override it.)

---

## Blockers (precise)

1. **LISA (stage 2)** — needs a `lisa` env and `xinlai/LISA-13B-llama2-v1-explanatory`
   (~26 GB, LLaVA-based) + DeepSpeed/old-transformers stack. Produces `hand_mask` + `obj_mask`.
   Setup: https://github.com/dvlab-research/LISA
2. **affordance_diffusion (stage 3)** — needs an `afford_diff` env + GLIDE/affordance
   weights. Produces the hand-removed object image (`input_for_lrm/full.png`) and `hoi_box`.
   Setup: https://github.com/NVlabs/affordance_diffusion/blob/master/docs/install.md
3. **Stage 7 requires stages 2–4 masks** (`hand_mask`, `obj_mask`, `inpaint_mask`, `hoi_box`).
   Without them the data loader returns `None` and skips the image. Masks were **not**
   fabricated.
4. **`fixed.obj` not generated** — trimesh 4.x renamed `simplify_quadratic_decimation`.
   Non-blocking (optimizer falls back to `full.obj`). To fix, edit `meshfix()` in
   `preprocess/resample_mesh.py` to call `simplify_quadric_decimation`.
5. **Tripo3D object path** — `preprocess/tripo3d_gen.py` needs a paid API key (none
   available). Use the local InstantMesh path instead.
