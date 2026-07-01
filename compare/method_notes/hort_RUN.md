# HORT — How to Run (verified working on this machine)

HORT reconstructs a **right hand (MANO mesh) + the held object (point cloud)** from a
single RGB image. This file documents the exact, *verified-working* setup and the
output format. Status: **RUNNABLE — full pipeline works (hand mesh + object point cloud).**

Verified 2026-06-30 on 2x RTX 4090, CUDA driver, conda env `hort`.

---

## 1. Environment

- Conda env name: **`hort`** (Python 3.12)
- Python interpreter (use this exact path): **`/workspace/miniconda3/envs/hort/bin/python`**
- Activate with:
  ```bash
  source /workspace/miniconda3/etc/profile.d/conda.sh
  conda activate hort
  ```
- GPU: pin to one card to avoid contention: **`CUDA_VISIBLE_DEVICES=0`**
- Torch: 2.4.1 + CUDA 12.1 (`torch.cuda.is_available() == True`)
- `transformers` is pinned to **4.44.2** (see Gotchas — do not let lang-sam upgrade it).

A local CUDA 12.1 toolkit was extracted to `./.cuda121/` (nvcc + headers). It is **only
needed to (re)build the `pointnet2_ops` CUDA op**; it is NOT needed to run the demo. The
op is already built and installed into the env, so normal runs need nothing extra.

---

## 2. Working demo command (this is the verified one)

```bash
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate hort
cd /workspace/code/hoi_recon/hort
CUDA_VISIBLE_DEVICES=0 python demo.py --img_folder demo_input --out_folder out_demo
```

- `--img_folder` : a **folder** of input images (the script globs `*.jpg *.png *.jpeg`).
  It is NOT a single-image arg. Put your image(s) in a folder and point at the folder.
- `--out_folder` : output directory (default `out_demo`), created if missing.
- Other args: `--rescale_factor` (default 2.0, bbox padding), `--file_type` (glob list).

The test image used here lives at `/tmp/wild6/frames/f0030.png` and was copied into
`demo_input/` before running. Repo sample images are in `demo_img/` (test1..test8).

First run downloads two extra models automatically to `~/.cache`: SAM2.1 (`sam2.1_hiera_large.pt`,
~856 MB) and GroundingDINO (`IDEA-Research/grounding-dino-base`) from HuggingFace. After
that, runs are offline-capable.

Optional open3d viewer (needs a display; this is headless so it will not pop a window):
```bash
python vis_ho.py -e out_demo/f0030.json
```

---

## 3. Input

- A single RGB image of a **right hand grasping an object**. (Left hands are flipped to
  right internally.) Any resolution; JPG/PNG/JPEG.
- Pipeline per image: LangSAM (text prompt `"manipulated object"`) finds the object bbox →
  WiLoR YOLO detector finds the hand → WiLoR reconstructs the MANO hand → HORT predicts the
  object point cloud in the hand frame.

---

## 4. OUTPUT — location and format  (IMPORTANT for the viewer task)

For an input image `f0030.png`, the demo writes three files into `--out_folder` (here
`/workspace/code/hoi_recon/hort/out_demo/`), all sharing the image basename:

| File | What it is |
|------|------------|
| `f0030.obj` | **Hand mesh** (Wavefront OBJ) |
| `f0030.json` | **Object point cloud + camera + palm/translation** |
| `f0030.jpg` | 224x224 RGB overlay render (hand mesh purple + object cloud blue over the crop) |

### 4a. Hand mesh — `f0030.obj`
- Wavefront OBJ, **778 vertices, 1552 triangular faces**.
- Topology = MANO right hand with a closed wrist (faces from `mano_data/closed_fmano.npy`).
- Vertices are **already in the camera coordinate frame** (`MANO verts + cam_t`).
- **Units: meters.** (Example: hand sits at depth z ≈ 3.98–4.07 m, spans ~12 cm.)
- Load directly, e.g. `trimesh.load('f0030.obj', process=False)`.

### 4b. JSON — `f0030.json`
Keys (all values are plain Python lists; shapes below):

| Key | Shape | Meaning |
|-----|-------|---------|
| `cam_extr` | (3,3) | Camera extrinsics = **identity** (world frame == camera frame). |
| `cam_intr` | (3,4) | Pinhole intrinsics for the **224x224 crop**: `fx=fy=4375.0`, `cx=cy=112`. |
| `pointclouds_up` | (16384, 3) | **Object point cloud**, upsampled, in the **object-local / hand-relative frame** (centered near origin). Already scaled by 0.3 inside the demo. **NOT yet translated into the camera frame.** |
| `objtrans` | (3,) | Object translation (object frame -> relative to palm). |
| `handpalm` | (3,) | Hand palm point in the **camera frame**, meters (`(verts[95]+verts[22])/2 + cam_t`). |

**To place the object point cloud in the SAME camera frame as the hand mesh** (this is exactly
what `vis_ho.py` does, with `cam_extr = I`):

```python
import json, numpy as np
d = json.load(open('out_demo/f0030.json'))
pts   = np.asarray(d['pointclouds_up'], np.float32)   # (16384,3) object-local
palm  = np.asarray(d['handpalm'],  np.float32)        # (3,) camera frame
trans = np.asarray(d['objtrans'],  np.float32)        # (3,)
obj_world = pts + palm + trans                        # (16384,3) in camera frame, meters
# Now obj_world and the OBJ hand-mesh vertices live in the same frame.
```

`vis_ho.py` additionally runs a statistical-outlier removal
(`remove_statistical_outlier(nb_neighbors=20, std_ratio=3.5)`) before display — optional.

### 4c. Coordinate frame summary
- Single right-handed **camera frame**, OpenCV-style (x right, y down, z forward into scene).
- `cam_extr` is identity, so camera frame == world frame.
- **Units: meters** throughout (hand mesh, palm, objtrans, and the translated object cloud).
- Hand mesh OBJ is pre-translated into this frame; the object cloud in JSON is NOT — add
  `handpalm + objtrans` as above.

---

## 5. Verified run (the deliverable run)

Command run:
```bash
CUDA_VISIBLE_DEVICES=0 python demo.py --img_folder demo_input --out_folder out_demo
# demo_input/ contained a copy of /tmp/wild6/frames/f0030.png
```
Produced files:
- `/workspace/code/hoi_recon/hort/out_demo/f0030.obj`  (hand mesh, 778 v / 1552 f)
- `/workspace/code/hoi_recon/hort/out_demo/f0030.json` (object cloud 16384 pts + cam + palm/trans)
- `/workspace/code/hoi_recon/hort/out_demo/f0030.jpg`  (overlay render, 224x224)

The overlay visually confirms a correct reconstruction: MANO hand aligned to the gloved hand,
object point cloud sitting on the held bottle.

---

## 6. Setup notes / gotchas (for reproducing the env from scratch)

The env was built following `README.md` with these deviations needed to make it work:

1. **MKL mismatch** (`undefined symbol: iJIT_NotifyEvent` on `import torch`):
   fixed with `conda install "mkl<2024.1"`.
2. **chumpy / legacy sdists fail under pip build isolation** (`No module named 'pip'` /
   `No module named 'numpy'`): installed with `pip install --no-build-isolation ...`
   after `pip install "setuptools<70" wheel pip cython numpy`.
3. **pytorch3d 0.7.8**: installed from the prebuilt conda package
   `pytorch3d-0.7.8-py312_cu121_pyt241.tar.bz2` (downloaded from anaconda.org/pytorch3d),
   via `conda install ./pytorch3d-...tar.bz2`.
4. **`pointnet2_ops` CUDA op** (`hort/models/tgs/models/snowflake/pointnet2_ops_lib`):
   - System nvcc is CUDA 13.2 (incompatible with torch cu121). A matching **CUDA 12.1**
     toolkit was assembled in `./.cuda121/` from nvidia conda packages (nvcc, cudart-dev,
     cusparse/cublas/cusolver/curand/cufft dev headers, nvrtc).
   - Host gcc is 13; nvcc 12.1 needs gcc<=12, so built with `gcc-12`/`g++-12`.
   - `setup.py` `TORCH_CUDA_ARCH_LIST` was edited to include **8.9** (RTX 4090).
   - Build command:
     ```bash
     export CUDA_HOME=/workspace/code/hoi_recon/hort/.cuda121
     export PATH=$CUDA_HOME/bin:$PATH CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12
     export TORCH_NVCC_FLAGS="-ccbin /usr/bin/gcc-12"
     cd hort/models/tgs/models/snowflake/pointnet2_ops_lib && python setup.py install
     ```
   (Already built/installed — only needed if the env is rebuilt.)
5. **`typeguard`** is required by the tgs module but missing from requirements: `pip install typeguard`.
6. **transformers version conflict (the important one):** `lang-segment-anything` upgrades
   `transformers` to 5.x, which breaks HORT's DINOv2 import
   (`cannot import name 'find_pruneable_heads_and_indices'`). **Fix: pin back**
   `pip install "transformers==4.44.2" "tokenizers==0.19.1"`. lang-sam (GroundingDINO + SAM2)
   still works fine at 4.44.2. (A harmless pip warning about gradio wanting huggingface-hub>=1.2
   remains; the demo does not use gradio.)
7. MANO: `mano_data/mano/MANO_RIGHT.pkl` was already present and is reused (license-walled;
   not re-downloaded).

---

## 7. Pretrained weights (in `pretrained_models/`)
- `detector.pt` — WiLoR hand detector (YOLO), from HF `rolpotamias/WiLoR`.
- `wilor_final.ckpt` — WiLoR hand reconstruction, from HF `rolpotamias/WiLoR`.
- `hort_final.pth.tar` — HORT object model, from HF `zerchen/hort_models`.
- `model_config.yaml`, `dataset_config.yaml` — shipped with the repo.
Auto-downloaded on first run (cached in `~/.cache/torch` and HF cache): SAM2.1 hiera-large,
GroundingDINO `IDEA-Research/grounding-dino-base`.
