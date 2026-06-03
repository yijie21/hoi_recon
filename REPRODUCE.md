# Reproducing the real (GPU) pipeline on another machine

End goal:

```bash
python -m hoi_recon.cli --video path/to/clip.mp4 --out runs/clip01 --real \
    --hand depthlift --object sam3d --depth moge      # runs fully today (no MANO)
python -m hoi_recon.cli --video path/to/clip.mp4 --out runs/clip01 --real \
    --hand hamer     --object sam3d --depth moge      # same, once MANO is in place
hoi-recon-view --run runs/clip01                      # view the 4D result in a browser
```

This guide assumes all checkpoints are already downloaded (the scripts below fetch
them, but you can also place them by hand using the directory tree in §4).

---

## 0. Machine assumptions

- Linux, NVIDIA GPU (verified on **RTX 5080, 16 GB**), ≥ ~30 GB RAM, ≥ ~15 GB disk for weights.
- `conda` (miniconda/anaconda) and `git` installed.
- **CUDA / PyTorch:** the RTX 50xx (Blackwell, `sm_120`) requires **PyTorch built for
  CUDA 12.8** (`torch==2.7.0+cu128`). On an older GPU, change the `--index-url` in
  `scripts/setup_real.sh` to your CUDA (e.g. `cu121`) and use a matching torch.

---

## 1. Code + conda env

```bash
git clone <your-repo-url> hoi_recon && cd hoi_recon
conda env create -f environment.yml        # creates env "hoi_recon" (Python 3.10)
conda activate hoi_recon                    # also installs this package (-e .)
```

Sanity check that the mock pipeline works with zero weights:

```bash
python -m hoi_recon.cli --out runs/demo --mock      # prints the error-attribution table
pip install pytest && python -m pytest tests/ -q    # 5 passed
```

## 2. Third-party model repos

```bash
bash scripts/setup_third_party.sh           # shallow-clones into third_party/
```

Clones HaMeR, WiLoR, SAM2, CoTracker, MoGe, Depth-Anything-V2, SAM-3D-Objects,
BundleSDF, FoundationPose, Dyn-HaMR, HaWoR, ViPE. The real pipeline below only needs
**MoGe, sam2, WiLoR, hamer**; the rest are for alternative backends.

## 3. Real-backend Python deps (GPU)

```bash
bash scripts/setup_real.sh                  # see §5 for the exact verified versions
```

Installs (into the active `hoi_recon` env): torch/torchvision (cu128), MoGe (`-e`),
SAM2 (`-e`), ultralytics + dill + trimesh, and HaMeR runtime deps
(pytorch-lightning, smplx, yacs, einops, timm, webdataset). **No detectron2 needed** —
we use WiLoR's YOLO hand boxes instead of HaMeR's detectron2 detector.

## 4. Checkpoints

```bash
bash scripts/download_checkpoints.sh        # MoGe + SAM2 + WiLoR + HaMeR (hf + wget)
```

Then place **MANO by hand** (license-gated — only needed for `--hand hamer/wilor`):

1. Register and accept the license at https://mano.is.tue.mpg.de
2. Copy the right (and left) hand models so the tree below has
   `checkpoints/mano/MANO_RIGHT.pkl` (and `MANO_LEFT.pkl`).

### Exact checkpoint layout the code expects

```
checkpoints/
├── moge/moge-2-vitl-normal/model.pt                         # depth + intrinsics (stage0)
├── sam2/sam2.1-hiera-large/sam2.1_hiera_large.pt            # object masks (stage1)
├── wilor/detector.pt                                        # YOLO hand boxes (stage1)
├── hamer/hamer_ckpts/
│   ├── checkpoints/hamer.ckpt                               # hand recon (stage2)
│   └── model_config.yaml
└── mano/MANO_RIGHT.pkl   (+ MANO_LEFT.pkl)                  # MANO model — LICENSE-GATED, manual
```

If you mirror weights manually on an offline machine, reproduce exactly this tree.
(The SAM2 config `configs/sam2.1/sam2.1_hiera_l.yaml` ships **inside the `sam2`
package**, not in `checkpoints/`, so nothing to place for it.)

## 5. Verified working versions

The set this was validated against (RTX 5080 / CUDA 12.8):

| package | version | | package | version |
|---|---|---|---|---|
| torch | 2.7.0+cu128 | | viser | 1.0.29 |
| torchvision | 0.22.0+cu128 | | numpy | 2.2.6 |
| moge | 2.0.0 | | opencv-python | 4.13.0.92 |
| SAM-2 | 1.0 | | trimesh | 4.12.2 |
| ultralytics | 8.4.60 | | huggingface-hub | 1.17.0 |
| dill | 0.4.1 | | gdown | 6.1.0 |
| pytorch-lightning | 2.6.5 | | smplx | 0.1.28 |

## 6. Run

```bash
# fully runnable today (MANO-free hand: lifts the hand region from MoGe depth)
python -m hoi_recon.cli --video clip.mp4 --out runs/clip01 --real \
    --hand depthlift --object sam3d --depth moge

# the HaMeR variant — identical once checkpoints/mano/MANO_RIGHT.pkl exists
python -m hoi_recon.cli --video clip.mp4 --out runs/clip01 --real \
    --hand hamer --object sam3d --depth moge

hoi-recon-view --run runs/clip01            # browser viewer of the 4D HOI
```

---

## What is verified vs. wired

| stage | backend | status |
|------|---------|--------|
| 0 depth + intrinsics | MoGe-2 (`--depth moge`) | ✅ verified on GPU |
| 0 camera extrinsics | identity fallback (ViPE not wired) | ✅ (static-camera assumption) |
| 1 hand boxes | WiLoR YOLO | ✅ verified |
| 1 object mask | SAM 2.1 (point-prompt + propagate) | ✅ verified |
| 2 hand (MANO-free) | `--hand depthlift` (MoGe depth lift) | ✅ verified end-to-end |
| 2 hand → MANO | `--hand hamer` (HaMeR) | ⚙️ wired; needs MANO + first real run to confirm |
| 3 object shape + 6D | depth-lift (SAM2 mask + MoGe depth → hull mesh) | ✅ verified |
| 4–7 align / contact optim, 8 eval | this repo (numpy) | ✅ verified |
| viewer | viser | ✅ verified |

So `--hand depthlift` runs end-to-end **today**. `--hand hamer` is wired up to the
**MANO license gate**; I could not execute it without the (license-gated) MANO model,
so treat that stage as "wired, pending first run."

## Troubleshooting

- **`torch.cuda.is_available()` is False / "no kernel image"** — torch CUDA doesn't match
  the GPU. RTX 50xx needs `cu128` (torch ≥ 2.7). Reinstall torch with the right `--index-url`.
- **`--hand hamer` → "MANO model required … LICENSE-GATED"** — expected; place
  `MANO_RIGHT.pkl` (§4) or use `--hand depthlift`.
- **MANO `.pkl` load fails inside `chumpy` (`cannot import name 'bool' from numpy`)** —
  chumpy needs `numpy<1.24`, but this env uses `numpy>=2` (for MoGe/SAM2). Use a patched
  chumpy or run stage 2 in a dedicated `numpy<1.24` env; all other stages are fine on numpy 2.
- **SAM2 `cannot import name '_C'` warning** — benign (optional CUDA post-processing
  extension not built); masks are unaffected.
- **Object mask grabs the wrong thing** — SAM2 is prompted at the hand-box centre by
  default. Edit `_object_prompt` in `backends/real_perception.py`, or feed a click point.
- **0 contacts / large gap** — expected if the clip isn't a real grasp (hand and object
  far apart). Use a clip where the hand actually holds an object near it.
```
