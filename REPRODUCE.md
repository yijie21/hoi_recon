# Reproducing the real (GPU) pipeline on another machine

End goal — **the validated configuration** (the result in `runs/grab`):

```bash
python -m hoi_recon.cli --video examples/grab.mp4 --out runs/grab --real \
    --hand hamer --object sam3d --depth moge --config configs/new.yaml
hoi-recon-view --run runs/grab                        # view the 4D result in a browser
```

`--config configs/new.yaml` selects the redesigned differentiable pipeline:
SAM-3D textured object mesh, **render-and-compare** object 6D (silhouette +
photometric), and the **joint MANO-articulation + object** grasp optimizer in
stage 7. It needs **two conda envs**: the main `hoi_recon` env (§1–3) plus a
`sam3d-objects` env (§3b) that hosts the PyTorch3D/SAM-3D components, run as
cached subprocesses. Without `--config configs/new.yaml` you get the older path
(silhouette-only object rotation, rigid non-articulated grasp), which runs in the
single main env:

```bash
python -m hoi_recon.cli --video path/to/clip.mp4 --out runs/clip01 --real \
    --hand depthlift --object sam3d --depth moge      # MANO-free, single env
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

Clones HaMeR, WiLoR, SAM2, CoTracker, MoGe, Depth-Anything-V2/-3, VGGT,
SAM-3D-Objects, BundleSDF, FoundationPose, Dyn-HaMR, HaWoR, ViPE. The validated
pipeline needs **MoGe, sam2, WiLoR, hamer, sam-3d-objects**; the rest are for
alternative backends.

The script also installs this repo's **subprocess entry scripts** into the clones
(the new pipeline drives the heavy components through them):
`sam3d_infer.py`, `render_compare.py`, `joint_opt.py` → `third_party/sam-3d-objects/`,
`vggt_geom.py` → `third_party/vggt/`, `fp_track.py` → `third_party/FoundationPose/`.
Their tracked source of truth is `scripts/subprocess_entries/<repo>/` — if you edit
one there, re-run `setup_third_party.sh` to re-install it.

## 3. Real-backend Python deps (GPU)

```bash
bash scripts/setup_real.sh                  # see §5 for the exact verified versions
```

Installs (into the active `hoi_recon` env): torch/torchvision (cu128), MoGe (`-e`),
SAM2 (`-e`), ultralytics + dill + trimesh, and HaMeR runtime deps
(pytorch-lightning, smplx, yacs, einops, timm, webdataset). **No detectron2 needed** —
we use WiLoR's YOLO hand boxes instead of HaMeR's detectron2 detector.

Optional: **Depth-Anything-3** (`--depth da3`, metric depth + real camera poses) —
installed by `setup_real.sh` with `--no-deps` (its `numpy<2` pin is over-conservative;
it runs fine on numpy 2). Use it instead of MoGe for moving-camera clips; weights
auto-download from HF (or pre-fetch `depth-anything/DA3METRIC-LARGE` into `checkpoints/da3/`).

### Verify the single env runs everything

```bash
python scripts/check_env.py
```

This is an **import matrix + fake-load pipeline run** (stub models, random weights, no
checkpoints) that confirms one env holds all backends. Expected result:

```
passed: 34   hard-fail: 0   expected-fail: 1
✅ One conda env runs ALL related code (ours + MoGe + DA3 + SAM2 + ultralytics +
   HaMeR + WiLoR) and the full real pipeline (fake weights).
⚠️  Only `chumpy` cannot coexist with numpy>=2 — needed solely to load the official
   MANO .pkl for --hand hamer/wilor. Use --hand depthlift (no MANO), or a patched
   chumpy / numpy<1.24 side-env for that one step.
```

So **a single `hoi_recon` env runs every code path** of the older pipeline; the
MANO `.pkl` (chumpy vs numpy≥2) is handled by a runtime patch
(`_patch_numpy_for_chumpy` in `backends/real_perception.py`), so `--hand hamer`
also works in this env once MANO is placed. The differentiable components of
`configs/new.yaml` additionally need the env below.

## 3b. The `sam3d-objects` env (required for `configs/new.yaml`)

SAM-3D-Objects, PyTorch3D (render-compare + joint optimizer), VGGT and
FoundationPose have torch/numpy pins that conflict with the main env, so they live
in a second conda env and are invoked via `conda run` subprocesses. Build it per
`third_party/sam-3d-objects/doc/setup.md` (conda/mamba env named `sam3d-objects`,
torch 2.5.1 + cu121 + PyTorch3D + kaolin), then fetch its weights:

```bash
# HF access to facebook/sam-3d-objects is gated — request it first, then:
cd third_party/sam-3d-objects
hf download --repo-type model --local-dir checkpoints/hf-download facebook/sam-3d-objects
mv checkpoints/hf-download/checkpoints checkpoints/hf && rm -rf checkpoints/hf-download
```

(~13 GB; `sam3d_infer.py` reads `checkpoints/hf/pipeline.yaml` by default.)
The env name is configurable via `backend.sam3d_env` in the yaml. Subprocess
results are cached inside each run dir (`stage3_object/sam3d/object.npz`,
`stage3_object/rc/poses.npz`, `stage7_contact_optim/jo/out.npz`,
`stage0_preprocess/vggt/geo.npz`) — delete a file to recompute that piece;
stage-level `--force` alone does not regenerate them.

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
├── mano/MANO_RIGHT.pkl   (+ MANO_LEFT.pkl)                  # MANO model — LICENSE-GATED, manual
│                          (mano/mano_v1_2/models/ archive layout also works)
├── vggt/model.pt                                            # optional: --depth vggt (else auto-DL)
└── da3/DA3METRIC-LARGE/                                     # optional: --depth da3 (else auto-DL)
```

SAM-3D-Objects weights live separately under
`third_party/sam-3d-objects/checkpoints/hf/` (§3b).

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
# THE VALIDATED CONFIGURATION (differentiable render-and-compare + joint
# articulated-grasp optimizer; needs both envs + MANO):
python -m hoi_recon.cli --video examples/grab.mp4 --out runs/grab --real \
    --hand hamer --object sam3d --depth moge --config configs/new.yaml

# MANO-free fallback (single env, older non-differentiable path):
python -m hoi_recon.cli --video clip.mp4 --out runs/clip01 --real \
    --hand depthlift --object sam3d --depth moge

hoi-recon-view --run runs/grab              # browser viewer of the 4D HOI
```

The run also writes reprojection-overlay validation videos
(`hand_reproj.mp4`, `object_reproj.mp4`, `hoi_reproj.mp4` + `*_grid.png`) into the
run dir, so you can check hand/object registration against the input video.

---

## What is verified vs. wired

| stage | backend | status |
|------|---------|--------|
| 0 depth + intrinsics | MoGe-2 (`--depth moge`) | ✅ verified — the validated path |
| 0 consistent camera + depth | VGGT (`--depth vggt`, sam3d env subprocess) | ⚙️ wired+validated, **up-to-scale**; metric-scale resolution in the optimizer is WIP |
| 0 depth + real camera poses | Depth-Anything-3 (`--depth da3`) | ⚙️ wired; clone+install DA3 (metric depth + real extrinsics; replaces ViPE) |
| 1 hand boxes | WiLoR YOLO | ✅ verified |
| 1 object mask | SAM 2.1 (point-prompt + propagate) | ✅ verified |
| 2 hand → MANO | `--hand hamer` (HaMeR; depth-anchored, MANO params threaded to stage 7) | ✅ verified (needs MANO) |
| 2 hand (MANO-free) | `--hand depthlift` (MoGe depth lift) | ✅ verified end-to-end |
| 3 object shape | SAM-3D-Objects textured mesh (sam3d env); fails soft to depth-lift hull | ✅ verified |
| 3 object 6D | `object_pose: render_compare` (silhouette tracker → differentiable refine); alternatives `silhouette` / `foundationpose` / `hand` | ✅ verified |
| 7 grasp optimization | joint MANO-articulation + object (`optim.differentiable: true`, sam3d env); fallback rigid `joint_grasp.py` | ✅ verified |
| 8 eval + reprojection overlays | this repo | ✅ verified |
| viewer | viser | ✅ verified |

The full `configs/new.yaml` pipeline is verified end-to-end on `examples/grab.mp4`
(the `runs/grab` result). `--hand depthlift` remains the no-license, single-env way
to run everything.

## Troubleshooting

- **`torch.cuda.is_available()` is False / "no kernel image"** — torch CUDA doesn't match
  the GPU. RTX 50xx needs `cu128` (torch ≥ 2.7). Reinstall torch with the right `--index-url`.
- **`--hand hamer` → "MANO model required … LICENSE-GATED"** — expected; place
  `MANO_RIGHT.pkl` (§4) or use `--hand depthlift`.
- **MANO `.pkl` load fails inside `chumpy` (`cannot import name 'bool' from numpy`)** —
  should not happen anymore: the repo patches the removed numpy aliases at runtime
  before chumpy imports (`_patch_numpy_for_chumpy`). If you still hit it, make sure
  stage 2 runs through `hoi_recon` (not a direct chumpy import beforehand).
- **`SAM-3D subprocess failed` / `render-compare failed` / `joint optimizer failed`** —
  the `sam3d-objects` env is missing or incomplete (§3b), or the entry script is
  absent from the cloned repo (re-run `bash scripts/setup_third_party.sh`, which
  installs them from `scripts/subprocess_entries/`). Re-run the printed
  `conda run -n sam3d-objects python ...` command by hand to see the real traceback.
  Stage 3 fails soft to depth-lift; stage 7 falls back to the rigid grasp optimizer
  only if `optim.differentiable` is off.
- **Stale subprocess results after changing meshes/poses** — subprocess outputs are
  cached per run dir (`sam3d/object.npz`, `rc/poses.npz`, `jo/out.npz`,
  `vggt/geo.npz`) and survive `--force`; delete the file to recompute.
- **SAM2 `cannot import name '_C'` warning** — benign (optional CUDA post-processing
  extension not built); masks are unaffected.
- **Object mask grabs the wrong thing** — SAM2 is prompted at the hand-box centre by
  default. Edit `_object_prompt` in `backends/real_perception.py`, or feed a click point.
- **0 contacts / large gap** — expected if the clip isn't a real grasp (hand and object
  far apart). Use a clip where the hand actually holds an object near it.
- **`--depth vggt` result looks wrong-scale** — expected for now: VGGT geometry is
  up-to-scale; use `--depth moge` for the validated metric result.
