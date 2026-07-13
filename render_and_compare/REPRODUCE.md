# Reproducing the real (GPU) pipeline on another machine

## ⭐ Best method (current) — full 4D HOI (object + hand)

The current best result pairs the best **object** track with the best **hand**:
**object → `fpauto`** (FoundationPose auto, drift-gated — see the plain-language
[GLOSSARY.md](../GLOSSARY.md) for what the method codes mean) and **hand → the
hand-reprojection optimizer** (`joint_opt.py --freeze_object`, kp2d-aligned). Numbers + the
exact copy-paste run recipe live at the top of the repo [`README.md`](../README.md); design in
[`docs/adr/0001-hand-reprojection-optimizer.md`](docs/adr/0001-hand-reprojection-optimizer.md)
and [`../compare/hot3d/docs/T6_NOTES.md`](../compare/hot3d/docs/T6_NOTES.md).

**Prerequisites on a fresh clone** (once):

1. **Conda envs** (Blackwell / sm_120): `rc5090` (pipeline+eval+overlays; `scripts/setup_real.sh`),
   `sam3d5090` (SAM-3D mesh **and** the hand optimizer;
   [`scripts/subprocess_entries/sam-3d-objects/BLACKWELL_ENV.md`](scripts/subprocess_entries/sam-3d-objects/BLACKWELL_ENV.md)),
   `forehoi5090` (FoundationPose/Any6D core = clone of `sam3d5090` + Any6D's FoundationPose CUDA
   exts `mycpp` + `bundlesdf/mycuda` built for sm_120).
2. **Checkpoints** — `scripts/download_checkpoints.sh` (MoGe, SAM2, WiLoR, HaMeR, SAM-3D).
3. **MANO** (license-gated) — place `MANO_RIGHT.pkl` at `checkpoints/mano/`. The hand driver
   auto-creates the chumpy-free `MANO_RIGHT_np.pkl` it needs (sam3d5090 has no chumpy); or run
   `python scripts/convert_mano_chumpy_free.py` yourself.
4. **Third-party** — `scripts/setup_third_party.sh` clones the method repos **and deploys the
   tracked subprocess entries** (including the `--freeze_object` `joint_opt.py` — this is why the
   edit lives in `scripts/subprocess_entries/…`, not the gitignored `third_party/`). For `fpauto`
   also clone **Any6D** at the repo root and build its FP CUDA exts: run `setup_third_party.sh`
   (it now clones `any6d`), then build per the `forehoi5090` recipe + fetch the FP weights.
5. **HOT3D data** — bench clips under `/workspace/datasets/hot3d/` (adapter `compare/hot3d/make_rc_input.py`).

Then run the README recipe. The single-command `combined.yaml` pipeline below is the older
best-of-both arm, kept for reference.

---

End goal — **the best-performance configuration** (the result in `runs/grab_combined`):

```bash
python -m hoi_recon.cli --video examples/grab.mp4 --out runs/grab_combined --real \
    --config configs/combined.yaml
hoi-recon-view --run runs/grab_combined               # view the 4D result in a browser
```

`--config configs/combined.yaml` is the best-of-both pipeline: SAM-3D textured
object mesh, **render-and-compare** object 6D (silhouette + photometric), the
**CHOIR hand isolated fit (Eq 1)** in stage 2 for a much cleaner hand init, and the
**joint MANO-articulation + object** grasp optimizer in stage 7. It is
`configs/new.yaml` (the prior validated path) **plus** the one CHOIR hand-fit
addition; `new.yaml` still works if you want the baseline without the hand fit.
Either way you need **two conda envs**: the main `hoi_recon` env (§1–3) plus a
`sam3d-objects` env (§3b) that hosts the PyTorch3D/SAM-3D components, run as cached
subprocesses. (`configs/choir.yaml` is a separate A/B-study config that reproduces
CHOIR's coarse init only — not for production.) Without any `--config` you get the
oldest path (silhouette-only object rotation, rigid non-articulated grasp), which
runs in the single main env:

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
# BEST PERFORMANCE (render-and-compare object + CHOIR hand fit + joint
# articulated-grasp optimizer; needs both envs + MANO):
python -m hoi_recon.cli --video examples/grab.mp4 --out runs/grab_combined --real \
    --config configs/combined.yaml

# Prior validated baseline (same, minus the CHOIR hand fit):
python -m hoi_recon.cli --video examples/grab.mp4 --out runs/grab --real \
    --hand hamer --object sam3d --depth moge --config configs/new.yaml

# MANO-free fallback (single env, older non-differentiable path):
python -m hoi_recon.cli --video clip.mp4 --out runs/clip01 --real \
    --hand depthlift --object sam3d --depth moge

hoi-recon-view --run runs/grab_combined     # browser viewer of the 4D HOI
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
| 2 hand → MANO | `--hand hamer` (HaMeR; depth-anchored, MANO params threaded to stage 7); `configs/combined.yaml` adds the CHOIR isolated fit (Eq 1) | ✅ verified (needs MANO) |
| 2 hand (MANO-free) | `--hand depthlift` (MoGe depth lift) | ✅ verified end-to-end |
| 3 object shape | SAM-3D-Objects textured mesh (sam3d env); fails soft to depth-lift hull | ✅ verified |
| 3 object 6D | `object_pose: render_compare` (silhouette tracker → differentiable refine); alternatives `silhouette` / `foundationpose` / `hand` | ✅ verified |
| 7 grasp optimization | joint MANO-articulation + object (`optim.differentiable: true`, sam3d env); fallback rigid `joint_grasp.py` | ✅ verified |
| 8 eval + reprojection overlays | this repo | ✅ verified |
| viewer | viser | ✅ verified |

The full pipeline is verified end-to-end on `examples/grab.mp4`: `configs/combined.yaml`
(the best-performance `runs/grab_combined` result) and `configs/new.yaml` (the prior
`runs/grab` baseline). `--hand depthlift` remains the no-license, single-env way to run
everything.

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

## Running specific configurations

Extra notes from a real run on a **non-Blackwell** GPU (RTX 4090, `sm_89`) — useful if you're
not on the RTX 5090/Blackwell boxes this guide otherwise assumes.

### Single conda env instead of two

On a pre-Blackwell GPU, the `cu121` stack works, so the two-env split (§3 / §3b) collapses
into **one** env that holds both the main pipeline and the `sam3d-objects` subprocess side
(set `backend.sam3d_env` to that same env name in the config). This was verified on a
conda env named `forehoi` (torch 2.5.1+cu121, python 3.11) holding MoGe, SAM2, ultralytics,
HaMeR runtime deps, chumpy, PyTorch3D, kaolin, and the SAM-3D-Objects fork:

```bash
conda activate forehoi                       # single env holds everything
cd render_and_compare
export HF_HOME=/workspace/huggingface_cache/
export CUDA_VISIBLE_DEVICES=1                 # pick the GPU with the most free VRAM (>=20GB)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m hoi_recon.cli --video examples/wild6_trim.mp4 \
    --out runs/wild6_real --real --config configs/real_forehoi.yaml
```

`configs/real_forehoi.yaml` uses the in-process silhouette object-pose tracker and the numpy
grasp optimizer (robust, no heavy PyTorch3D subprocess). For the full differentiable path
(render-compare object pose + joint MANO-articulation optimizer) in the same single env, use
`configs/combined_forehoi.yaml` — identical to `configs/combined.yaml` except
`sam3d_env: forehoi`.

### SAM-3D / utils3d version conflict (single-env setups)

MoGe-v2 (main pipeline, stage 0) needs the **new** `utils3d` (`utils3d.pt`), while
`sam3d_objects` needs the **old** `utils3d` (`utils3d.numpy.depth_edge`) and its internal
depth model imports **moge 1.0.0** (`utils3d.torch`) — mutually exclusive in one
site-packages. Since the SAM-3D subprocess runs with `cwd = third_party/sam-3d-objects`, drop
**local package shadows** there so the subprocess (only) sees the old stack, while the main
process keeps the new one:

- `third_party/sam-3d-objects/utils3d/` = utils3d 0.0.2
- `third_party/sam-3d-objects/moge/` = moge 1.0.0

`sys.path[0]` (the script dir) wins over site-packages, so no env is polluted — this is what
makes SAM-3D run for real inside a single collapsed env.

### GPU memory ordering (avoiding OOM in the differentiable subprocesses)

The SAM-3D / render-compare / joint-opt subprocesses each need ~13 GB. If stages 0–2 run **in
the same process** first, MoGe+SAM2+HaMeR stay resident and SAM-3D can OOM (it then falls back
to the depth-lift hull, and the differentiable path is silently skipped). The robust recipe is
to run stages 0–2 once, then re-invoke with `--stages 3-8` so the main process is light and
the GPU subprocesses get the full card:

```bash
python -m hoi_recon.cli --video examples/wild6_trim.mp4 --out runs/wild6_combined \
    --real --config configs/combined_forehoi.yaml --stages 0-2
python -m hoi_recon.cli --video examples/wild6_trim.mp4 --out runs/wild6_combined \
    --real --config configs/combined_forehoi.yaml --stages 3-8
```

(Or pin `CUDA_VISIBLE_DEVICES` to the emptier GPU.)

### Output file reference

Besides the reprojection-overlay videos (§6), the run dir holds (example shapes from a
74-frame run):

- **`stage8_eval/pseudo_gt.npz`** — `hand_verts[T,778,3]`, `hand_joints[T,21,3]`,
  `obj_verts[V,3]`, `obj_faces[F,3]`, `obj_poses[T,4,4]`, `contact_map[T,285]`,
  `rectify_delta`, `object_delta`.
- **`stage7_contact_optim/arrays.npz`** — the same 4D HOI plus `obj_colors[V,3]` and
  `hand_faces[1538,3]` (textured object + MANO topology).
- **`stage8_eval/report.json`** — self-consistency diagnostics.
