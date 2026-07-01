# Running HOLD on this machine

HOLD (CVPR'24): category-agnostic joint hand + object reconstruction from a monocular
video. This file documents the **exact, reproducible** setup that was installed and
verified on this box, what runs, what is blocked, and the precise output format a
downstream viewer (e.g. viser) should consume.

TL;DR
- **Env: built and verified.** `conda env = hold`, python = `/workspace/miniconda3/envs/hold/bin/python`.
- **Training loop / volumetric rendering / marching-cubes meshing / checkpointing: VERIFIED to run end-to-end on the RTX 4090** (a structurally-valid *synthetic* smoke-test sequence — see §5).
- **A *real* reconstruction is BLOCKED** on input data: the preprocessed sequences and
  the preprocessing model weights are behind the MPI account wall
  (`download.is.tue.mpg.de`, returns HTTP 401). No public mirror exists. See §6.
- Use `CUDA_VISIBLE_DEVICES=1` (sibling tasks use GPU 0).

---

## 1. Environment (already created, reproducible)

Created as conda env **`hold`** (python 3.10). Key versions, chosen so the stack runs
on the RTX 4090 (Ada / sm_89, which needs CUDA >= 11.8, so the repo's original
torch 1.9 / cu111 is impossible here):

| package | version | why |
|---|---|---|
| python | 3.10 | wheels for kaolin/pytorch3d |
| torch / torchvision | 2.1.0+cu118 / 0.16.0+cu118 | first CUDA that supports sm_89 4090 |
| pytorch-lightning | **1.9.5** | HOLD's `common/abstract_pl.py` uses the PL 1.9 hook API (`on_validation_epoch_end(self)` + manual `val_step_outputs`); `train.py` uses `gpus=1` (removed in PL 2.0). 1.9.x is the only compatible line. (setup.md's "1.5.7" is wrong for this code.) |
| kaolin | 0.15.0 (wheel for torch2.1.0_cu118) | `check_sign`, `point_to_mesh_distance`, `index_vertices_by_faces` |
| pytorch3d | 0.7.5 (prebuilt py310_cu118_pyt210) | knn / chamfer / Meshes |
| numpy | **1.23.5** (pinned, hard) | chumpy imports `np.bool/np.float` (removed in numpy>=1.24); the mise C-extension is compiled against 1.23 |
| scikit-image | 0.25.2 | `marching_cubes` (the old `marching_cubes_lewiner` call was patched — see §7) |
| smplx, chumpy, pygit2, comet-ml==3.40.0, kornia==0.6.12, open3d, pymeshlab, trimesh, omegaconf, loguru, opencv | as resolved | repo requirements |

The custom **mise** C-extension is built in place at
`code/src/libmise/mise.cpython-310-x86_64-linux-gnu.so`.

### Recreate from scratch (if ever needed)
```bash
source /workspace/miniconda3/etc/profile.d/conda.sh
conda create -y -n hold python=3.10 && conda activate hold
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install "numpy==1.23.5"
pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.0_cu118.html
pip install fvcore iopath
pip install --no-index pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt210/download.html
pip install "pytorch-lightning==1.9.5" "setuptools==69.5.1" "cython==0.29.36" \
    easydict ipdb "kornia==0.6.12" loguru matplotlib open3d \
    opencv-contrib-python-headless opencv-python pyparsing scikit-learn tqdm trimesh \
    omegaconf pymeshlab scikit-image "comet-ml==3.40.0" smplx pygit2
pip install --no-build-isolation chumpy
pip install "numpy==1.23.5"   # re-pin: later installs bump it to 2.x
cd /workspace/code/hoi_recon/hold/code && python setup.py build_ext --inplace
```

### Body models (assembled from local assets — no new MANO license needed)
HOLD expects `code/body_models/`. Already populated:
```
code/body_models/MANO_RIGHT.pkl     # copied from hort/mano_data/mano/MANO_RIGHT.pkl
code/body_models/contact_zones.pkl  # copied from easyhoi/assets/contact_zones.pkl
```
Notes:
- **Right hand only.** No `MANO_LEFT.pkl` is available on this box, so only single-right-hand
  sequences are supported. Two-hand (ARCTIC / bimanual) needs `MANO_LEFT.pkl`.
- `sealed_vertices_sem_idx.npy` (in the MPI "mandatory" bundle) is **not referenced**
  by the training code — not needed. Wrist sealing uses hard-coded faces in
  `common/body_models.py`.
- The pretrained hand-shape net `--shape_init` (`5c09be8ac` / `75268d864`) is
  account-walled but **optional**: pass `--shape_init ""` to skip it
  (`hold_net.py:152` → "Skipping INIT human models"). Default is `75268d864`, which
  does not exist locally, so you MUST pass `--shape_init ""` unless you obtain it.

### Required env vars at runtime
```bash
export CUDA_VISIBLE_DEVICES=1
export COMET_API_KEY=dummy COMET_WORKSPACE=dummy   # parser.py reads these even with --mute
```

---

## 2. What HOLD expects as input

Training reads a **prebuilt** dataset at `code/data/<seq>/build/`:
```
code/data/<seq>/build/
  image/*.png            # frames, sorted by name (any size)
  mask/*.png             # OPTIONAL grayscale segm: 0=bg, 50=object, 150=right, 250=left
  data.npy               # dict (see below); REQUIRED
  corres.txt             # orig frame names (optional)
```
`data.npy` (`np.load(...).item()`), keys (per `docs/data_doc.md` + verified against loaders):
```
seq_name : str
scene_bounding_sphere : float
max_radius_ratio : float
cameras : { scale_mat_<i>:4x4 , world_mat_<i>:4x4 }     # for i in 0..T-1; P = world_mat@scale_mat
entities.right  : { hand_poses:Tx48(aa, 3 global+45), hand_trans:Tx3, mean_shape:(10,) }
entities.object : { object_poses:Tx6(3 aa rot + 3 trans), obj_scale:float,
                    pts.cano:Nx3 (canonical SfM cloud), norm_mat:4x4 }
```
This `build/` is produced by either the account-walled download **or** the custom
preprocessing pipeline (`docs/custom.md`). See §6.

---

## 3. The run command (single-stage)

```bash
cd /workspace/code/hoi_recon/hold/code
source /workspace/miniconda3/etc/profile.d/conda.sh && conda activate hold
export CUDA_VISIBLE_DEVICES=1 COMET_API_KEY=dummy COMET_WORKSPACE=dummy
seq_name=<your_seq>                      # e.g. hold_bottle1_itw once data is obtained
python train.py --case $seq_name --shape_init "" \
       --num_epoch 100 --eval_every_epoch 1 --num_sample 64 --mute
```
- `--mute` = no online Comet logging. `--num_sample` down to 8 if OOM (default sampling fine on 49 GB 4090).
- A random 9-char `exp_id` is assigned; everything is written under `code/logs/<exp_id>/`.
- `--eval_every_epoch` controls BOTH validation and checkpoint cadence (default 6).
  Canonical meshes are dumped at validation epochs where `epoch>0 and epoch%3==0`.

### Full 3-stage HOLD pipeline (paper-accurate; needs `--shape_init` weights)
```bash
python train.py --case $seq_name --num_epoch 100 --shape_init <id>          # -> exp_id A
python optimize_ckpt.py --batch_size 51 --iters 300 --ckpt_p logs/A/checkpoints/last.ckpt
python train.py --case $seq_name --num_epoch 200 --load_pose logs/A/checkpoints/last.pose_ref --shape_init <id>
```

### Render full sequence to images / mp4 (after training)
```bash
python render.py --case $seq_name --load_ckpt logs/<exp_id>/checkpoints/last.ckpt --mute --agent_id -1 --render_downsample 4
bash ./create_videos.sh <exp_id>
```

---

## 4. OUTPUT location & FORMAT (for the viewer task)

Everything for a run is under **`code/logs/<exp_id>/`**:
```
args.json                         # run args, git commit/branch
train.log                         # loguru log
checkpoints/last.ckpt             # + {epoch:04d}-{loss}.ckpt per eval epoch
checkpoints/last.pose_ref         # only from optimize_ckpt.py
mesh_cano/mesh_cano_right_step_<S>.obj    # CANONICAL hand mesh (.obj), dumped at epoch%3==0
mesh_cano/mesh_cano_object_step_<S>.obj   # CANONICAL object mesh (.obj), marching-cubes of SDF
misc/<global_step:09d>.npy        # dict for visualize_ckpt.py / pose refinement
visuals/<key>/...png              # only if rendering enabled (omit --no_vis)
```

### How a viewer reconstructs per-frame meshes
**There are NO per-frame hand/object `.obj` files on disk.** Per-frame geometry is
derived on the fly from the checkpoint. The `.obj` files in `mesh_cano/` are
*canonical* templates only. Recipe (mirrors `code/src/utils/io/ours.py`):

1. Load `logs/<exp>/checkpoints/last.ckpt` → `ckpt["state_dict"]`.
   Per-frame pose params live as `nn.Embedding` weights:
   - Hand: `model.nodes.right.params.global_orient.weight` (Tx3 aa),
     `...params.pose.weight` (Tx45 aa), `...params.transl.weight` (Tx3),
     `...params.betas.weight` (1x10, shared).
   - Object: `model.nodes.object.params.global_orient.weight` (Tx3 aa),
     `...params.transl.weight` (Tx3); scale buffer
     `model.nodes.object.server.object_model.obj_scale` (scalar).
2. Load latest `logs/<exp>/misc/*.npy` for `K` (4x4 intrinsics), `w2c`,
   `scale` (scene→metric scalar), `img_paths`, and the canonical object trimesh
   (`object_cano`).
3. Hand verts: feed params through the MANO server (`src/model/mano/server.py`,
   `use_pca=False, flat_hand_mean=False`) → **T×778×3** verts; faces = MANO
   `server.faces` = **1538×3** (UNSEALED — sealing to 779/1554 is internal only,
   not applied to exported verts).
4. Object verts: canonical object mesh (N verts) posed by `ObjectModel.forward`
   (`src/model/obj/object_model.py`): `scale_mat·pose·obj_scale·denorm_mat` applied to
   canonical verts, `denorm_mat = inv(norm_mat)` → **T×N×3**; object faces constant
   = `object_cano.faces`.
5. Optional `map_deform2eval` (`ours.py:15-29`) to land in metric **OpenCV camera frame**:
   multiply by `1/scale`, flip Y,Z via `diag(1,-1,-1)`, add `normalize_shift`.

**Coordinate frame / units:** server output is the normalized scene frame scaled back to
**metres** (`verts*scene_scale`, `scene_scale = 1/scale_mat[0,0]`). After
`map_deform2eval` the meshes are in **metres, OpenCV camera frame** (camera at origin
looking +Z), matching the HO3D eval convention. Camera for the viewer: intrinsics `K`
from `misc`, extrinsics ≈ identity with a Y/Z flip.

**Summary for the viewer:** consume `last.ckpt` + latest `misc/*.npy`; produce
`T×V×3` vertex arrays (hand 778, object N) with constant face arrays (hand 1538,
object from `object_cano`). If you want baked per-frame `.obj/.ply`, that export does
not exist — add a trimesh `.export` loop around the step-3/4 outputs.

---

## 5. Verified smoke test (synthetic — NOT a real reconstruction)

To prove the env + GPU + full optimization/meshing/checkpoint path work without the
account-walled data, a **structurally-valid but FAKE** sequence was generated from the
`/tmp/wild6` frames (zero MANO pose, dummy camera, sphere point cloud):
```bash
cd /workspace/code/hoi_recon/hold/code
python scripts_smoke/make_synthetic_dataset.py        # -> data/hold_smoke_synthetic/build/
export CUDA_VISIBLE_DEVICES=1 COMET_API_KEY=dummy COMET_WORKSPACE=dummy
python train.py --case hold_smoke_synthetic --shape_init "" \
       --num_epoch 4 --eval_every_epoch 1 --mute --no_vis --num_sample 8
```
Result (verified, exp `a823bb4d1`): trains at ~4 it/s on GPU 1, loss 1.39 → 0.28,
4 epochs in ~7 min. Files actually produced under `code/logs/a823bb4d1/`:
```
checkpoints/last.ckpt                          (+ epoch=0000..0003 ckpts, 28 MB each)
mesh_cano/mesh_cano_object_step_1600.obj       (object canonical mesh: 45373 v / 90090 f)
mesh_cano/mesh_cano_object_step_misc.obj
misc/000001600.npy                             (keys: object.obj_scale, img_paths, K[4x4], w2c[4x4], scale, object_cano)
args.json, train.log
```
Checkpoint `state_dict` keys confirmed exactly as documented in §4
(`model.nodes.right.params.{global_orient(12,3),pose(12,45),transl(12,3),betas(1,10)}`,
`model.nodes.object.params.{global_orient,transl}(12,3)`,
`model.nodes.object.server.object_model.obj_scale(1,)`; T=12 frames).

**Caveats (honest):** the geometry is meaningless (fake inputs); this only validates the
end-to-end code path + output format. The **object** mesh extracts fine (VolSDF
geometry-init gives a valid sphere SDF). The **canonical hand** mesh logged
"Failed to mesh out right" — with the fake zero-pose hand + untrained SDF after only 4
epochs there is no SDF zero-crossing for marching cubes; this is a data/convergence
artifact, not an env bug, and resolves on a real sequence (proper `--shape_init` and
more epochs). Generator: `code/scripts_smoke/make_synthetic_dataset.py`.

---

## 6. BLOCKERS and the exact remaining work for a REAL reconstruction

**Blocker: input data is account-walled.** All real assets download from
`download.is.tue.mpg.de/download.php?domain=hold&...` which requires a registered HOLD
account (verified: HTTP 401 "Username/Password wrong" without creds). No public mirror
found. This covers: preprocessed `build/` sequences (incl. `hold_bottle1_itw`),
pretrained hand-shape nets (`5c09be8ac`, `75268d864`), and the MPI MANO bundle.

You have two ways to get a real `build/` dataset:

### Path A — download a preprocessed sequence (fastest, needs a HOLD account)
Register at https://hold.is.tue.mpg.de/register.php, then:
```bash
cd /workspace/code/hoi_recon/hold
export HOLD_USERNAME=... HOLD_PASSWORD=... MANO_USERNAME=... MANO_PASSWORD=...
conda activate hold
bash ./bash/download_data.sh           # preprocessed sequences (data.txt)
bash ./bash/download_mandatory.sh      # shape-init nets + contact_zones + MANO
python scripts/unzip_download.py
bash ./bash/setup_files.sh             # populates code/data/, code/saved_models/, code/body_models/
# then: python train.py --case hold_bottle1_itw --eval_every_epoch 1 --num_sample 64 --mute
```

### Path B — preprocess a custom clip (e.g. /tmp/wild6, NO HOLD account needed, but heavy)
`docs/custom.md` pipeline → produces `build/` from raw frames. Requires the
`generator/` submodules (already cloned) and **4 extra conda envs**
(`generator/install/*.sh`): SAM-Track (segmentation), Hierarchical-Localization/COLMAP
(object SfM pose), and METRO **or** HAMER (hand pose). Their model weights are largely
public (SAM = Meta, HAMER = HuggingFace), so this path avoids the HOLD account, but is a
multi-hour multi-env setup and was **not** completed here. Outline:
```bash
cd generator && bash ./install/conda.sh
# segmentation (sam-track), hand pose (metro/hamer), object SfM (hloc):
#   docs/custom.md §Segmentation, §Hand pose estimation, §Object pose estimation
python scripts/build_dataset.py --seq_name <seq> --rebuild --no_fixed_shift   # -> build/data.npy
```
Right-hand-only is fine (use METRO, or HAMER with `--hand_type right`).

---

## 7. Source patches applied (to make the repo run on this stack)
Minimal, behaviour-preserving:
- `code/src/utils/meshing.py`: use `measure.marching_cubes` when
  `marching_cubes_lewiner` is absent (skimage ≥ 0.19 removed it).
- `common/torch_utils.py` `one_hot_embedding`: move the identity matrix to
  `labels.device` (torch 2.x forbids cross-device indexing; older torch tolerated it).
- `code/scripts_smoke/make_synthetic_dataset.py`: **new**, smoke-test data generator (§5).
