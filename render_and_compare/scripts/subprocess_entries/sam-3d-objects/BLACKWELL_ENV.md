# SAM-3D Objects on RTX 5090 (Blackwell, sm_120) — env `sam3d5090`

Built 2026-07-07. The official pins (torch 2.5.1+cu121) predate Blackwell; this
env deviates to torch 2.8.0+cu128 and rebuilds/substitutes every CUDA dep.
Verified end-to-end: `sam3d_infer.py --no-texture` on kettle_N15 frame 58
produced a correct kettle mesh (~30 s inference after weight load) on GPU 0.

## Recipe (as executed)

1. `conda create -n sam3d5090 -c conda-forge python=3.11 gcc_linux-64=13.* \
    gxx_linux-64=13.* cmake ninja git`
   `conda install -n sam3d5090 -c nvidia/label/cuda-12.8.1 cuda-toolkit`
2. `pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
    --index-url https://download.pytorch.org/whl/cu128`
3. CUDA deps (each verified with a real GPU op, not just import):
   - **spconv-cu126** wheel WORKS on sm_120 (SubMConv3d forward verified) —
     no source build needed. (cu128 wheel did not exist.)
   - **kaolin 0.18.0** from the NVIDIA index
     `-f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html`
     (FlexiCubes import verified).
   - **pytorch3d @ 75ebeea** (repo pin) + **gsplat @ 2323de5** (repo pin) built
     from source with `TORCH_CUDA_ARCH_LIST="12.0" FORCE_CUDA=1
     CUDA_HOME=$ENV CPATH=$ENV/targets/x86_64-linux/include
     LIBRARY_PATH=$ENV/targets/x86_64-linux/lib` — the CPATH/LIBRARY_PATH lines
     are required: conda's cuda-toolkit keeps headers under targets/, and the
     build otherwise dies with `cuda_runtime_api.h: No such file`.
4. `requirements.txt` filtered (CUDA pins + cloud/dev tooling removed), then
   `pip install --no-deps -e .`, then `python patching/hydra`.
5. Post-fixes discovered by the smoke test:
   - `seaborn==0.13.2` (lives in requirements.inference.txt),
   - `numpy==1.26.4` (pinned opencv-python 4.9 is numpy-1 ABI),
   - `setuptools<81` (pinned lightning 2.3.3 imports pkg_resources).
   - flash_attn / xformers NOT installed: attention defaults to torch sdpa
     (see modules/attention/__init__.py; override via ATTN_BACKEND if desired).
   - `sam3d_objects.init` is Meta-internal and absent from the public repo;
     notebook/inference.py sets LIDRA_SKIP_INIT=true itself, so imports work.

## Checkpoints

`checkpoints/hf/` = facebook/sam-3d-objects snapshot (13 GB, gated — this
account has access). NOTE: download with `HF_HUB_DISABLE_XET=1` — the xet
path silently stalled; plain HTTP pulled 13 GB in ~1 min.

## Pipeline wiring

`configs/real_forehoi{,_icp}.yaml` now set `backend.sam3d_env: sam3d5090`;
stage 3 invokes `sam3d_infer.py` in this env via `conda run`. The smoke-test
mesh (vs the archived kettle_gt mesh, which was flat/tray-like) is visibly
better — see scratchpad mesh_compare.png from the build session.
