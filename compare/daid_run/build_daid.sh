#!/bin/bash
# Layer do-as-i-do deps into the daid env (copied from forehoi; already has
# pytorch3d/kaolin/nvdiffrast/moge/spconv/smplx for sm_89, torch 2.5.1+cu121).
# Never bump torch/numpy: use --no-deps where risky, re-pin numpy at the end.
set +e
source /workspace/miniconda3/etc/profile.d/conda.sh; conda activate daid
cd /workspace/code/hoi_recon/do-as-i-do/reconstruction
export CUDA_HOME="$CONDA_PREFIX" CUDA_PATH="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST=8.9 FORCE_CUDA=1 MAX_JOBS=8
PY="python -m pip"
mark(){ echo "===== [$(date +%H:%M:%S)] $* ====="; }

mark "1. SAM3 package deps"
$PY install --no-deps -e modules/sam3 2>&1 | tail -3
$PY install fairscale fvcore yacs scikit-image numba submitit regex iopath omegaconf 2>&1 | tail -2
SITE="$(python -c 'import site;print(site.getsitepackages()[0])')"
mkdir -p "$SITE/assets" && cp -rn modules/sam3/assets/. "$SITE/assets/" 2>/dev/null; echo "copied CLIP bpe assets -> $SITE/assets"

mark "2. HaWoR deps"
$PY install --no-deps smplx ultralytics ultralytics-thop supervision lapx roma \
    pytorch-minimize mmengine hydra-colorlog webdataset natsort einops timm 2>&1 | tail -3
$PY install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html 2>&1 | tail -3
$PY install --no-build-isolation "chumpy @ git+https://github.com/mattloper/chumpy.git" 2>&1 | tail -3

mark "3. TAPIR (torch path)"
$PY install --no-deps -e modules/tapnet 2>&1 | tail -3
$PY install mediapy ffmpeg-python einshape dm-tree 2>&1 | tail -2

mark "4. geocalib (gravity)"
$PY install --no-deps "geocalib @ git+https://github.com/cvg/GeoCalib.git" 2>&1 | tail -3

mark "5. diff-gaussian-rasterization (sm_89, for SAM-3D texture baking)"
DG=/tmp/claude-1002/-workspace-code-hoi-recon/2580edc3-f5c5-491e-b66c-81be5f17a7a4/scratchpad/mip-splatting
if ! python -c "import diff_gaussian_rasterization" 2>/dev/null; then
  rm -rf "$DG"; git clone --recursive -q https://github.com/autonomousvision/mip-splatting.git "$DG" 2>&1 | tail -2
  cd "$DG/submodules/diff-gaussian-rasterization"
  python setup.py install 2>&1 | tail -5
  cd /workspace/code/hoi_recon/do-as-i-do/reconstruction
fi

mark "6. sam-3d-objects package (if installable)"
$PY install --no-deps -e modules/sam-3d-objects 2>&1 | tail -3
$PY install utils3d 2>&1 | tail -2
# un-shadow SAM3D's local notebook/ package
$PY uninstall -y notebook notebook-shim 2>&1 | tail -1

mark "7. re-pin numpy 1.26.4 (CUDA exts need it)"
$PY install "numpy==1.26.4" 2>&1 | tail -2

mark "IMPORT CHECK"
python - <<'PYEOF'
import importlib.util as u
mods=["torch","pytorch3d","kaolin","nvdiffrast","moge","spconv","smplx","torch_scatter",
      "diff_gaussian_rasterization","geocalib","chumpy","mediapy","sam3","tapnet","utils3d"]
for m in mods:
    print(f"  {m:30s} {'OK' if u.find_spec(m) else 'MISSING'}")
import numpy,torch; print("numpy",numpy.__version__,"torch",torch.__version__)
PYEOF
echo "BUILD_DAID_DONE"
