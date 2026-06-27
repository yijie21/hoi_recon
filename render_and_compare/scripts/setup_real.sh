#!/usr/bin/env bash
# Install all real-backend dependencies into the hoi_recon conda env (GPU).
# Run AFTER: conda env create -f environment.yml  &&  bash scripts/setup_third_party.sh
#
#   conda activate hoi_recon
#   bash scripts/setup_real.sh
set -e
cd "$(dirname "$0")/.."

echo "## PyTorch (CUDA 12.8 — required for RTX 50xx / Blackwell sm_120)"
echo "   adjust the index-url to match your CUDA if you are on an older GPU"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo "## tooling"
pip install gdown "huggingface_hub"

echo "## MoGe-2  (depth + intrinsics, stage0)"
pip install -e third_party/MoGe

echo "## Depth-Anything-3  (optional: --depth da3 = metric depth + real camera poses)"
echo "   merges ViPE's camera-pose estimation; gives stage0 real extrinsics."
echo "   --no-deps avoids DA3's over-conservative numpy<2 pin; it runs fine on numpy 2."
if [ -d third_party/Depth-Anything-3 ]; then
  pip install --no-deps -e third_party/Depth-Anything-3
  pip install addict "moviepy==1.0.3" plyfile pycolmap evo e3nn open3d pillow_heif \
              typer omegaconf safetensors   # DA3 runtime deps (numpy-2 compatible)
else
  echo "   (clone DA3 first via setup_third_party.sh)"
fi

echo "## SAM 2  (object segmentation, stage1)"
pip install -e third_party/sam2

echo "## detection + meshing (dill: needed to load the WiLoR YOLO detector)"
pip install ultralytics trimesh dill

echo "## HaMeR / WiLoR  (hand, stage2) — runtime deps only; detectron2 NOT needed"
echo "   (we use WiLoR's YOLO boxes instead of HaMeR's detectron2 detector)"
pip install pytorch-lightning "smplx==0.1.28" yacs einops timm webdataset \
            scikit-image pyrender   # scikit-image + pyrender: HaMeR import deps

echo
echo "## verify the single env can run everything:"
echo "   python scripts/check_env.py"
echo "## NOTE: loading the official MANO .pkl needs chumpy, which does NOT build on"
echo "   numpy>=2 (this env). --hand depthlift runs without MANO; for --hand hamer/wilor"
echo "   use a patched chumpy or a dedicated numpy<1.24 env for stage 2."

cat <<'EOF'

done. Next:
  bash scripts/download_checkpoints.sh        # fetch public weights (MoGe/SAM2/WiLoR/HaMeR)
  # then place MANO_RIGHT.pkl (license) — see that script's final notes

MANO/chumpy caveat: the official MANO .pkl is loaded via `chumpy`, which needs
numpy<1.24. This env uses numpy>=2 (for MoGe/SAM2). If `--hand hamer` errors on a
numpy import inside chumpy, install a patched chumpy or run the hand stage in a
dedicated env. All non-hand stages work with numpy>=2.
EOF
