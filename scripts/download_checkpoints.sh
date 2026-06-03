#!/usr/bin/env bash
# Download model weights into checkpoints/ for real mode.
#
# Downloads everything that is publicly fetchable (MoGe, SAM2, WiLoR detector,
# HaMeR) via hf / wget. MANO is LICENSE-GATED and cannot be auto-downloaded —
# you must register and place it yourself (instructions printed at the end).
#
#   bash scripts/download_checkpoints.sh            # download the public weights
#   bash scripts/download_checkpoints.sh --dry-run  # just print the plan
#
# Run inside the hoi_recon env (needs the `hf` CLI):  conda activate hoi_recon
set -u
cd "$(dirname "$0")/.." || exit 1
CK="$(pwd)/checkpoints"
mkdir -p "$CK"
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
run() { echo "+ $*"; [ $DRY -eq 1 ] || "$@"; }

echo "checkpoints -> $CK   (mode: $([ $DRY -eq 1 ] && echo DRY-RUN || echo DOWNLOAD))"
echo

# --- MoGe-2 (depth + intrinsics) -----------------------------------------
echo "## MoGe-2 (depth, stage0)"
run hf download Ruicheng/moge-2-vitl-normal --local-dir "$CK/moge/moge-2-vitl-normal"

# --- SAM 2.1 (object segmentation) ---------------------------------------
echo "## SAM 2.1 large (segmentation, stage1)"
run hf download facebook/sam2.1-hiera-large --local-dir "$CK/sam2/sam2.1-hiera-large"

# --- WiLoR YOLO hand detector --------------------------------------------
echo "## WiLoR detector + recon weights (hand detection, stage1/2)"
run hf download rolpotamias/WiLoR --local-dir "$CK/wilor"

# --- HaMeR (hand reconstruction) -----------------------------------------
echo "## HaMeR demo weights (hand, stage2)"
if [ ! -f "$CK/hamer/hamer_ckpts/checkpoints/hamer.ckpt" ]; then
  run wget -c "https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz" \
      -O "$CK/hamer/hamer_demo_data.tar.gz"
  run tar -xzf "$CK/hamer/hamer_demo_data.tar.gz" -C "$CK/hamer" --strip-components=1
else
  echo "  (already present)"
fi

cat <<EOF

==========================================================================
MANUAL (license-gated) — required for the hand stage (--hand hamer/wilor):

  MANO hand model:
    1. Register + accept the license at https://mano.is.tue.mpg.de
    2. Download MANO_v1_2 and copy the right-hand model to:
         $CK/mano/MANO_RIGHT.pkl
       (HaMeR also uses MANO_LEFT.pkl for left hands — copy it too.)

  NOTE: loading the official MANO .pkl needs `chumpy`, which requires numpy<1.24.
  If your env has numpy>=2 (default here), create a small side-env for the hand
  stage or use a patched chumpy — see README "real-mode notes".
==========================================================================
EOF
