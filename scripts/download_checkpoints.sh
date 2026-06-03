#!/usr/bin/env bash
# Download model weights into checkpoints/.
#
# By default this script ONLY PRINTS what it would download (dry run), because
# several weights require accepting a license / registering an account and cannot
# be fetched blindly. Review every URL, then pass --run to actually download the
# items that have a direct URL.
#
#   bash scripts/download_checkpoints.sh           # dry run: print plan
#   bash scripts/download_checkpoints.sh --run     # download the direct-URL items
#
# Layout produced under checkpoints/:
#   mano/MANO_RIGHT.pkl                 (manual: register at mano.is.tue.mpg.de)
#   hamer/hamer_ckpts/                  (HaMeR provides a download script in-repo)
#   wilor/                              (WiLoR weights, often on HuggingFace)
#   sam2/sam2.1_hiera_large.pt
#   cotracker/scaled_offline.pth
#   moge/                              (MoGe weights on HuggingFace)
#   depth_anything_v2/depth_anything_v2_vitl.pth
#   sam-3d-objects/  foundationpose/  bundlesdf/   (per-repo instructions)
set -u
cd "$(dirname "$0")/.." || exit 1
CK="$(pwd)/checkpoints"
mkdir -p "$CK"
RUN=0; [ "${1:-}" = "--run" ] && RUN=1

# label | dest (relative to checkpoints/) | url   ('-' url == manual/registration)
ITEMS=(
  "MANO right hand model|mano/MANO_RIGHT.pkl|-|register + accept license at https://mano.is.tue.mpg.de , then place MANO_RIGHT.pkl here"
  "HaMeR weights|hamer/|-|run the in-repo downloader: cd third_party/hamer && bash fetch_demo_data.sh"
  "WiLoR weights|wilor/|https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt|"
  "SAM 2.1 large|sam2/sam2.1_hiera_large.pt|https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt|"
  "CoTracker3 offline|cotracker/scaled_offline.pth|https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth|"
  "Depth-Anything-V2 vitl|depth_anything_v2/depth_anything_v2_vitl.pth|https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth|"
  "MoGe-v2|moge/|-|see third_party/MoGe README (weights auto-download via HuggingFace on first run)"
  "SAM-3D-Objects|sam-3d-objects/|-|see third_party/sam-3d-objects README for weights"
  "FoundationPose|foundationpose/|-|download weights per https://github.com/NVlabs/FoundationPose (Google Drive)"
  "BundleSDF|bundlesdf/|-|BundleSDF needs LoFTR + nerf weights; see its README"
  "VIPE|vipe/|-|see third_party/vipe README for weights"
)

dl() {  # url dest
  local url="$1" dest="$2"
  mkdir -p "$(dirname "$dest")"
  if command -v wget >/dev/null 2>&1; then wget -c -O "$dest" "$url"
  else curl -L -C - -o "$dest" "$url"; fi
}

echo "checkpoints dir: $CK"
echo "mode: $([ $RUN -eq 1 ] && echo DOWNLOAD || echo 'DRY RUN (pass --run to download direct-URL items)')"
echo
for it in "${ITEMS[@]}"; do
  IFS='|' read -r label dest url note <<< "$it"
  echo "• $label  ->  checkpoints/$dest"
  if [ "$url" = "-" ]; then
    echo "    MANUAL: $note"
  else
    echo "    URL: $url"
    if [ $RUN -eq 1 ]; then
      echo "    downloading..."
      dl "$url" "$CK/$dest" && echo "    done." || echo "    FAILED (check URL/access)."
    fi
  fi
  echo
done
echo "Review licenses before use. MANO and some weights require registration."
