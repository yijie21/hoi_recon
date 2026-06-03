#!/usr/bin/env bash
# Clone third-party model repos into third_party/.
# Code only — model weights are downloaded separately (download_checkpoints.sh).
#
# Usage:
#   bash scripts/setup_third_party.sh                 # clone the default set (shallow)
#   bash scripts/setup_third_party.sh hamer sam2      # clone only the named repos
#   FULL=1 bash scripts/setup_third_party.sh          # full history (no --depth 1)
#
# Notes:
#   * Each model has its OWN heavy deps (torch/CUDA/pytorch3d/nvdiffrast). Install
#     them inside each repo per its README; see requirements-backends.txt.
#   * Some URLs may move or require access — verify before relying on them.
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"
DST="$ROOT/third_party"
mkdir -p "$DST"

DEPTH="--depth 1"
[ "${FULL:-0}" = "1" ] && DEPTH=""

# name|url   (name is the folder created under third_party/)
REPOS=(
  "hamer|https://github.com/geopavlakos/hamer"
  "WiLoR|https://github.com/rolpotamias/WiLoR"
  "Dyn-HaMR|https://github.com/ZhengdiYu/Dyn-HaMR"
  "HaWoR|https://github.com/ThunderVVV/HaWoR"
  "sam2|https://github.com/facebookresearch/sam2"
  "co-tracker|https://github.com/facebookresearch/co-tracker"
  "sam-3d-objects|https://github.com/facebookresearch/sam-3d-objects"
  "BundleSDF|https://github.com/NVlabs/BundleSDF"
  "FoundationPose|https://github.com/NVlabs/FoundationPose"
  "MoGe|https://github.com/microsoft/MoGe"
  "Depth-Anything-V2|https://github.com/DepthAnything/Depth-Anything-V2"
  "vipe|https://github.com/nv-tlabs/vipe"
)

want=("$@")
should_clone() {
  [ ${#want[@]} -eq 0 ] && return 0
  for w in "${want[@]}"; do [ "$w" = "$1" ] && return 0; done
  return 1
}

ok=(); fail=()
for entry in "${REPOS[@]}"; do
  name="${entry%%|*}"; url="${entry##*|}"
  should_clone "$name" || continue
  target="$DST/$name"
  if [ -d "$target/.git" ]; then
    echo "[skip] $name already cloned"
    ok+=("$name"); continue
  fi
  echo "[clone] $name  <-  $url"
  if git clone $DEPTH "$url" "$target" 2>/dev/null; then
    ok+=("$name")
  else
    echo "  !! failed (URL moved / access required?) — clone manually into $target"
    fail+=("$name")
  fi
done

echo
echo "==== summary ===="
echo "cloned/present: ${ok[*]:-none}"
echo "failed:         ${fail[*]:-none}"
echo
echo "next: bash scripts/download_checkpoints.sh   (review URLs/licenses first)"
