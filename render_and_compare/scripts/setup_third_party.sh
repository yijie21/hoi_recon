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
  "Depth-Anything-3|https://github.com/ByteDance-Seed/Depth-Anything-3"
  "vggt|https://github.com/facebookresearch/vggt"
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

# Any6D (taeyeopl/Any6D) — the FoundationPose/Any6D core for the fpauto object arm (and the
# any6dp learned core). It is imported directly (not a subprocess entry) and lives at the OUTER
# repo root ("<repo>/any6d"), where run_fp_hot3d.py / object_any6d.py expect it. It bundles
# foundationpose/; build its CUDA exts (mycpp, bundlesdf/mycuda) + fetch FP weights per the
# forehoi5090 recipe (REPRODUCE.md "Best method").
if should_clone any6d; then
  A6="$ROOT/../any6d"
  if [ -d "$A6/.git" ]; then
    echo "[skip] any6d already cloned"; ok+=("any6d")
  elif git clone $DEPTH https://github.com/taeyeopl/Any6D "$A6" 2>/dev/null; then
    echo "[clone] any6d  ->  $A6  (build FoundationPose CUDA exts + FP weights next)"; ok+=("any6d")
  else
    echo "  !! any6d clone failed — clone taeyeopl/Any6D manually into $A6"; fail+=("any6d")
  fi
fi

# Install this repo's subprocess entry scripts into the cloned repos. The pipeline
# invokes them via `conda run -n <sam3d_env> python third_party/<repo>/<script>.py`
# (they must live inside the repo: they import its packages and run with cwd=repo).
# Tracked source of truth: scripts/subprocess_entries/<repo>/*.py — third_party/ is
# gitignored, so without this copy a fresh clone cannot run the new.yaml pipeline.
ENTRIES="$ROOT/scripts/subprocess_entries"
if [ -d "$ENTRIES" ]; then
  echo
  for d in "$ENTRIES"/*/; do
    name="$(basename "$d")"
    if [ -d "$DST/$name" ]; then
      cp "$d"*.py "$DST/$name/"
      echo "[entry] installed $(ls "$d" | tr '\n' ' ')-> third_party/$name/"
    else
      echo "[entry] skip $name (repo not cloned)"
    fi
  done
fi

echo
echo "==== summary ===="
echo "cloned/present: ${ok[*]:-none}"
echo "failed:         ${fail[*]:-none}"
echo
echo "next: bash scripts/download_checkpoints.sh   (review URLs/licenses first)"
