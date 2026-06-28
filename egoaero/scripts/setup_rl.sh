#!/usr/bin/env bash
# Install the SP2 RL stack and vendor the Shadow Hand model. Run once.
set -euo pipefail
python -m pip install "mujoco>=3" "stable-baselines3>=2" "gymnasium>=1"
# torch with CUDA (box has RTX 4090s); falls back to default index if cu wheels unavailable
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124 || python -m pip install torch
DEST="$(cd "$(dirname "$0")/.." && pwd)/assets/shadow_hand"
if [ ! -f "$DEST/right_hand.xml" ]; then
  TMP="$(mktemp -d)"
  git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie "$TMP/mm"
  mkdir -p "$DEST"
  cp -r "$TMP/mm/shadow_hand/." "$DEST/"
  rm -rf "$TMP"
fi
echo "RL stack installed; Shadow Hand vendored at $DEST"
