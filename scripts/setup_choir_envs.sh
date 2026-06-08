#!/usr/bin/env bash
# Build the two extra conda envs for FULL-FAITHFUL CHOIR coarse reproduction:
#   * dynhamr  — Dyn-HaMR 4D hand stabilization (torch 1.13 / cu117 + DROID-SLAM)
#   * vipe     — VIPE camera trajectory (cu128 + lietorch custom ops)
#
# These are heavy, CUDA-compiling builds that can fail on driver/toolkit mismatch.
# The CHOIR coarse path runs WITHOUT them (graceful fallback: HaMeR + isolated fit
# for the hand, identity extrinsics for the camera) — this script upgrades it to
# full fidelity. Run from the repo root. Inspect/runs steps individually if a build
# fails; nothing here is required for the algorithmic CHOIR comparison.
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"
CONDA="${CONDA_EXE:-conda}"

echo "==== 1/3  install subprocess entry scripts into the clones ===="
bash scripts/setup_third_party.sh vipe Dyn-HaMR

echo "==== 2/3  VIPE env (cu128) ===="
# Easiest path: a fresh py3.10 env + pip install nvidia-vipe (auto-builds CUDA ops).
if ! $CONDA env list | grep -q '^vipe '; then
  $CONDA create -y -n vipe python=3.10 || echo "!! vipe env create failed"
  $CONDA run -n vipe pip install nvidia-vipe || \
    echo "!! 'pip install nvidia-vipe' failed — build from third_party/vipe per its envs/cu128.yml"
else
  echo "[skip] vipe env exists"
fi

echo "==== 3/3  Dyn-HaMR env (cu117) ===="
DH="$ROOT/third_party/Dyn-HaMR"
if ! $CONDA env list | grep -q '^dynhamr '; then
  if [ -f "$DH/install_conda.sh" ]; then
    ( cd "$DH" && bash install_conda.sh ) || echo "!! Dyn-HaMR install_conda.sh failed"
    ( cd "$DH" && bash install_pip.sh 2>/dev/null ) || true
  else
    echo "!! $DH/install_conda.sh not found"
  fi
else
  echo "[skip] dynhamr env exists"
fi

cat <<'NOTE'

==== weights / MANO (manual) ====
Dyn-HaMR needs its _DATA tree populated:
  cd third_party/Dyn-HaMR && bash prepare.sh        # HaMeR/ViTPose/HMP/DROID weights
  # then place the license-gated MANO_RIGHT.pkl at third_party/Dyn-HaMR/_DATA/data/mano/
VIPE auto-downloads its weights on first run.

Once both envs build + weights are in place, re-run the CHOIR coarse pipeline:
  python -m hoi_recon.cli --video examples/grab.mp4 --out runs/grab_choir --real \
      --config configs/choir.yaml --stages 0-5 --force
It will auto-detect the 'dynhamr' and 'vipe' envs and use them (the logs say which).
NOTE
