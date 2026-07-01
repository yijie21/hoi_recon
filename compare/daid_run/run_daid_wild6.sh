#!/bin/bash
# Run do-as-i-do reconstruction on wild6 (single 'daid' env, all stages).
# Waits for a >=33GB VRAM window (shared box), then pins that GPU and runs.
set -eo pipefail
cd /workspace/code/hoi_recon/do-as-i-do/reconstruction
VIDEO="$(realpath wild6/wild6.mp4)"
REF=25; OBJECT="white bottle"; HAND=left
NEED=33000

echo "[run] waiting for >=${NEED}MiB free VRAM ..."
GPU=""
for attempt in $(seq 1 160); do
  read GPU FREE < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | tr ',' ' ')
  if [ "${FREE:-0}" -ge "$NEED" ]; then echo "[run] GPU$GPU has ${FREE}MiB free -> launching"; break; fi
  echo "[run] attempt $attempt: best GPU$GPU ${FREE}MiB free (<${NEED}); waiting 45s"; sleep 45; GPU=""
done
[ -z "$GPU" ] && { echo "[run] TIMED OUT waiting for VRAM"; exit 3; }

export ENV_SAM3=daid ENV_SAM3D=daid ENV_HAWOR=daid ENV_TAPNET=daid
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[run] $(date +%H:%M:%S) starting pipeline on GPU$GPU: video=$VIDEO ref=$REF obj='$OBJECT' hand=$HAND"
bash run_pipeline_headless.sh "$VIDEO" "$REF" "$OBJECT" "$HAND"
echo "[run] $(date +%H:%M:%S) DAID_PIPELINE_DONE"
echo "=== outputs ==="
find wild6 -maxdepth 2 -name "*.json" -o -name "*.obj" -o -name "*.npz" 2>/dev/null | head -30
