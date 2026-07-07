#!/bin/bash
# Batch: auto-masks + kill test for all manifest clips. Hardened for the shared
# GPU box: per-stage timeouts (no pipe-buffered hangs), dynamic GPU pick gated
# on free memory, full per-clip logs, skip-if-done resume.
set -uo pipefail
cd /workspace/code/hoi_recon/compare/hoi4d/kill_test
SCRATCH=/tmp/claude-1002/-workspace-code-hoi-recon/2580edc3-f5c5-491e-b66c-81be5f17a7a4/scratchpad

wait_gpu() {  # $1 = MB needed; pinned to GPU 0 (GPU 1 driver-wedged since 06:13), waits until free
  while true; do
    f=$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits |
        awk -F', *' '$1==0 {print $2-$3}')
    if [ "${f:-0}" -ge "$1" ]; then echo 0; return; fi
    echo "  (waiting for ${1}MB free on GPU0, have ${f}MB) $(date +%H:%M:%S)" >&2
    sleep 60
  done
}

for entry in $(python3 -c "
import json
m = json.load(open('/workspace/hoi4d/clips/clips_manifest.json'))
print(' '.join(f'{k}:{v[\"category\"].lower()}' for k, v in m.items() if k != 'kettle_N15'))"); do
  name="${entry%%:*}"; cat="${entry##*:}"
  clip=/workspace/hoi4d/clips/$name
  if [ -f "$clip/kill_test/result.json" ]; then echo "skip $name (already done)"; continue; fi

  g=$(wait_gpu 12000)
  echo "=== masks $name ($cat) GPU$g $(date +%H:%M:%S)"
  timeout -k 30 1800 env CUDA_VISIBLE_DEVICES=$g /workspace/miniconda3/envs/hort/bin/python \
      make_masks.py --clip "$clip" --category "$cat" > "$clip/masks_log.txt" 2>&1
  rc=$?
  tail -2 "$clip/masks_log.txt"
  if [ $rc -ne 0 ]; then echo "!!! MASKS FAILED rc=$rc for $name — skipping clip"; continue; fi

  /workspace/miniconda3/envs/daid/bin/python - "$clip" "$SCRATCH/ov_$name.png" << 'PYEOF'
import sys, numpy as np, cv2
clip, out = sys.argv[1], sys.argv[2]
tiles = []
for t in (20, 75, 130):
    img = cv2.imread(f'{clip}/rgb/{t:06d}.jpg')
    o = cv2.imread(f'{clip}/masks/frame_{t:06d}_masks/object.png', 0) > 127
    h = cv2.imread(f'{clip}/masks/frame_{t:06d}_masks/hand.png', 0) > 127
    img[o] = img[o] // 2 + np.array([128, 0, 0], np.uint8)
    img[h] = img[h] // 2 + np.array([0, 128, 0], np.uint8)
    tiles.append(cv2.resize(img, (512, 288)))
cv2.imwrite(out, np.hstack(tiles))
PYEOF

  g2=$(wait_gpu 8000)
  echo "=== killtest $name GPU$g2 $(date +%H:%M:%S)"
  timeout -k 30 1800 env CUDA_VISIBLE_DEVICES=$g2 /workspace/miniconda3/envs/daid/bin/python \
      kill_test.py --clip "$clip" --masks "$clip/masks/frame_{idx:06d}_masks/*.png" --half \
      > "$clip/kill_test_log.txt" 2>&1 || { echo "!!! KILLTEST FAILED for $name"; continue; }
  grep -m1 '"R_MAE"' "$clip/kill_test_log.txt" || true
done
echo ALL_DONE
