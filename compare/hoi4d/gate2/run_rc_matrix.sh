#!/bin/bash
# RC substrate matrix: 12 HOI4D clips x {moge, vggt, gt} depth conditions.
# Sequential on one GPU (default 0). Skip-if-done via stage8 report. Usage:
#   bash run_rc_matrix.sh [clip_filter] [cond_filter]
set -u
GATE2=/workspace/code/hoi_recon/compare/hoi4d/gate2
RC=/workspace/code/hoi_recon/render_and_compare
PY=/workspace/miniconda3/envs/rc5090/bin/python
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export HF_HOME=/workspace/huggingface_cache/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$RC"
mkdir -p runs/hoi4d_matrix

CLIP_FILTER="${1:-}"
COND_FILTER="${2:-}"

python3 - "$GATE2/rc_inputs.json" <<'PYEOF' > /tmp/rc_matrix_list.txt
import json, sys
for name, d in json.load(open(sys.argv[1])).items():
    print(name, d["video"], d["obj_prompt"][0], d["obj_prompt"][1])
PYEOF

[ "${REVERSE:-0}" = "1" ] && tac /tmp/rc_matrix_list.txt > /tmp/rc_matrix_list_r.txt \
  && mv /tmp/rc_matrix_list_r.txt /tmp/rc_matrix_list.txt

while read -r name video px py; do
  [ -n "$CLIP_FILTER" ] && [[ "$name" != *"$CLIP_FILTER"* ]] && continue
  for cond in gt vggt moge; do
    [ -n "$COND_FILTER" ] && [[ "$cond" != "$COND_FILTER" ]] && continue
    out="runs/hoi4d_matrix/${name}__${cond}"
    if [ -f "$out/stage8_eval/report.json" ]; then echo "skip $name $cond"; continue; fi
    mkdir "$out.lock" 2>/dev/null || { echo "locked $name $cond"; continue; }
    clip="/workspace/hoi4d/clips/$name"
    args=(--video "$video" --out "$out" --real --config configs/real_forehoi.yaml
          --object-prompt "$px" "$py")
    unset RC_GT_DEPTH_DIR RC_GT_INTRINSICS
    # validated kill-test masks for ALL conditions (identical segmentation
    # across substrates; avoids the SAM2 part-whole prompt lottery)
    export RC_OBJECT_MASK_PATTERN="$clip/masks/frame_{idx:06d}_masks/object.png"
    export RC_OBJECT_MASK_ERODE=5
    case "$cond" in
      moge) args+=(--depth moge);;
      gt)   args+=(--depth gt); export RC_GT_DEPTH_DIR="$clip/depth" RC_GT_INTRINSICS="$clip/intrin.npy";;
      vggt) args+=(--depth gt); export RC_GT_DEPTH_DIR="$clip/gate2/vggt_depth_mm" RC_GT_INTRINSICS="$clip/intrin.npy";;
    esac
    echo "=== $name $cond $(date +%H:%M:%S)"
    timeout -k 30 2400 "$PY" -m hoi_recon.cli "${args[@]}" > "$out.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then echo "!!! FAIL $name $cond rc=$rc"; tail -3 "$out.log"; fi
  done
done < /tmp/rc_matrix_list.txt
echo MATRIX_DONE
