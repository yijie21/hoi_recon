#!/usr/bin/env bash
# Score all three FP pose modes (register_each / track / fuse) for a clip from the
# cached _fp_all_poses.npz — instant (no FP rerun). Prints one score line per mode.
#   score_fp_modes.sh <rc_input> <fp_out_dir> <icpjgr_run>
set -uo pipefail
RC=$1; OUT=$2; RUN=$3
FPY=/workspace/miniconda3/envs/forehoi5090/bin/python
SPY=/workspace/miniconda3/envs/rc5090/bin/python
cd /workspace/code/hoi_recon/compare/hot3d
for MODE in register_each track fuse; do
  # rewrite pseudo_gt.npz for this mode (cache-instant, no FP)
  $FPY run_fp_hot3d.py "$RC" "$RUN" "$OUT" --mode "$MODE" >/dev/null 2>&1
  printf "  %-14s " "$MODE:"
  $SPY gt_pose_eval_hot3d.py "$RC" "$OUT" 2>/dev/null | sed 's/^[^:]*: //'
done
