#!/bin/bash
# Resume do-as-i-do from Stage 2.5 (TAPIR) — reuses the already-computed frames, SAM3 masks,
# SAM-3D bottle mesh, MoGe pointmaps, gravity.json, HaWoR hand meshes.
set -eo pipefail
HERE=/workspace/code/hoi_recon/do-as-i-do/reconstruction
cd "$HERE"; source "$HERE/config/paths.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"

n=25; OBJECT_ID=white_bottle; ANCHOR_HAND=left
VIDEO_PATH="$(realpath wild6/wild6.mp4)"; VIDEO_DIR="$(dirname "$VIDEO_PATH")"
VIDEO_MASKS_DIR="$VIDEO_DIR/video_segmentation/masks"
MASKS_DIR="$VIDEO_MASKS_DIR/frame_$(printf '%06d' "$n")_masks"
export ENV_SAM3=daid ENV_SAM3D=daid ENV_HAWOR=daid ENV_TAPNET=daid
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# VRAM guard
NEED=33000; GPU=""
for a in $(seq 1 160); do
  read GPU FREE < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | tr ',' ' ')
  [ "${FREE:-0}" -ge "$NEED" ] && { echo "[resume] GPU$GPU ${FREE}MiB free"; break; }
  echo "[resume] waiting VRAM (best GPU$GPU ${FREE}MiB)"; sleep 45; GPU=""
done
[ -z "$GPU" ] && { echo "TIMEOUT VRAM"; exit 3; }
export CUDA_VISIBLE_DEVICES=$GPU

echo "=== [2.5] TAPIR velocity tracking ==="
conda activate "$ENV_TAPNET"; cd "$SCRIPTS_DIR"
python tapir_velocity_tracking.py --video "$VIDEO_PATH" --mask-dir "$VIDEO_MASKS_DIR" \
    --object "$OBJECT_ID" --checkpoint "$TAPNET_CKPT"

echo "=== [3] guided pose prediction (track_object) ==="
conda activate "$ENV_SAM3D"; cd "$FASTSAM3D_DIR"
python track_object.py --config checkpoints/hf/pipeline.yaml --vid_dir "$VIDEO_DIR" \
    --masks_root "$VIDEO_MASKS_DIR" --object_name "$OBJECT_ID" --init_frame "$n" \
    --output_dir "$VIDEO_DIR/obj_tracking_out/$OBJECT_ID" --guidance_strength 1 --save_layout \
    --fix_scale_to_init_frame --pose_guidance_strength 0.5 --num_pose_samples 25 \
    --scoring_metric render_iou --pose_selection cluster --cluster_dist_thresh 0.3 \
    --cluster_min_size 3 --cluster_w_rot 1.5 --chain_poses --post_optimize --no-enable_shape_icp \
    --chain_on_diffusion --euler_steps 25 \
    --rotvel_json "$VIDEO_DIR/perframe_tracking_$OBJECT_ID/motion_stats.json"

echo "=== [3] project mesh ==="
cd "$SCRIPTS_DIR"
python run_project_mesh_combined.py --video "$VIDEO_PATH" \
    --mesh "$MASKS_DIR/$OBJECT_ID/${OBJECT_ID}.obj" \
    --json "$VIDEO_DIR/obj_tracking_out/$OBJECT_ID/combined_visualization/layout.json" \
    --output-base "$VIDEO_DIR/obj_tracking_out/$OBJECT_ID/combined_visualization/projected"

echo "=== [3] convert layout to camera frame ==="
python convert_layout_to_camera_frame.py \
    --input "$VIDEO_DIR/obj_tracking_out/$OBJECT_ID/combined_visualization/layout.json" \
    --output "$VIDEO_DIR/obj_tracking_out/$OBJECT_ID/combined_visualization/layout_camera_frame.json"

echo "=== [4] optimize translation/scale ==="
python optimize_translation_scale.py --video-dir "$VIDEO_DIR" \
    --layout-json "$VIDEO_DIR/obj_tracking_out/$OBJECT_ID/combined_visualization/layout_camera_frame.json" \
    --anchor-hand "$ANCHOR_HAND" --ref-frame "$n"

echo "=== DAID_RESUME_DONE ==="
find "$VIDEO_DIR/obj_tracking_out" -name "layout_camera_frame_optimized.json" 2>/dev/null
