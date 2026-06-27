"""Stage 0 (§2.1): egocentric RGB-D observation stream + GT passthrough (mock)."""
from __future__ import annotations
from ..bundle import Bundle
from ..core.mock_scene import generate_ego_hoi

NAME = "stage0_ego_io"; INDEX = 0

def run(ctx) -> Bundle:
    cfg = ctx.cfg
    if not cfg.mock:
        raise NotImplementedError("real ego-io backend (RGB-D loader) — see backends/real.py")
    s = generate_ego_hoi(num_frames=int(cfg.num_frames), seed=int(cfg.seed))
    arrays = {
        "intrinsics": s.intrinsics, "cam_traj": s.cam_traj, "table_T_gt": s.table_T,
        "depth": s.depth, "obj_mask": s.obj_mask, "hand_mask": s.hand_mask,
        "gt_obj_poses_w": s.obj_poses_w, "gt_obj_verts": s.obj_verts, "gt_obj_faces": s.obj_faces,
        "gt_hand_verts_w": s.hand_verts_w, "gt_hand_joints_w": s.hand_joints_w,
    }
    meta = {"T": s.T, "fps": s.fps, "image_size": list(s.image_size),
            "stage_labels": list(map(str, s.stage_labels)),
            "finger_idx": {k: v.tolist() for k, v in s.finger_idx.items()}}
    return Bundle(arrays=arrays, meta=meta)
