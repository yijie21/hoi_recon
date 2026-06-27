"""Stage 5 (§2.1.4): ego-motion compensation. Transform all states into a fixed
table frame (SLAM in real mode; known camera + plane-fit table in mock), then
light temporal smoothing. No table/vertical constraint on the object."""
from __future__ import annotations
import numpy as np
from ..bundle import Bundle
from ..core.geometry import se3_inv, transform_points

NAME = "stage5_ego_comp"; INDEX = 5


def _smooth(x, w):
    if w <= 1:
        return x
    k = np.ones(w) / w; pad = w // 2
    xp = np.concatenate([x[pad:0:-1], x, x[-2:-pad - 2:-1]], 0)[:x.shape[0] + 2 * pad]
    flat = xp.reshape(xp.shape[0], -1); out = np.empty((x.shape[0], flat.shape[1]))
    for c in range(flat.shape[1]):
        out[:, c] = np.convolve(flat[:, c], k, "valid")[:x.shape[0]]
    return out.reshape(x.shape)


def run(ctx) -> Bundle:
    cfg = ctx.cfg
    s0 = ctx.load("stage0_ego_io"); s2 = ctx.load("stage2_track"); s4 = ctx.load("stage4_hand")
    if not cfg.mock:
        raise NotImplementedError("real ORB-SLAM3 backend — backends/real.py")
    table_T = s0["table_T_gt"]                       # real: estimate from SLAM + plane fit
    w2t = se3_inv(table_T)
    T = s4["hand_verts_w"].shape[0]
    hv = np.stack([transform_points(s4["hand_verts_w"][i], w2t) for i in range(T)], 0)
    hj = np.stack([transform_points(s4["hand_joints_w"][i], w2t) for i in range(T)], 0)
    op = np.stack([w2t @ s2["obj_poses_w"][i] for i in range(T)], 0)
    win = int(cfg.ego.smooth_window)
    hj = _smooth(hj, win)
    op[:, :3, 3] = _smooth(op[:, :3, 3], win)          # smooth object translation only
    hv = _smooth(hv, win)
    return Bundle(arrays={"hand_verts_t": hv, "hand_joints_t": hj, "obj_poses_t": op,
                          "obj_verts": s2["obj_verts"], "obj_faces": s2["obj_faces"]},
                  meta={"finger_idx": s0.meta["finger_idx"],
                        "stage_labels": s0.meta["stage_labels"]})
