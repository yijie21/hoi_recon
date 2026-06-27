"""Stage 4 (§2.1.3): MANO hand (HaWoR in real mode) + RGB-D global translation
correction. Mock injects a global depth bias on the GT hand and removes it via
robust residuals between predicted hand surface and observed depth."""
from __future__ import annotations
import numpy as np
from ..bundle import Bundle
from ..core.geometry import se3_inv, transform_points

NAME = "stage4_hand"; INDEX = 4


def _depth_residual_correction(verts_w, cam2world, depth, K, nbr):
    """Estimate a global translation that best aligns predicted hand depth to
    observed depth (robust median of per-vertex depth residual along +z cam)."""
    w2c = se3_inv(cam2world)
    vc = transform_points(verts_w, w2c)
    z = np.clip(vc[:, 2], 1e-6, None)
    u = np.round(vc[:, 0] / z * K[0, 0] + K[0, 2]).astype(int)
    v = np.round(vc[:, 1] / z * K[1, 1] + K[1, 2]).astype(int)
    H, W = depth.shape
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    obs = np.zeros(len(z)); obs[ok] = depth[v[ok], u[ok]]
    valid = ok & (obs > 0)
    if valid.sum() < 10:
        return np.zeros(3)
    dz = np.median(obs[valid] - z[valid])               # robust depth residual
    # back to world: translation along camera +z
    return cam2world[:3, :3] @ np.array([0.0, 0.0, dz])


def run(ctx) -> Bundle:
    cfg = ctx.cfg
    s0 = ctx.load("stage0_ego_io")
    if not cfg.mock:
        raise NotImplementedError("real HaWoR hand backend — backends/real.py")
    hv = s0["gt_hand_verts_w"].copy(); hj = s0["gt_hand_joints_w"].copy()
    cam = s0["cam_traj"]; depth = s0["depth"]; K = s0["intrinsics"]; T = hv.shape[0]
    gt_root = hj[:, 0].copy()
    bias = cfg.hand.depth_bias_m
    # inject global depth bias (along each frame's camera +z, in world)
    for i in range(T):
        b = cam[i, :3, :3] @ np.array([0.0, 0.0, bias])
        hv[i] += b; hj[i] += b
    before = float(np.mean(np.linalg.norm(hj[:, 0] - gt_root, axis=1)) * 1000)
    for i in range(T):
        dp = _depth_residual_correction(hv[i], cam[i], depth[i], K, cfg.hand.corr_neighborhood_px)
        hv[i] += dp; hj[i] += dp
    after = float(np.mean(np.linalg.norm(hj[:, 0] - gt_root, axis=1)) * 1000)
    return Bundle(arrays={"hand_verts_w": hv, "hand_joints_w": hj},
                  meta={"transl_err_before_mm": before, "transl_err_after_mm": after})
