# hoi_recon/choir_fine/terms_torch.py
"""Differentiable CHOIR Stage-3 energy terms (arXiv:2605.20992 §4.3). Each is a pure torch
function over a per-frame optimization state, testable on tiny CPU tensors. The optimizer
(follow-on plan) computes each term and sums them through registry.assemble_energy."""
from __future__ import annotations

import torch


def contact_loss(hand_c, anchors, weights, confidence) -> torch.Tensor:
    """CHOIR Eq 6 soft barycentric contact loss.
      hand_c:     (T,Nc,3) hand contact vertices
      anchors:    (T,Nc,K,3) object-surface anchor points (top-K per vertex)
      weights:    (T,Nc,K) softmax weights over anchors (sum to 1 per vertex)
      confidence: (T,Nc) per-vertex contact confidence/gate
    Returns scalar = sum_{t,i} c * sum_k w * ||v-a||^2  /  sum_{t,i} c."""
    diff = hand_c.unsqueeze(-2) - anchors              # (T,Nc,K,3)
    sq = (diff ** 2).sum(-1)                            # (T,Nc,K)
    per_vert = (weights * sq).sum(-1)                   # (T,Nc)
    return (confidence * per_vert).sum() / confidence.sum().clamp(min=1e-8)


def penetration_loss(hand_verts, nearest_surface, surface_normal, eps=0.005, clip=0.04) -> torch.Tensor:
    """CHOIR Eq 23 one-sided non-penetration. Penalizes hand vertices inside the object
    beyond a tolerance eps, per-vertex residual clamped to `clip`. Normals are normalized
    internally so non-unit normals do not scale the depth.
      hand_verts/nearest_surface/surface_normal: (T,Nh,3)."""
    n = surface_normal / surface_normal.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    signed = ((nearest_surface - hand_verts) * n).sum(-1)   # >0 when hand is inside
    return (signed - eps).clamp(min=0.0, max=clip).mean()


def velocity_loss(x) -> torch.Tensor:
    """Mean squared first temporal difference. x: (T, ...). Smoothness on any per-frame
    quantity (MANO pose, object rotation/translation). CHOIR L_temp velocity terms."""
    d = x[1:] - x[:-1]
    return d.pow(2).mean() if d.numel() > 0 else x.new_zeros(())


def acceleration_loss(x) -> torch.Tensor:
    """Mean squared second temporal difference. x: (T, ...). Kills residual jitter that
    velocity terms leave. CHOIR L_temp acceleration terms."""
    d = x[2:] - 2 * x[1:-1] + x[:-2]
    return d.pow(2).mean() if d.numel() > 0 else x.new_zeros(())


def keypoint_reproj_loss(joints_cam, kp2d, K, valid, sigma_px=60.0) -> torch.Tensor:
    """Geman-McClure-robust 2D keypoint reprojection (CHOIR L^h_2D). Projects camera-frame
    joints with K and compares to kp2d; the robust kernel r2/(r2+sigma^2) is bounded in
    [0,1) so outlier joints can't dominate.
      joints_cam: (T,J,3); kp2d: (T,J,2) full-image px; K: (3,3); valid: (T,) frame mask."""
    z = joints_cam[..., 2].clamp(min=1e-4)
    u = K[0, 0] * joints_cam[..., 0] / z + K[0, 2]
    v = K[1, 1] * joints_cam[..., 1] / z + K[1, 2]
    r2 = (u - kp2d[..., 0]) ** 2 + (v - kp2d[..., 1]) ** 2
    s2 = float(sigma_px) ** 2
    numer = (r2 / (r2 + s2)) * valid[:, None]              # (T,J), zero on invalid frames
    denom = valid.sum().clamp(min=1e-8) * joints_cam.shape[1]   # valid_frames * J
    return numer.sum() / denom
