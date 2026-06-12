# hoi_recon/choir_fine/terms_torch.py
"""Differentiable CHOIR Stage-3 energy terms (arXiv:2605.20992 §4.3). Each is a pure torch
function over a per-frame optimization state, testable on tiny CPU tensors. The optimizer
(follow-on plan) computes each term and sums them through registry.assemble_energy."""
from __future__ import annotations

import torch


def contact_loss(hand_c, anchors, weights, confidence):
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


def penetration_loss(hand_verts, nearest_surface, surface_normal, eps=0.005, clip=0.04):
    """CHOIR Eq 23 one-sided non-penetration. Penalizes hand vertices inside the object
    beyond a tolerance eps, per-vertex residual clamped to `clip`. Normals are normalized
    internally so non-unit normals do not scale the depth.
      hand_verts/nearest_surface/surface_normal: (T,Nh,3)."""
    n = surface_normal / surface_normal.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    signed = ((nearest_surface - hand_verts) * n).sum(-1)   # >0 when hand is inside
    return (signed - eps).clamp(min=0.0, max=clip).mean()
