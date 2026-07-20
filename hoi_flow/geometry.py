"""Pure-torch geometry utilities for the HOI flow refiner.

Everything here is batched, float32, and autograd-friendly (no in-place tricks that
break gradients) so it can live inside the loss and the sampling-time guidance term.
Rotations use the Zhou et al. (CVPR'19) *continuous* 6D representation — the first two
rows of the rotation matrix, re-orthonormalized by Gram-Schmidt — because a network
regressing raw SO(3) (quaternions/Euler) crosses that representation's discontinuities
and stalls. This module is also the package's small "math utils" home (sinusoidal
embedding), so model.py and conditioning.py can share it without an import cycle.
"""
import math

import torch
import torch.nn.functional as F


def rot6d_to_matrix(d6):
    """[...,6] -> [...,3,3]. Gram-Schmidt on the first two rows (pytorch3d convention).
    F.normalize is eps-guarded, so an all-zero 6D (a masked/invalid state entry) maps to
    a zero matrix instead of NaN — masked states never poison the batch."""
    a1, a2 = d6[..., 0:3], d6[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=1e-8)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)  # rows are b1,b2,b3


def matrix_to_rot6d(R):
    """[...,3,3] -> [...,6]: first two rows flattened. Exact inverse of rot6d_to_matrix
    on any orthonormal R."""
    return R[..., :2, :].reshape(*R.shape[:-2], 6)


def so3_geodesic(R1, R2):
    """Geodesic angle in radians, [...], between two rotations = |axis-angle of R1^T R2|."""
    Rrel = R1.transpose(-1, -2) @ R2
    tr = Rrel[..., 0, 0] + Rrel[..., 1, 1] + Rrel[..., 2, 2]
    cos = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.arccos(cos)


def decompose_along_ray(vecs, ray_dirs):
    """Split vecs [...,3] into components parallel / perpendicular to ray_dirs [...,3].
    Returns (along [...,1], perp [...,3]). Lets a loss weight depth error (along the
    camera ray, ill-constrained) separately from lateral error (across it)."""
    n = F.normalize(ray_dirs, dim=-1, eps=1e-8)
    along = (vecs * n).sum(-1, keepdim=True)
    perp = vecs - along * n
    return along, perp


def project_K(K, pts):
    """Pinhole projection. K [...,3,3] (broadcastable over pts' batch), pts [...,3] in the
    camera frame -> uv [...,2]."""
    uvw = torch.matmul(K, pts.unsqueeze(-1)).squeeze(-1)
    z = uvw[..., 2:3]
    z = torch.where(z.abs() < 1e-8, torch.full_like(z, 1e-8), z)
    return uvw[..., :2] / z


def sinusoidal_embedding(pos, dim, max_period=10000.0):
    """Transformer sinusoidal embedding, shared by frame-index (int) and flow-time
    (float t in [0,1]). pos [...] -> [..., dim]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=pos.device, dtype=torch.float32) / max(half, 1)
    )
    args = pos.float().unsqueeze(-1) * freqs
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:  # odd width: pad a zero column
        emb = torch.cat([emb, torch.zeros(*emb.shape[:-1], 1, device=pos.device)], dim=-1)
    return emb
