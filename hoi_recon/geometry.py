"""Geometry primitives used by the real (non-learned) stages: SE3, meshes,
KNN, Umeyama alignment, vertex normals, signed-distance / penetration.

Pure numpy so the alignment / contact / optimization stages run without torch.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


# --------------------------------------------------------------------------
# SE3 / rotations
# --------------------------------------------------------------------------
def rotvec_to_R(rotvec: np.ndarray) -> np.ndarray:
    """Rodrigues: axis-angle (3,) -> rotation matrix (3,3)."""
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-8:
        return np.eye(3)
    k = rotvec / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def transform_points(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform to (N,3) points."""
    return pts @ T[:3, :3].T + T[:3, 3]


# --------------------------------------------------------------------------
# Meshes
# --------------------------------------------------------------------------
def uv_sphere(radius: float = 1.0, nlat: int = 16, nlon: int = 24
              ) -> Tuple[np.ndarray, np.ndarray]:
    """A simple UV sphere mesh -> (verts[N,3], faces[M,3])."""
    verts = []
    for i in range(nlat + 1):
        theta = np.pi * i / nlat
        for j in range(nlon):
            phi = 2 * np.pi * j / nlon
            verts.append([
                radius * np.sin(theta) * np.cos(phi),
                radius * np.cos(theta),
                radius * np.sin(theta) * np.sin(phi),
            ])
    verts = np.asarray(verts, dtype=np.float64)
    faces = []
    for i in range(nlat):
        for j in range(nlon):
            a = i * nlon + j
            b = i * nlon + (j + 1) % nlon
            c = (i + 1) * nlon + j
            d = (i + 1) * nlon + (j + 1) % nlon
            faces.append([a, c, b])
            faces.append([b, c, d])
    return verts, np.asarray(faces, dtype=np.int64)


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals -> (N,3) unit vectors."""
    n = np.zeros_like(verts)
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)  # area-weighted face normals
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    return n / norm


# --------------------------------------------------------------------------
# Nearest neighbours
# --------------------------------------------------------------------------
def knn(query: np.ndarray, ref: np.ndarray, k: int = 1
        ) -> Tuple[np.ndarray, np.ndarray]:
    """Brute-force KNN. Returns (dist[Q,k], idx[Q,k]). Uses cKDTree if available."""
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(ref)
        d, i = tree.query(query, k=k)
        if k == 1:
            d, i = d[:, None], i[:, None]
        return d, i
    except Exception:
        d2 = ((query[:, None, :] - ref[None, :, :]) ** 2).sum(-1)
        idx = np.argsort(d2, axis=1)[:, :k]
        dist = np.sqrt(np.take_along_axis(d2, idx, axis=1))
        return dist, idx


# --------------------------------------------------------------------------
# Signed distance to a mesh (vertex-normal approximation)
# --------------------------------------------------------------------------
def signed_distance_to_mesh(points: np.ndarray, verts: np.ndarray,
                            normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Approx signed distance of each query point to the surface using the
    nearest vertex and its normal. Negative = inside (penetration).

    Returns (signed_dist[Q], nearest_idx[Q]).
    """
    dist, idx = knn(points, verts, k=1)
    idx = idx[:, 0]
    nearest = verts[idx]
    nrm = normals[idx]
    sign = np.sign(np.sum((points - nearest) * nrm, axis=1))
    sign[sign == 0] = 1.0
    return sign * dist[:, 0], idx


# --------------------------------------------------------------------------
# Umeyama similarity alignment (with optional scale)
# --------------------------------------------------------------------------
def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True
            ) -> Tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity mapping src -> dst. Returns (s, R, t) with
    dst ≈ s * R @ src + t. src,dst are (N,3)."""
    assert src.shape == dst.shape and src.shape[1] == 3
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Sc, Dc = src - mu_s, dst - mu_d
    cov = (Dc.T @ Sc) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (Sc ** 2).sum() / n
    s = float((D * np.diag(S)).sum() / var_s) if with_scale and var_s > 1e-12 else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def mesh_volume(verts: np.ndarray, faces: np.ndarray) -> float:
    """Signed volume of a closed triangle mesh via the divergence theorem."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    return float(np.abs(np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2))) / 6.0))
