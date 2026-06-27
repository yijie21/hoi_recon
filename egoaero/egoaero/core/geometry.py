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


# --- additions for egoaero pose-graph / ego-motion ---
def se3_inv(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4); Ti[:3, :3] = R.T; Ti[:3, 3] = -R.T @ t
    return Ti


def R_to_rotvec(R: np.ndarray) -> np.ndarray:
    ang = np.arccos(np.clip((np.trace(R) - 1) / 2.0, -1.0, 1.0))
    if ang < 1e-8:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return ang * w / (2.0 * np.sin(ang))


def geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(R_to_rotvec(Ra.T @ Rb))))


def se3_log(T: np.ndarray) -> np.ndarray:
    """SE3 -> 6-vector twist [rho(3), phi(3)] (translation-part, rotation-part)."""
    phi = R_to_rotvec(T[:3, :3])
    return np.concatenate([T[:3, 3], phi])   # left-trivialized approx (small-step use)


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """6-vector twist -> SE3 (matching se3_log's convention)."""
    return se3(rotvec_to_R(xi[3:]), xi[:3])


def box_mesh(half: np.ndarray):
    """Axis-aligned box (half-extents (3,)) -> (verts[8,3], faces[12,3])."""
    s = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    verts = s * half
    faces = np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],
                      [2,3,7],[2,7,6],[1,5,7],[1,7,3],[0,2,6],[0,6,4]])
    return verts, faces
