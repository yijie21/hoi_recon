"""Shared helpers for the geometric stages (4-8): expand object world geometry
and compute hand<->object contact-gap diagnostics."""
from __future__ import annotations

import numpy as np

from ..geometry import transform_points, vertex_normals, knn


def object_world(verts, faces, pose):
    """Canonical object verts -> world verts + world vertex normals for one frame."""
    wv = transform_points(verts, pose)
    wn = vertex_normals(wv, faces)
    return wv, wn


def contact_gap(hand_cand, obj_world_verts):
    """Min surface gap (m) between hand contact candidates and object surface."""
    d, _ = knn(hand_cand, obj_world_verts, k=1)
    return float(d.min())


def all_object_world(verts, faces, poses):
    T = poses.shape[0]
    wv = np.stack([transform_points(verts, poses[i]) for i in range(T)], 0)
    wn = np.stack([vertex_normals(wv[i], faces) for i in range(T)], 0)
    return wv, wn  # [T,No,3], [T,No,3]


def radial_penetration(points, obj_world_verts):
    """Robust penetration depth for a (near-)convex object: a point is inside if it
    is closer to the object centroid than the surface is in that direction. Returns
    (depth[Q] >= 0, dir[Q,3]) where dir = (point - centroid)/|.| (outward push).

    For a sphere this is exact; for convex meshes it is a good approximation and,
    unlike vertex-normal signed distance, it does not produce silhouette false
    positives for exterior points.
    """
    oc = obj_world_verts.mean(0)
    _, idx = knn(points, obj_world_verts, k=1)
    r_local = np.linalg.norm(obj_world_verts[idx[:, 0]] - oc, axis=1)
    r = points - oc
    rn = np.linalg.norm(r, axis=1)
    depth = np.clip(r_local - rn, 0, None)
    dirv = r / np.clip(rn[:, None], 1e-9, None)
    return depth, dirv


def correspondences(hand_cand, obj_world_verts, obj_world_normals,
                    dist_thresh, cos_thresh):
    """KNN contact correspondences for one frame, gated by distance and
    surface-normal compatibility. Returns (idx, dist, valid)."""
    d, idx = knn(hand_cand, obj_world_verts, k=1)
    d, idx = d[:, 0], idx[:, 0]
    anchor = obj_world_verts[idx]
    nrm = obj_world_normals[idx]
    dirv = hand_cand - anchor
    dn = np.linalg.norm(dirv, axis=1, keepdims=True)
    cosang = np.sum((dirv / np.clip(dn, 1e-9, None)) * nrm, axis=1)
    valid = (d < dist_thresh) & (cosang > cos_thresh)
    return idx, d, valid
