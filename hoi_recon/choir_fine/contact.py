# hoi_recon/choir_fine/contact.py
"""Barycentric contact correspondence (CHOIR §7.2). For each hand contact point, find up
to top-K object-surface anchors that pass a distance gate (<dist_thresh) and a
surface-normal compatibility gate (hand->anchor direction within normal_deg of the object
normal). Anchors are sampled surface points stored as (face_id, barycentric, softmax
weight) so they move rigidly with the object pose during optimization."""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def build_correspondences(hand_pts, mesh, *, n_surface=10000, knn=50, topk=8,
                          dist_thresh=0.02, normal_deg=60.0, softmax_sigma=0.01, seed=0):
    """hand_pts: (Nh,3). mesh: trimesh.Trimesh (object, current frame, world coords).
    Returns dict of arrays indexed [hand_vertex, k]:
      face_id (Nh,topk) int (-1 where invalid), bary (Nh,topk,3), anchor (Nh,topk,3),
      weight (Nh,topk) softmax over kept anchors (0 where invalid), valid (Nh,) bool.
    Anchor weights are softmax(-d^2 / softmax_sigma) over the kept anchors, so
    softmax_sigma is a bandwidth (variance-scale), NOT a standard deviation; CHOIR uses 0.01."""
    hand_pts = np.asarray(hand_pts, float)
    Nh = len(hand_pts)
    pts, face_idx = trimesh.sample.sample_surface(mesh, n_surface, seed=seed)
    fnormals = np.asarray(mesh.face_normals)[face_idx]            # (n_surface,3)
    tree = cKDTree(pts)
    cos_thresh = np.cos(np.deg2rad(normal_deg))

    face_id = np.full((Nh, topk), -1, np.int64)
    bary = np.zeros((Nh, topk, 3), float)
    anchor = np.zeros((Nh, topk, 3), float)
    weight = np.zeros((Nh, topk), float)
    valid = np.zeros(Nh, bool)

    k_query = min(knn, n_surface)
    dists, nn = tree.query(hand_pts, k=k_query)
    if k_query == 1:
        dists, nn = dists[:, None], nn[:, None]

    for i in range(Nh):
        cand = nn[i]
        d = dists[i]
        a = pts[cand]                                            # candidate anchors
        # direction hand->anchor must oppose the outward normal (hand outside, anchor
        # on the contact-facing side): (anchor-hand) . normal < 0  AND aligned within cone
        dirv = a - hand_pts[i]
        dn = np.linalg.norm(dirv, axis=1) + 1e-12
        cosang = -(dirv / dn[:, None] * fnormals[cand]).sum(1)   # +1 when facing the hand
        ok = (d < dist_thresh) & (cosang > cos_thresh)
        if not ok.any():
            continue
        sel = np.where(ok)[0][:topk]                             # nearest-first, top-K
        gi = cand[sel]
        face_id[i, :len(sel)] = face_idx[gi]
        anchor[i, :len(sel)] = pts[gi]
        bary[i, :len(sel)] = trimesh.triangles.points_to_barycentric(
            mesh.triangles[face_idx[gi]], pts[gi])
        w = np.exp(-(d[sel] ** 2) / softmax_sigma)
        weight[i, :len(sel)] = w / w.sum()
        valid[i] = True

    return {"face_id": face_id, "bary": bary, "anchor": anchor,
            "weight": weight, "valid": valid}
