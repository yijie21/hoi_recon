"""Stage 6 — Generative spatial rectification + contact correspondences.

In:  coarse 4D HOI (stage5) + object geometry.
Out: rectified object placement + per-frame contact correspondences
     (hand candidate vert -> object surface anchor), validity-gated by
     distance (<2cm) and surface-normal compatibility (<60deg cone).
Model (real): flow-matching grasp prior trained on GraspPair (~500k DexGraspNet
     grasps), predicting ray-depth corrections to the relative placement.
Mock/fallback: a heuristic snap-to-contact on interaction frames — moves the
object so its surface meets the hand contact centroid, then interpolates the
correction across the clip. This is the cheap stand-in for the learned rectifier.
"""
from __future__ import annotations

import numpy as np

from ..bundle import Bundle
from ..geometry import se3
from ..logging_utils import log
from ._scene import object_world

NAME = "stage6_rectify"
INDEX = 6


def _interp_nan(delta):
    """Linear-interpolate per-axis over frames where delta is NaN."""
    T = delta.shape[0]
    x = np.arange(T)
    out = delta.copy()
    for a in range(3):
        col = delta[:, a]
        ok = ~np.isnan(col)
        if ok.sum() == 0:
            out[:, a] = 0.0
        else:
            out[:, a] = np.interp(x, x[ok], col[ok])
    return out


def _rectify(stage, obj_radius, loose_thresh=0.06, alpha=0.9):
    """Heuristic flow-matching surrogate: snap object to hand contact centroid on
    interaction frames; returns per-frame translation correction [T,3]."""
    hand_verts = stage["hand_verts"]
    contact_idx = stage["contact_idx"].astype(int)
    obj_verts, obj_faces = stage["obj_verts"], stage["obj_faces"].astype(int)
    poses = stage["obj_poses"]
    T = poses.shape[0]
    r = float(obj_radius) if obj_radius and float(obj_radius) > 0 else \
        float(np.linalg.norm(obj_verts - obj_verts.mean(0), axis=1).mean())

    delta = np.full((T, 3), np.nan)
    for i in range(T):
        hc = hand_verts[i, contact_idx].mean(0)
        oc = poses[i, :3, 3]
        gap = np.linalg.norm(hc - oc) - r
        if gap < loose_thresh:                       # likely interacting
            dir_ = (oc - hc)
            n = np.linalg.norm(dir_)
            dir_ = dir_ / n if n > 1e-9 else np.array([0, 0, 1.0])
            desired_oc = hc + r * dir_               # surface just touches hc
            delta[i] = alpha * (desired_oc - oc)
    return _interp_nan(delta)


def run(ctx) -> Bundle:
    cfg = ctx.cfg
    s5 = ctx.load("stage5_coarse_fit")
    dist_thresh = float(cfg.contact.dist_thresh_m)
    cos_thresh = float(np.cos(np.deg2rad(cfg.contact.normal_thresh_deg)))

    contact_idx = s5["contact_idx"].astype(int)
    obj_verts, obj_faces = s5["obj_verts"], s5["obj_faces"].astype(int)
    radius = s5.get("obj_radius", np.array(0.0))

    # 1) rectify object placement
    delta = _rectify(s5, radius)
    poses = s5["obj_poses"].copy()
    poses[:, :3, 3] += delta

    # 2) build per-frame contact correspondences on rectified geometry
    T, Nc = poses.shape[0], len(contact_idx)
    corr_obj_idx = -np.ones((T, Nc), dtype=np.int64)
    corr_dist = np.full((T, Nc), np.inf)
    corr_valid = np.zeros((T, Nc), bool)
    for i in range(T):
        ow, on = object_world(obj_verts, obj_faces, poses[i])
        hc = s5["hand_verts"][i, contact_idx]
        from ..geometry import knn
        d, idx = knn(hc, ow, k=1)
        d, idx = d[:, 0], idx[:, 0]
        anchor = ow[idx]
        nrm = on[idx]
        dirv = hc - anchor
        dn = np.linalg.norm(dirv, axis=1, keepdims=True)
        cosang = np.sum((dirv / np.clip(dn, 1e-9, None)) * nrm, axis=1)
        valid = (d < dist_thresh) & (cosang > cos_thresh)
        corr_obj_idx[i] = idx
        corr_dist[i] = d
        corr_valid[i] = valid

    # Predicted contact MAP (for output/eval) is proximity-based — parallels the
    # physical GT (closeness to surface). The normal gate above is used only to
    # build stable optimization correspondences, not to predict contact.
    contact_map = corr_dist < dist_thresh

    n_active = int(corr_valid.sum())
    log(f"rectified: applied delta max={np.abs(delta).max()*100:.1f}cm; "
        f"contact correspondences active={n_active} over {T} frames "
        f"(gate: <{dist_thresh*100:.0f}cm, <{cfg.contact.normal_thresh_deg:.0f}deg)")

    return Bundle(
        arrays={"hand_verts": s5["hand_verts"], "hand_joints": s5["hand_joints"],
                "contact_idx": contact_idx, "obj_verts": obj_verts,
                "obj_faces": obj_faces, "obj_poses": poses, "obj_radius": radius,
                "corr_obj_idx": corr_obj_idx, "corr_dist": corr_dist,
                "corr_valid": corr_valid, "contact_map": contact_map,
                "rectify_delta": delta},
        meta={"n_active_contacts": n_active,
              "dist_thresh_m": dist_thresh,
              "normal_thresh_deg": float(cfg.contact.normal_thresh_deg)})
