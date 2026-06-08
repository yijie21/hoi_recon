"""Stage 7 — Contact-aware joint optimization (final 4D HOI).

In:  rectified frames + contact correspondences (stage6) + coarse evidence.
Out: refined object 6D trajectory + per-frame contact maps; final 4D HOI.
Method: optimize a per-frame object translation field d[T,3] (the object is the
freely-moving body; the hand is held fixed from stage5 here — extend to joint
hand+object as needed) under CHOIR-style energies:
    L_contact   pull active hand verts to their object-surface anchors
    L_pen       one-sided non-penetration
    L_temporal  smooth object motion
    L_anchor    stay near the stage-6 prior
A *soft contact cache* (anchors + penetration correspondences) is rebuilt every
CACHE_PERIOD iterations from the current geometry — mirroring CHOIR — which keeps
the pull well-posed as the object moves. Energies are normalized by their active
count so the step size is independent of mesh resolution. Optimizer: numpy Adam
with analytic gradients (torch autograd path left as a TODO).
"""
from __future__ import annotations

import numpy as np

from ..bundle import Bundle
from ..geometry import knn
from ..logging_utils import log
from ._scene import all_object_world, correspondences, radial_penetration

NAME = "stage7_contact_optim"
INDEX = 7
CACHE_PERIOD = 10


def _build_cache(hand_c, hand_pen, ow, on, d, dist_thresh, cos_thresh):
    """Rebuild the soft contact cache at current geometry (object translated by d):
      * contact anchors (active hand vert -> object surface point), and
      * per-frame object centroid + local surface radius for radial penetration.
    Normals are translation-invariant -> reuse `on`."""
    T = ow.shape[0]
    con_a0, con_h = [], []          # per-frame anchor(base) + hand point (active)
    pen_oc0, pen_rloc = [], []      # object centroid (base) + local radius per hand vert
    n_active = 0
    for i in range(T):
        owc = ow[i] + d[i]
        idx, _, valid = correspondences(hand_c[i], owc, on[i], dist_thresh, cos_thresh)
        con_a0.append(ow[i][idx[valid]])        # base anchor (d added analytically)
        con_h.append(hand_c[i][valid])
        n_active += int(valid.sum())
        oc0 = ow[i].mean(0)                      # base object centroid
        nidx = knn(hand_pen[i], owc, k=1)[1][:, 0]
        pen_oc0.append(oc0)
        pen_rloc.append(np.linalg.norm(ow[i][nidx] - oc0, axis=1))
    return con_a0, con_h, pen_oc0, pen_rloc, max(n_active, 1)


def _object_only_optim(hand_verts, hand_c, obj_verts, obj_faces, poses0, o,
                       dist_thresh, cos_thresh):
    """Object-translation-only contact optimization (numpy Adam, analytic grads).
    Pulls active hand verts onto object-surface anchors with one-sided radial
    non-penetration; the hand is held fixed. Returns the per-frame object
    translation delta d[T,3]. Used in mock mode (real mode uses joint_optimize)."""
    T, Nh = poses0.shape[0], hand_verts.shape[1]
    ow, on = all_object_world(obj_verts, obj_faces, poses0)
    norm_pen = float(T * Nh)
    d = np.zeros((T, 3))
    m = np.zeros_like(d); v = np.zeros_like(d)
    b1, b2, eps = 0.9, 0.999, 1e-8
    cache = None
    for it in range(int(o.iters)):
        if it % CACHE_PERIOD == 0:
            cache = _build_cache(hand_c, hand_verts, ow, on, d, dist_thresh, cos_thresh)
        con_a0, con_h, pen_oc0, pen_rloc, n_active = cache
        g = np.zeros_like(d)
        for i in range(T):
            if con_h[i].shape[0]:
                res = con_a0[i] + d[i] - con_h[i]
                g[i] += o.w_contact * 2.0 * res.sum(0) / n_active
        for i in range(T):
            r = hand_verts[i] - (pen_oc0[i] + d[i])
            rn = np.linalg.norm(r, axis=1)
            depth = np.clip(pen_rloc[i] - rn, 0, None)
            mask = depth > 0
            if mask.any():
                dirv = r[mask] / np.clip(rn[mask, None], 1e-9, None)
                g[i] += o.w_pen * 2.0 * (depth[mask, None] * dirv).sum(0) / norm_pen
        diff = d[1:] - d[:-1]
        g[1:] += o.w_temporal * 2.0 * diff
        g[:-1] -= o.w_temporal * 2.0 * diff
        g += o.w_anchor * 2.0 * d
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g * g)
        mh = m / (1 - b1 ** (it + 1))
        vh = v / (1 - b2 ** (it + 1))
        d -= o.lr * mh / (np.sqrt(vh) + eps)
    return d


def run(ctx) -> Bundle:
    cfg = ctx.cfg
    s6 = ctx.load("stage6_rectify")
    o = cfg.optim
    dist_thresh = float(cfg.contact.dist_thresh_m)
    cos_thresh = float(np.cos(np.deg2rad(cfg.contact.normal_thresh_deg)))

    hand_verts = s6["hand_verts"]
    contact_idx = s6["contact_idx"].astype(int)
    hand_c = hand_verts[:, contact_idx]
    obj_verts, obj_faces = s6["obj_verts"], s6["obj_faces"].astype(int)
    poses0 = s6["obj_poses"].copy()
    T, Nh = poses0.shape[0], hand_verts.shape[1]

    differentiable = (o.get("differentiable", False) if hasattr(o, "get") else False)
    if not cfg.mock and differentiable and s6.get("obj_colors") is not None:
        # real + differentiable: full joint MANO-articulation + object render-and-
        # compare optimizer (PyTorch3D, sam3d env). Fingers actually curl to grasp.
        import os as _os
        from ..backends.real_perception import run_joint_optimizer, list_frames
        s0 = ctx.load("stage0_preprocess"); s1 = ctx.load("stage1_detect_track"); s2 = ctx.load("stage2_hand")
        frame_paths = list_frames(s0.assets["frames_dir"])
        mdir = s1.assets["masks_dir"]
        mask_paths = [(_os.path.join(mdir, f"{i:05d}.npy")
                       if _os.path.exists(_os.path.join(mdir, f"{i:05d}.npy")) else None)
                      for i in range(T)]
        hand_verts, hj, poses = run_joint_optimizer(cfg, ctx.stage_dir(NAME), s2, s6,
                                                    frame_paths, mask_paths, s0["intrinsics"])
        # optimized MANO joints (consistent with the moved hand); fall back to the
        # stage-6 joints only for caches predating the hand_joints output
        hand_joints = hj if hj is not None else s6["hand_joints"]
        hand_c = hand_verts[:, contact_idx]
        d = poses[:, :3, 3] - poses0[:, :3, 3]
        log("joint optimizer (differentiable MANO+object) applied")
    elif not cfg.mock:
        # real: rigid JOINT hand+object grasp optimization (numpy; no articulation).
        from ..joint_grasp import joint_optimize
        hand_verts, hand_joints, poses, stats = joint_optimize(
            hand_verts, s6["hand_joints"], contact_idx, obj_verts, obj_faces,
            poses0, float(s6.get("obj_radius", np.array(0.04))), o, iters=300)
        hand_c = hand_verts[:, contact_idx]
        d = poses[:, :3, 3] - poses0[:, :3, 3]
        log(f"joint grasp: hand verts within 1cm of object = "
            f"{stats['env_within_1cm']:.0f}/{Nh} over {stats['grasp_frames']} grasp frames")
    else:
        # mock: object-translation-only contact optimization (numpy Adam)
        hand_joints = s6["hand_joints"]
        d = _object_only_optim(hand_verts, hand_c, obj_verts, obj_faces, poses0,
                               o, dist_thresh, cos_thresh)
        poses = poses0.copy()
        poses[:, :3, 3] += d

    # finalize
    owf, onf = all_object_world(obj_verts, obj_faces, poses)
    contact_map = np.zeros((T, len(contact_idx)), bool)
    gaps = np.zeros(T)
    pen_depth = 0.0
    for i in range(T):
        idx, dd, valid = correspondences(hand_c[i], owf[i], onf[i],
                                         dist_thresh, cos_thresh)
        contact_map[i] = dd < dist_thresh        # proximity-based predicted contact
        gaps[i] = float(dd.min())
        depth, _ = radial_penetration(hand_verts[i], owf[i])
        pen_depth += float(depth.sum())

    log(f"final: active contacts={int(contact_map.sum())}, "
        f"gap median={np.median(gaps)*1000:.1f}mm, "
        f"object moved median={np.median(np.linalg.norm(d,axis=1))*100:.1f}cm")

    arrays = {"hand_verts": hand_verts, "hand_joints": hand_joints,
              "contact_idx": contact_idx, "obj_verts": obj_verts,
              "obj_faces": obj_faces, "obj_poses": poses,
              "obj_radius": s6.get("obj_radius", np.array(0.0)),
              "object_delta": d, "contact_map": contact_map, "gaps": gaps}
    if s6.get("obj_colors") is not None:
        arrays["obj_colors"] = s6["obj_colors"]
    if s6.get("hand_faces") is not None:
        arrays["hand_faces"] = s6["hand_faces"]
    return Bundle(
        arrays=arrays,
        meta={"n_active_contacts": int(contact_map.sum()),
              "gap_median_mm": float(np.median(gaps) * 1000),
              "penetration_depth_sum": pen_depth})
