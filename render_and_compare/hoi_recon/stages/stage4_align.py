"""Stage 4 — Spatial alignment into one metric world frame.

In:  hand motion (stage2), object trajectory (stage3), depth+camera (stage0).
Out: hand & object expressed in ONE world frame; resolved global scale gauge.
Method: lift both via camera extrinsics; solve a single global similarity to the
metric depth (Umeyama). In mock (world==camera, no metric depth) this is a
structural compose + scale=1, and we record the residual hand<->object contact
gap — the misalignment that stages 6-7 exist to fix.
"""
from __future__ import annotations

import os

import numpy as np

from ..bundle import Bundle
from ..logging_utils import log
from ._scene import all_object_world, contact_gap

NAME = "stage4_align"
INDEX = 4


def run(ctx) -> Bundle:
    cfg = ctx.cfg
    s0 = ctx.load("stage0_preprocess")
    s2 = ctx.load("stage2_hand")
    s3 = ctx.load("stage3_object")
    T = int(s0.meta["T"])

    hand_verts = s2["verts"]
    hand_joints = s2["joints"]
    contact_idx = s2["contact_idx"].astype(int)
    obj_verts, obj_faces = s3["verts"], s3["faces"].astype(int)
    obj_poses = s3["poses"]

    # In real mode: transform hand (camera frame) and object into world via
    # extrinsics, then estimate one global metric scale from depth. In mock the
    # synthetic scene is already world-metric (extrinsics = I, scale = 1).
    world_scale = 1.0
    if not cfg.mock and s0.meta.get("has_depth"):
        # TODO(real): solve global similarity to metric depth (geometry.umeyama).
        log("real-mode metric scale solve not yet wired; using world_scale=1", "warn")

    icp_cfg = cfg.get("object_icp") if hasattr(cfg, "get") else None
    pose_core = (icp_cfg.get("pose_core", "icp")
                 if (icp_cfg and hasattr(icp_cfg, "get")) else "icp")
    icp_stats = None
    s_icp = 1.0
    learned_radius = None

    if pose_core == "learned":
        raise NotImplementedError(
            "the in-pipeline learned pose core lives on the dev branch; on main use "
            "compare/hot3d/run_fp_hot3d.py (FoundationPose) on top of a registration run")

    # CHOIR coarse: ray-scale alignment — slide the object along the camera ray so
    # its interaction depth matches the hand (preserves the silhouette fit).
    coarse = cfg.get("coarse") if hasattr(cfg, "get") else None
    if coarse == "choir" and not cfg.mock and \
            (cfg.choir.ray_scale.get("enable", True) if hasattr(cfg.choir.ray_scale, "get") else True):
        from ..choir import ray_scale_align
        obj_poses = ray_scale_align(obj_poses, obj_verts, hand_verts, contact_idx,
                                    s0["intrinsics"])
        log("CHOIR: ray-scale alignment applied (object slid along camera ray to "
            "match hand depth)")

    # Optional locked-scale rigid ICP: re-register the canonical object mesh
    # per frame onto the masked depth cloud (replaces the depth-lift-anchored
    # trajectory; scale stays global — see hoi_recon/object_icp.py).
    if pose_core != "learned" and icp_cfg and icp_cfg.get("enable", False) \
            and not cfg.mock and s0.meta.get("has_depth"):
        from ..object_icp import refine_object_poses
        from ..grasp_segments import grasp_mask_contact, grasp_spans
        s1 = ctx.load("stage1_detect_track")
        # T3 (v2): CONTACT-based stable-grasp mask + per-frame wrist orientation,
        # feeding _joint_refine's grasp-rigidity term (config key w_grasp,
        # default 0.0 = off). wrist_R = HaMeR's MANO global orientation
        # (mano_global, [T,3,3]) — used directly (its frame-to-frame deltas were
        # verified consistent with joint-derived deltas). Contact detection uses
        # the per-frame min distance between hand vertices and the posed stage-3
        # object surface — a RELATIVE hand-to-object quantity, so it is immune to
        # the absolute-trajectory ICP jitter that broke the old velocity
        # detector, and it includes in-hand rotation (no hand-speed gate).
        # Computed only when the term is actually on (w_grasp>0), so the flag-off
        # path stays byte-identical. Skipped if mano_global is absent.
        wrist_R = s2.get("mano_global")
        grasp_mask = None
        w_grasp = float(icp_cfg.get("w_grasp", 0.0)) \
            if hasattr(icp_cfg, "get") else 0.0
        if wrist_R is not None and w_grasp > 0:
            # posed object surface points per frame from the stage-3 init
            # trajectory (subsample obj_verts to ~500 for the KD-tree query)
            No = obj_verts.shape[0]
            n_sub = min(500, No)
            sub_idx = np.linspace(0, No - 1, n_sub).astype(int)
            sub = obj_verts[sub_idx]
            R = obj_poses[:, :3, :3]
            t_ = obj_poses[:, :3, 3]
            obj_pts_world = np.einsum("tij,nj->tni", R, sub) + t_[:, None, :]
            grasp_mask = grasp_mask_contact(hand_verts, obj_pts_world)
            n_gf, n_sp = grasp_spans(grasp_mask)
            log(f"T3 contact grasp: {n_gf}/{T} frames in contact over "
                f"{n_sp} contiguous span(s) (d_contact=3cm)")
        obj_poses, icp_stats = refine_object_poses(
            obj_verts, obj_faces, obj_poses, s0.assets["depth_dir"],
            s1.assets["masks_dir"], s0["intrinsics"], icp_cfg,
            hand_boxes=s1.get("hand_boxes"), hand_valid=s1.get("hand_valid"),
            obj_colors=s3.get("colors"),          # None for depth-lift fallback
            frames_dir=s0.assets.get("frames_dir"),
            grasp_mask=grasp_mask, wrist_R=wrist_R)
        s_icp = icp_stats.get("global_scale", 1.0)
        s_axes = icp_stats.get("global_scale_axes")
        if s_axes is not None:              # anisotropic (joint silhouette)
            obj_verts = obj_verts * np.asarray(s_axes, dtype=obj_verts.dtype)
            s_icp = float(np.cbrt(np.prod(s_axes)))   # volume-equivalent radius
        elif abs(s_icp - 1.0) > 1e-6:       # fused-cloud metric-scale correction
            obj_verts = obj_verts * s_icp

    objw, _ = all_object_world(obj_verts, obj_faces, obj_poses)
    gaps = np.array([contact_gap(hand_verts[i, contact_idx], objw[i]) for i in range(T)])
    log(f"contact-frame surface gap: min={gaps.min()*1000:.1f}mm "
        f"median={np.median(gaps)*1000:.1f}mm (pre-rectification)")

    meta = {"world_scale": world_scale,
            "gap_min_mm": float(gaps.min() * 1000),
            "gap_median_mm": float(np.median(gaps) * 1000)}
    if icp_stats is not None:
        meta["object_icp"] = icp_stats
    arrays = {"hand_verts": hand_verts, "hand_joints": hand_joints,
              "contact_idx": contact_idx, "obj_verts": obj_verts,
              "obj_faces": obj_faces, "obj_poses": obj_poses,
              "obj_radius": (learned_radius if learned_radius is not None
                             else s3.get("radius", np.array(0.0)) * s_icp),
              "gaps": gaps}
    obj_colors = s3.get("colors")              # present only for textured backends (sam3d)
    if obj_colors is not None:
        arrays["obj_colors"] = obj_colors
    hand_faces = s2.get("hand_faces")          # MANO mesh faces (real HaMeR hand)
    if hand_faces is not None:
        arrays["hand_faces"] = hand_faces
    return Bundle(arrays=arrays, meta=meta)
