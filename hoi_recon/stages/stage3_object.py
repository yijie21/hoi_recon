"""Stage 3 — Object shape + 6D pose trajectory (model-free primary branch).

In:  frames + amodal masks + metric depth + camera + point tracks.
Out: verts[No,3] canonical, faces[Mo,3], poses[T,4,4] (object->world),
     scale/radius estimate.
Backends (real): SAM-3D-Objects (anchor mesh + guarded follow-track) and/or
     BundleSDF; CAD branch FoundationPose/MegaPose as calibration control.
Errors logged (highest in model-free): anchor-frame shape/scale ambiguity,
     6D drift in the occluded contact phase, symmetry flips.

Mock: take the ground-truth object and inject the two characteristic model-free
errors — a global *shape-scale* error and a *6D translation drift* (random walk).
These are what stage6 (rectify) + stage7 (contact optim) must undo.
"""
from __future__ import annotations

import numpy as np

from ..bundle import Bundle
from ..geometry import rotvec_to_R, se3
from ..logging_utils import log
from ..mock.scene import generate_mock_hoi

NAME = "stage3_object"
INDEX = 3


def run(ctx) -> Bundle:
    cfg = ctx.cfg
    s0 = ctx.load("stage0_preprocess")
    T = int(s0.meta["T"])

    if cfg.mock:
        scene = generate_mock_hoi(T, seed=cfg.seed,
                                  image_size=(s0.meta["H"], s0.meta["W"]),
                                  fps=s0.meta["fps"])
        rng = np.random.default_rng(cfg.seed + 202)
        # Object SHAPE error caps how well contact alone can recover the object;
        # keep it small by default (clean demo) — raise the spread to study how the
        # shape-error bottleneck degrades contact recovery (see DESIGN.md error budget).
        scale_err = float(rng.uniform(0.99, 1.01))         # near-correct shape
        verts = scene.obj_verts * scale_err
        radius = scene.obj_radius * scale_err

        # Characteristic monocular-object error: a *systematic depth offset* (object
        # reconstructed too far along the camera axis) along the contact normal,
        # plus a small smoothed translation drift and rotation noise.
        depth_off = float(rng.uniform(0.015, 0.030))       # 1.5-3 cm too far in +z
        step = rng.normal(0, 0.002, (T, 3))
        drift = np.cumsum(step, axis=0)
        drift -= drift.mean(0)
        drift = np.clip(drift, -0.015, 0.015)
        drift[:, 2] += depth_off
        poses = np.zeros((T, 4, 4))
        for i in range(T):
            R_gt = scene.obj_poses[i, :3, :3]
            R_n = rotvec_to_R(rng.normal(0, 0.02, 3)) @ R_gt
            t = scene.obj_poses[i, :3, 3] + drift[i]
            poses[i] = se3(R_n, t)
        log(f"object: scale_err={scale_err:.3f}, depth_offset={depth_off*100:.1f}cm, "
            f"max drift={np.abs(drift).max()*100:.1f}cm")
        meta = {"branch": cfg.backend.object, "scale_est": scale_err,
                "silhouette_iou": None, "mask_reproj_iou": None,
                "rot_jump_deg": None}
        return Bundle(
            arrays={"verts": verts, "faces": scene.obj_faces,
                    "poses": poses, "radius": np.array(radius)},
            meta=meta)

    # --- real: model-free object from SAM mask + MoGe depth (depth-lift) ---
    import os
    from ..backends.real_perception import run_object_depthlift, list_frames
    s1 = ctx.load("stage1_detect_track")
    frame_paths = list_frames(s0.assets["frames_dir"])
    depth_dir = s0.assets["depth_dir"]
    masks_dir = s1.assets["masks_dir"]
    depth_paths = [os.path.join(depth_dir, f"{i:05d}.npy") for i in range(T)]
    mask_paths = [os.path.join(masks_dir, f"{i:05d}.npy") for i in range(T)]
    mask_paths = [p if os.path.exists(p) else None for p in mask_paths]

    out, branch, textured = None, cfg.backend.object, False
    if cfg.backend.object == "sam3d":
        from ..backends.real_perception import run_object_sam3d
        try:
            out = run_object_sam3d(cfg, ctx.stage_dir(NAME), frame_paths,
                                   mask_paths, depth_paths, s0["intrinsics"])
            branch, textured = "sam3d", True
        except Exception as e:                 # fail soft to the working depth-lift
            log(f"object: SAM-3D failed ({e}); falling back to depth-lift", "warn")
            out = None
    if out is None:
        out = run_object_depthlift(cfg, frame_paths, mask_paths, depth_paths,
                                   s0["intrinsics"])

    # Object 6D pose. Selectable via cfg.backend.object_pose:
    #   'silhouette'     (default) keep the image-grounded depth-lift translation and
    #                    recover ROTATION by matching the rendered mesh silhouette to
    #                    the SAM2 mask each frame. Best on monocular video.
    #   'foundationpose' full 6D from the FoundationPose RGB-D tracker (needs reliable
    #                    sensor depth; on monocular MoGe depth it tends to drift).
    #   'hand'           inherit rotation from the grasping hand (wrist proxy).
    centroids = out["poses"][:, :3, 3]
    pose_method = (cfg.backend.get("object_pose", "silhouette")
                   if hasattr(cfg.backend, "get") else "silhouette")
    poses = np.tile(np.eye(4), (T, 1, 1))
    poses[:, :3, 3] = centroids
    rot_src = pose_method
    try:
        if pose_method == "foundationpose":
            from ..backends.real_perception import run_object_pose_foundationpose
            poses = run_object_pose_foundationpose(
                cfg, ctx.stage_dir(NAME), frame_paths, mask_paths, depth_paths,
                s0["intrinsics"], out["verts"], out["faces"])
        elif pose_method == "hand":
            from ..backends.real_perception import couple_object_to_hand
            s2 = ctx.load("stage2_hand")
            poses = couple_object_to_hand(out["poses"], s2["joints"],
                                          s1["hand_valid"].astype(bool), float(out["radius"]))
        elif pose_method == "choir_tracker":
            # CHOIR guarded follow-tracker (silhouette track + 60deg angular guard)
            # -> object isolated fit (Eq 2-3, repulsion+attraction) on the fixed mesh.
            from ..object_pose_track import track_object_rotation
            from ..choir import angular_guard
            from ..backends.real_perception import run_choir_object_fit
            visible = np.array([(np.load(p).sum() > 2000) if p else False
                                for p in mask_paths])
            R = track_object_rotation(out["verts"], centroids, mask_paths,
                                      s0["intrinsics"], visible, log=log)
            guard = float(cfg.choir.object.get("guard_deg", 60.0))
            prev = None                           # reject >guard jumps, hold previous
            for i in range(T):
                if prev is not None and visible[i] and not angular_guard(R[i], prev, guard):
                    R[i] = prev
                elif visible[i]:
                    prev = R[i]
            poses[:, :3, :3] = R
            poses = run_choir_object_fit(cfg, ctx.stage_dir(NAME), frame_paths,
                                         mask_paths, s0["intrinsics"], out["verts"],
                                         out["faces"], poses)
            rot_src = "choir_tracker"
        else:                                   # 'silhouette' (default) or 'render_compare'
            from ..object_pose_track import track_object_rotation
            visible = np.array([(np.load(p).sum() > 2000) if p else False
                                for p in mask_paths])
            poses[:, :3, :3] = track_object_rotation(
                out["verts"], centroids, mask_paths, s0["intrinsics"], visible, log=log)
            if pose_method == "render_compare" and textured:
                # differentiable refinement (silhouette + photometric -> recovers spin)
                from ..backends.real_perception import run_object_pose_render_compare
                poses = run_object_pose_render_compare(
                    cfg, ctx.stage_dir(NAME), frame_paths, mask_paths, s0["intrinsics"],
                    out["verts"], out["faces"], out["vertex_colors"], poses)
    except Exception as e:
        rot_src = "fallback(translation-only)"
        log(f"object: pose method '{pose_method}' failed ({e}); "
            f"image-grounded translation + identity rotation", "warn")

    arrays = {"verts": out["verts"], "faces": out["faces"],
              "poses": poses, "radius": out["radius"]}
    if textured:
        arrays["colors"] = out["vertex_colors"]
    return Bundle(arrays=arrays,
                  meta={"branch": branch, "textured": textured,
                        "object_rotation": rot_src,
                        "anchor_frame": int(out["anchor_frame"])})
