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

    from ..backends.perception import run_object
    s1 = ctx.load("stage1_detect_track")
    out = run_object(cfg, s0.assets.get("frames_dir"), s1.arrays, None, s0.arrays)
    return Bundle(arrays=out, meta={"branch": cfg.backend.object})
