"""Stage 2 — Hand reconstruction (per-frame MANO -> world-space motion).

In:  frames + hand boxes/sides (stage1) + camera (stage0).
Out: betas[10], orient[T,3], pose[T,45], transl[T,3], joints[T,21,3],
     verts[T,778,3], contact_idx[Nc].
Backends (real): HaMeR (+ Dyn-HaMR for world stabilization) / WiLoR / HaWoR.
Errors logged: 2D-keypoint reprojection vs wrist depth, jitter (accel), beta var.

Mock: take ground-truth hand and inject the *dominant* monocular hand errors —
zero-mean per-frame depth/translation jitter (depth >> lateral) plus small mesh
noise. These are exactly the errors stage5 temporal smoothing is meant to remove.
"""
from __future__ import annotations

import numpy as np

from ..bundle import Bundle
from ..logging_utils import log
from ..mock.scene import generate_mock_hoi

NAME = "stage2_hand"
INDEX = 2


def run(ctx) -> Bundle:
    cfg = ctx.cfg
    s0 = ctx.load("stage0_preprocess")
    T = int(s0.meta["T"])

    if cfg.mock:
        scene = generate_mock_hoi(T, seed=cfg.seed,
                                  image_size=(s0.meta["H"], s0.meta["W"]),
                                  fps=s0.meta["fps"])
        rng = np.random.default_rng(cfg.seed + 101)
        # per-frame rigid jitter: depth (z) ambiguity dominates lateral error
        jit = rng.normal(0, 1, (T, 3)) * np.array([0.0015, 0.0015, 0.006])
        mesh_noise = rng.normal(0, 0.0010, scene.hand_verts.shape)
        verts = scene.hand_verts + jit[:, None, :] + mesh_noise
        joints = scene.hand_joints + jit[:, None, :]
        transl = scene.hand_joints[:, 0, :] + jit       # wrist as MANO transl
        orient = np.zeros((T, 3))
        pose = np.zeros((T, 45))
        betas = np.zeros(10)
        contact_idx = scene.contact_idx
        log(f"hand: {verts.shape[1]} verts, {len(contact_idx)} contact candidates, "
            f"injected depth jitter sigma=6mm")
        meta = {"hand_side": "right", "kp_reproj_px": None,
                "jitter_accel": float(np.mean(np.abs(np.diff(joints, 2, axis=0)))),
                "beta_var": 0.0}
        return Bundle(
            arrays={"betas": betas, "orient": orient, "pose": pose,
                    "transl": transl, "joints": joints, "verts": verts,
                    "contact_idx": contact_idx},
            meta=meta)

    from ..backends.perception import run_hand
    s1 = ctx.load("stage1_detect_track")
    out = run_hand(cfg, s0.assets.get("frames_dir"), s1.arrays, s0.arrays)
    return Bundle(arrays=out, meta={"hand_side": "right"})
