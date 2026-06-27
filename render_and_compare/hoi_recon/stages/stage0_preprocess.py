"""Stage 0 — Preprocess & camera.

In:  raw RGB video (cfg.video) or, in mock mode, nothing.
Out: intrinsics[3,3], extrinsics[T,4,4] (world->cam), frames on disk, metric depth.
Backends (real): VIPE (camera), MoGe-v2 / Depth-Anything-V2 (metric depth).
Errors logged: monocular scale ambiguity, camera drift, blur (see DESIGN.md).
"""
from __future__ import annotations

import os

import numpy as np

from ..bundle import Bundle
from ..logging_utils import log
from ..mock.scene import generate_mock_hoi

NAME = "stage0_preprocess"
INDEX = 0


def _decode_video(path, out_dir, max_frames=300):
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    paths, H, W, i = [], None, None, 0
    while i < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]
        p = os.path.join(out_dir, f"{i:05d}.jpg")
        cv2.imwrite(p, frame)
        paths.append(p)
        i += 1
    cap.release()
    return paths, (H, W), float(fps)


def run(ctx) -> Bundle:
    cfg = ctx.cfg
    frames_dir = os.path.join(ctx.stage_dir(NAME), "frames")

    if cfg.video:
        log(f"decoding video {cfg.video}")
        paths, (H, W), fps = _decode_video(cfg.video, frames_dir)
        T = len(paths)
        log(f"decoded {T} frames @ {fps:.1f}fps, {W}x{H}")
    else:
        T, (H, W), fps = int(cfg.num_frames), (480, 640), 30.0
        paths = []

    if cfg.mock:
        # Use the synthetic scene's (consistent) camera so downstream mock stages
        # share one coordinate frame. Real depth is left symbolic in mock.
        scene = generate_mock_hoi(T, seed=cfg.seed, image_size=(H, W), fps=fps)
        K, extr = scene.intrinsics, scene.extrinsics
        T = scene.T
        has_depth = False
        depth_dir = ""
    else:
        if not paths:
            raise RuntimeError("real mode needs --video (no frames decoded)")
        from ..backends.real_perception import run_stage0_geometry
        geo = run_stage0_geometry(cfg, paths, ctx.stage_dir(NAME))
        K, extr = geo["intrinsics"], geo["extrinsics"]
        depth_dir = geo["depth_dir"]
        H, W = geo["image_size"]
        has_depth = True
        # CHOIR / --camera vipe: override extrinsics with the VIPE trajectory
        # (graceful fallback to identity/MoGe if the vipe env is not set up).
        if (cfg.backend.get("camera") if hasattr(cfg.backend, "get") else None) == "vipe":
            from ..backends.real_perception import run_vipe_camera
            vp = run_vipe_camera(cfg, paths, ctx.stage_dir(NAME))
            if vp is not None:
                extr = vp["extrinsics"]; geo["camera_source"] = "vipe"
                log("camera: VIPE extrinsics applied")
            else:
                log("camera: VIPE env not set up; identity extrinsics fallback "
                    "(run scripts/setup_choir_envs.sh for full-faithful CHOIR)", "warn")
        if geo.get("camera_source") in ("da3", "vggt", "vipe"):
            log(f"camera: using {geo['camera_source'].upper()} estimated poses "
                f"(real extrinsics, consistent geometry)")
        else:
            log("camera: identity extrinsics (static-camera assumption); "
                "use --depth da3 for real camera motion", "warn")

    meta = {
        "T": int(T), "H": int(H), "W": int(W), "fps": float(fps),
        "mock": bool(cfg.mock), "seed": int(cfg.seed), "has_depth": has_depth,
        # diagnostics placeholders (real backends fill these for error attribution)
        "camera_reproj_residual_px": None,
        "depth_confidence": None,
    }
    assets = {"frames_dir": frames_dir if paths else "", "depth_dir": depth_dir}
    return Bundle(arrays={"intrinsics": K, "extrinsics": extr}, meta=meta, assets=assets)
