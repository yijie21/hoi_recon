"""Reproject the reconstructed hand / object back onto the original video frames.

Validation byproduct of the pipeline: do the reconstructed 6D object pose and hand
motion line up with the real pixels over time? Renders the metric meshes with the
camera (the pipeline reconstructs per-frame in camera coordinates) via a small
painter's-algorithm rasterizer, alpha-blended over each frame, side-by-side with
the original. Writes object_reproj.mp4 / hand_reproj.mp4 / hoi_reproj.mp4 (+ sample
montages) into the run dir.
"""
from __future__ import annotations

import glob
import os

import numpy as np

from ..bundle import Bundle

SKIN = np.array([235, 200, 170], np.float32)        # RGB


def _project(P3, K):
    z = np.clip(P3[:, 2], 1e-4, None)
    u = K[0, 0] * P3[:, 0] / z + K[0, 2]
    v = K[1, 1] * P3[:, 1] / z + K[1, 2]
    return np.stack([u, v], 1), P3[:, 2]


def render_multi(frame, parts, K, alpha=0.65, light=np.array([0.4, 0.5, -1.0])):
    """Alpha-blend one or more shaded meshes onto frame (BGR). All faces across all
    parts are z-sorted together so the hand and object occlude each other correctly.
    parts: list of (verts_cam[N,3], faces[M,3], face_rgb[M,3]); face_rgb is RGB."""
    import cv2
    H, W = frame.shape[:2]
    l = light / np.linalg.norm(light)
    pts_list, fz_list, col_list, valid_list = [], [], [], []
    for verts, faces, frgb in parts:
        uv, z = _project(verts, K)
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        n = np.cross(v1 - v0, v2 - v0)
        n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
        shade = 0.35 + 0.65 * np.clip(np.abs(n @ l), 0, 1)
        fz = z[faces].mean(1)
        fu = uv[faces][:, :, 0]; fv = uv[faces][:, :, 1]
        on = ~((fu < 0).all(1) | (fu >= W).all(1) | (fv < 0).all(1) | (fv >= H).all(1))
        pts_list.append(uv[faces].astype(np.int32))
        fz_list.append(fz)
        col_list.append(np.clip(frgb * shade[:, None], 0, 255))
        valid_list.append((z[faces] > 1e-3).all(1) & on)
    pts = np.concatenate(pts_list); fz = np.concatenate(fz_list)
    col = np.concatenate(col_list); valid = np.concatenate(valid_list)
    vi = np.where(valid)[0]
    order = vi[np.argsort(-fz[vi])]                                  # back-to-front
    mesh = frame.copy(); mask = np.zeros((H, W), np.uint8)
    for fi in order:
        c = col[fi][::-1]                                            # RGB->BGR
        cv2.fillConvexPoly(mesh, pts[fi], (int(c[0]), int(c[1]), int(c[2])), lineType=cv2.LINE_AA)
        cv2.fillConvexPoly(mask, pts[fi], 255)
    out = frame.copy(); m = mask > 0
    out[m] = (frame[m] * (1 - alpha) + mesh[m] * alpha).astype(np.uint8)
    return out


def _label(img, text):
    import cv2
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def generate_overlays(run_dir, stage="stage7_contact_optim", fps=24.0, montage=True):
    """Render object / hand / combined reprojection videos for a finished run.
    Returns the list of written paths (empty if the run has no frames / stage)."""
    import cv2
    s0_dir = os.path.join(run_dir, "stage0_preprocess")
    frames = sorted(glob.glob(os.path.join(s0_dir, "frames", "*.jpg")))
    if not frames or not Bundle.exists(os.path.join(run_dir, stage)):
        return []
    K = Bundle.load(s0_dir)["intrinsics"]
    b = Bundle.load(os.path.join(run_dir, stage))
    ov, ofc = b["obj_verts"], b["obj_faces"].astype(np.int32)
    ocol = (b["obj_colors"].astype(np.float32) if b.get("obj_colors") is not None
            else np.full((len(ov), 3), 180, np.float32))
    obj_frgb = ocol[ofc].mean(1)
    op = b["obj_poses"]
    hv = b["hand_verts"]
    hfc = b.get("hand_faces")
    has_hand = hfc is not None
    if has_hand:
        hfc = hfc.astype(np.int32); hand_frgb = np.tile(SKIN, (len(hfc), 1))
    T = len(frames)

    H, W = cv2.imread(frames[0]).shape[:2]
    sw, sh = W // 2, H // 2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers, paths = {}, []
    for name in (["object", "hand", "hoi"] if has_hand else ["object"]):
        p = os.path.join(run_dir, f"{name}_reproj.mp4")
        writers[name] = cv2.VideoWriter(p, fourcc, fps, (2 * sw, sh)); paths.append(p)
    grid = {n: [] for n in writers}
    grid_frames = {38, 75, 110, 150, 185}
    tags = {"object": "object reproj", "hand": "hand reproj", "hoi": "hand+object reproj"}

    for t in range(T):
        frame = cv2.imread(frames[t])
        R, tt = op[t][:3, :3], op[t][:3, 3]
        ow = ov @ R.T + tt
        panels = {"object": render_multi(frame, [(ow, ofc, obj_frgb)], K, alpha=0.7)}
        if has_hand:
            panels["hand"] = render_multi(frame, [(hv[t], hfc, hand_frgb)], K, alpha=0.6)
            panels["hoi"] = render_multi(frame, [(ow, ofc, obj_frgb),
                                                 (hv[t], hfc, hand_frgb)], K, alpha=0.65)
        orig = _label(cv2.resize(frame, (sw, sh)), "original")
        for name, over in panels.items():
            row = np.hstack([orig.copy(),
                             _label(cv2.resize(over, (sw, sh)), f"{tags[name]}  f{t}")])
            writers[name].write(row)
            if t in grid_frames:
                grid[name].append(row)
    for name, w in writers.items():
        w.release()
        if montage and grid[name]:
            cv2.imwrite(os.path.join(run_dir, f"{name}_reproj_grid.png"), np.vstack(grid[name]))
    return paths
