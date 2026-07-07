"""Locked-scale rigid ICP object placement: register the canonical object mesh
per frame onto the depth point cloud inside the object mask.

Replaces the depth-lift-anchored object trajectory with a direct geometric
registration against the depth substrate. Scale is deliberately NOT a per-frame
free variable: partial-view fits cannot observe it (a 53% oversized mesh fits
the visible front at unchanged residual — see compare/hoi4d/gate2/sam3d_icp/),
so the object stays a rigid body. What IS observable is ONE global scale from
the union of all registered frames (`global_scale_refit`): the fused
multi-frame cloud covers enough of the object to pin the metric size that the
per-frame front views hide (kettle_N15: stage-3's bbox heuristic was 13%
undersized, per-frame bias showed only −0.7 cm; the fused refit recovered
s=1.13 and cut the per-frame residual 6.3→3.7 mm — Route B of RESULTS.md).

Validated on kettle_N15 vs GT depth: visible-surface MAE 2.97 → 1.12 cm,
wiggle 1.58 → 0.79 cm over stages 4-7 (rigid), plus the scale-refit gains
above. Numpy/scipy/cv2/trimesh only — runs in-process in the main env.
"""
from __future__ import annotations

import os

import numpy as np

from .logging_utils import log


def _umeyama_rigid(src, dst):
    """Rigid (R, t) minimizing ||R@src + t - dst|| (Kabsch)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    cov = (dst - mu_d).T @ (src - mu_s) / len(src)
    U, _, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    return R, mu_d - R @ mu_s


def _icp(src_pts, src_tree, tgt, R0, t0, trim, iters, rot_free=True):
    """Trimmed rigid ICP; correspondences target -> canonical mesh samples
    (the target is a partial front view, so every target point has a match).
    rot_free=False keeps R fixed at R0 and solves translation only — use when
    rotation comes from image evidence (stage-3 silhouette tracker): the
    top-down depth patch is near rotation-symmetric AND trimming discards the
    few disambiguating points (spout/handle), so free rotation walks into a
    wrong basin (33-97 deg off on kettle_N15) that depth residuals can't see."""
    R, t = R0.copy(), t0.copy()
    prev = np.inf
    res = prev
    for _ in range(iters):
        d, j = src_tree.query((tgt - t) @ R, workers=-1)
        keep = d <= np.quantile(d, trim)
        if rot_free:
            R, t = _umeyama_rigid(src_pts[j[keep]], tgt[keep])
        else:
            t = tgt[keep].mean(0) - R @ src_pts[j[keep]].mean(0)
        res = float(np.sqrt(np.mean(
            np.sum((src_pts[j[keep]] @ R.T + t - tgt[keep]) ** 2, 1))))
        if abs(prev - res) < 1e-6:
            break
        prev = res
    return R, t, res


def _solve_shared_scale(src_tree, src_pts, qs, s0, trim=0.95, iters=5):
    """One global scale about the canonical origin from the FUSED cloud (all
    frames' canonical-frame targets q, poses frozen): q ~ s*m. No per-frame
    centering and a mild trim — per-frame centering + tight trims remove the
    very evidence (radial extent mismatch, boundary correspondences) that
    makes global scale observable; the validated Route-B solve is exactly
    this fixed-pose fused-cloud fit (compare/hoi4d/gate2/sam3d_icp)."""
    q = np.concatenate(qs)
    s = s0
    for _ in range(iters):
        _, j = src_tree.query(q / s, workers=-1)
        m = src_pts[j]
        r = np.linalg.norm(s * m - q, axis=1)
        keep = r <= np.quantile(r, trim)
        s_new = float((m[keep] * q[keep]).sum() / (m[keep] * m[keep]).sum())
        if abs(s_new / s - 1.0) < 1e-4:
            return s_new
        s = s_new
    return s


def refine_object_poses(obj_verts, obj_faces, poses0, depth_dir, masks_dir, K,
                        opts=None):
    """Return (poses[T,4,4], stats) — per-frame rigid registration of the
    canonical mesh onto masked depth. Frame 0 initializes from poses0[0];
    frame t from the t-1 result (poses0[t] as re-init after gaps). Frames
    without usable depth keep their poses0. With `global_scale_refit`, the
    registration alternates with a shared-scale solve over all frames'
    pooled correspondences; stats["global_scale"] then carries the factor
    the CALLER must apply to the canonical verts (poses stay det=1 rigid)."""
    import cv2
    import trimesh
    from scipy.spatial import cKDTree

    o = opts or {}
    get = o.get if hasattr(o, "get") else lambda k, d: getattr(o, k, d)
    trim = float(get("trim", 0.8))
    iters = int(get("iters", 60))
    erode = int(get("erode_px", 5))
    n_tgt = int(get("max_points", 3000))
    n_src = int(get("mesh_samples", 20000))
    min_pts = int(get("min_points", 200))
    scale_refit = bool(get("global_scale_refit", False))
    passes = int(get("scale_refit_rounds", 2)) if scale_refit else 1
    rot_free = str(get("rotation", "free")) == "free"
    fg_band = float(get("fg_band_m", 0.15))

    mesh = trimesh.Trimesh(obj_verts, np.asarray(obj_faces, int), process=False)
    src_pts = np.asarray(trimesh.sample.sample_surface(mesh, n_src, seed=0)[0])
    rng = np.random.default_rng(0)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode + 1,) * 2)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    T = len(poses0)
    targets = [None] * T
    for i in range(T):
        dp = os.path.join(depth_dir, f"{i:05d}.npy")
        mp = os.path.join(masks_dir, f"{i:05d}.npy")
        if not (os.path.exists(dp) and os.path.exists(mp)):
            continue
        g = np.load(dp).astype(np.float32)
        m = (cv2.erode(np.load(mp).astype(np.uint8), ker) > 0) \
            & (g > 0.25) & (g < 5.0)
        if m.sum() < min_pts:
            continue
        ys, xs = np.nonzero(m)
        z = g[ys, xs]
        # foreground band: real sensor depth bleeds background values into the
        # mask at the boundary (halo/misregistration — measured ~8% of the
        # ERODED kettle_N15 mask is >12cm off the object); reject them here or
        # they survive the trims and inflate the scale solve.
        if fg_band > 0:
            keep = np.abs(z - np.median(z)) < fg_band
            if keep.sum() >= min_pts:
                ys, xs, z = ys[keep], xs[keep], z[keep]
        P = np.stack([(xs - cx) / fx * z, (ys - cy) / fy * z, z], 1)
        if len(P) > n_tgt:
            P = P[rng.choice(len(P), n_tgt, replace=False)]
        targets[i] = P

    src_tree = cKDTree(src_pts)

    def registration_pass(s):
        src_s = src_pts * s
        tree = cKDTree(src_s)
        poses = poses0.copy()
        resid = np.full(T, np.nan)
        qs = []
        R, t = None, None
        for i, P in enumerate(targets):
            if P is None:
                R, t = None, None
                continue
            if R is None:                   # (re)start from the prior trajectory
                R, t = poses0[i][:3, :3].copy(), poses0[i][:3, 3].copy()
            if not rot_free:                # image-informed per-frame rotation
                R = poses0[i][:3, :3].copy()
            R, t, resid[i] = _icp(src_s, tree, P, R, t, trim, iters, rot_free)
            poses[i] = np.eye(4)
            poses[i][:3, :3], poses[i][:3, 3] = R, t
            if scale_refit:                 # this frame's cloud in canonical frame
                qs.append((P - t) @ R)
        return poses, resid, qs

    s = 1.0
    for p in range(passes):
        poses, resid, qs = registration_pass(s)
        if p == passes - 1 or not qs:
            break
        s_new = _solve_shared_scale(src_tree, src_pts, qs, s)
        if abs(s_new / s - 1.0) < 1e-3:
            break
        s = s_new

    ok = ~np.isnan(resid)
    n_reg = int(ok.sum())
    moved = np.linalg.norm(poses[ok][:, :3, 3] - poses0[ok][:, :3, 3], axis=1)
    stats = {"frames_registered": n_reg, "frames_total": int(T),
             "icp_resid_mm_med": float(np.nanmedian(resid) * 1000),
             "icp_resid_mm_p90": float(np.nanpercentile(resid, 90) * 1000),
             "moved_cm_med": float(np.median(moved) * 100) if n_reg else 0.0,
             "global_scale": float(s)}
    log(f"object ICP: {n_reg}/{T} frames registered; residual "
        f"med={stats['icp_resid_mm_med']:.1f}mm p90={stats['icp_resid_mm_p90']:.1f}mm; "
        f"moved med={stats['moved_cm_med']:.1f}cm vs prior trajectory"
        + (f"; global scale refit s={s:.4f}" if scale_refit else ""))
    return poses, stats
