"""Per-frame object 6D pose by render-and-compare against the SAM2 mask.

The depth-lift centroid gives a good per-frame object TRANSLATION (it reprojects
onto the real object to a few px), but the rotation was previously borrowed from
the hand (a wrist proxy) -> ~16deg orientation error. This module instead recovers
the object's ROTATION from the object's own image evidence: it renders the SAM-3D
mesh's silhouette at a candidate pose and maximises overlap (IoU) with the SAM2
object mask, frame by frame.

Design choices for robustness + speed (pure numpy + cv2, no differentiable
renderer / new env):
  * silhouette  = filled convex hull of the projected mesh vertices (1 polygon
    fill per evaluation) -> fast; captures the object's position, elongation
    direction and foreshortening, which is what pins the visible orientation.
  * translation = the depth-lift centroid (kept fixed); only rotation is solved.
  * anchor      = the largest-mask frame: a coarse SO(3) search seeds the absolute
    orientation from the image (replacing the canonical guess); then the pose is
    tracked bidirectionally with a greedy local refine, so motion stays smooth.
A near-symmetric object's spin about its long axis is unobservable from the
silhouette and is left to the temporal prior; the visible tilt is recovered.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except Exception:                                  # pragma: no cover
    cv2 = None


def _rodrigues(r):
    th = float(np.linalg.norm(r))
    if th < 1e-9:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def _align(a, b):
    """Rotation mapping unit vector a onto unit vector b."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(a @ b)
    if c < -0.9999:
        perp = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        axis = np.cross(a, perp); axis /= np.linalg.norm(axis)
        return _rodrigues(axis * np.pi)
    s = np.linalg.norm(v)
    if s < 1e-9:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def _fib_sphere(n):
    """n roughly-uniform directions on the unit sphere (Fibonacci)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    gold = np.pi * (1 + 5 ** 0.5)
    theta = gold * i
    return np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta),
                     np.cos(phi)], 1)


def _silhouette(vc, K, H, W):
    """Filled convex hull of projected vertices (binary mask)."""
    z = np.clip(vc[:, 2], 1e-4, None)
    u = K[0, 0] * vc[:, 0] / z + K[0, 2]
    v = K[1, 1] * vc[:, 1] / z + K[1, 2]
    pts = np.stack([u, v], 1)
    ok = np.isfinite(pts).all(1) & (vc[:, 2] > 1e-3)
    pts = pts[ok]
    sil = np.zeros((H, W), np.uint8)
    if len(pts) < 3:
        return sil
    hull = cv2.convexHull(pts.astype(np.float32))
    cv2.fillConvexPoly(sil, hull.astype(np.int32), 1)
    return sil


def _iou(sil, mask):
    inter = np.logical_and(sil, mask).sum()
    union = np.logical_or(sil, mask).sum()
    return inter / max(union, 1)


def _cost(R, v, K, H, W, mask, cent):
    vc = v @ R.T + cent
    return 1.0 - _iou(_silhouette(vc, K, H, W).astype(bool), mask)


def _refine(R0, v, K, H, W, mask, cent, iters=30, step0=0.18, seed=0):
    """Greedy random local search on the rotation (axis-angle perturbations)."""
    rng = np.random.default_rng(seed)
    best, bc = R0, _cost(R0, v, K, H, W, mask, cent)
    step = step0
    for _ in range(iters):
        improved = False
        for _ in range(6):
            cand = _rodrigues(rng.normal(0, step, 3)) @ best
            c = _cost(cand, v, K, H, W, mask, cent)
            if c < bc:
                best, bc, improved = cand, c, True
        if not improved:
            step *= 0.6
            if step < 0.012:
                break
    return best, bc


def _anchor_search(v, long_axis, K, H, W, mask, cent, n_dir=60, n_spin=6):
    """Coarse SO(3) search for the absolute orientation at the anchor frame."""
    best, bc = np.eye(3), 1.0
    dirs = _fib_sphere(n_dir)
    spins = np.linspace(0, 2 * np.pi, n_spin, endpoint=False)
    for d in dirs:
        base = _align(long_axis, d)
        for s in spins:
            R = _rodrigues(d * s) @ base
            c = _cost(R, v, K, H, W, mask, cent)
            if c < bc:
                best, bc = R, c
    return _refine(best, v, K, H, W, mask, cent, iters=40)


def track_object_rotation(verts, centroids, mask_paths, K, grasp, scale=0.25,
                          log=None):
    """Per-frame object rotation from silhouette matching.

    verts:       (No,3) centered, metric, oriented object mesh vertices.
    centroids:   (T,3) per-frame object translation (depth-lift, image-grounded).
    mask_paths:  list[str|None] SAM2 object mask per frame.
    K:           (3,3) intrinsics at full resolution.
    grasp:       (T,) bool — frames to track (object visible / interacting).
    Returns rotations (T,3,3); non-grasp frames get identity.
    """
    assert cv2 is not None, "opencv required for object pose tracking"
    T = len(centroids)
    # mesh long axis (for anchor seeding)
    long_axis = np.linalg.svd(verts - verts.mean(0), full_matrices=False)[2][0]

    Ks = K.copy(); Ks[:2] *= scale
    masks, areas = {}, {}
    for t in range(T):
        if grasp[t] and mask_paths[t] is not None:
            m = np.load(mask_paths[t]).astype(bool)
            H0, W0 = m.shape
            ms = cv2.resize(m.astype(np.uint8), (int(W0 * scale), int(H0 * scale)),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
            masks[t] = ms; areas[t] = int(ms.sum())
    Rs = np.tile(np.eye(3), (T, 1, 1))
    if not areas:
        return Rs
    H, W = next(iter(masks.values())).shape
    anchor = max(areas, key=areas.get)
    Ra, ca = _anchor_search(verts, long_axis, Ks, H, W, masks[anchor], centroids[anchor])
    Rs[anchor] = Ra
    if log:
        log(f"object pose: anchor frame {anchor} silhouette IoU={1 - ca:.2f} "
            f"(coarse SO(3) search + refine)")

    frames = sorted(masks.keys())
    ai = frames.index(anchor)
    # track forward then backward from the anchor, seeding from the neighbour pose
    for seq in (frames[ai + 1:], frames[ai - 1::-1] if ai > 0 else []):
        prev = Ra
        for t in seq:
            Rs[t], _ = _refine(prev, verts, Ks, H, W, masks[t], centroids[t],
                               iters=22, step0=0.14, seed=t)
            prev = Rs[t]
    # hold the nearest tracked rotation through untracked (occluded / off-screen)
    # frames so the object orientation stays continuous
    last = None
    for t in range(T):
        if t in masks:
            last = Rs[t]
        elif last is not None:
            Rs[t] = last
    last = None
    for t in range(T - 1, -1, -1):
        if t in masks:
            last = Rs[t]
        elif last is not None:
            Rs[t] = last
    return Rs
