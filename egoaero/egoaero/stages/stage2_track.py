"""Stage 2 (§2.1.2 / App A): asset-free object tracking.
Task 8: coarse RANSAC rigid init. Task 9: memory-pool pose-graph optimization."""
from __future__ import annotations
import numpy as np
from ..bundle import Bundle
from ..core.geometry import (se3, se3_inv, transform_points, umeyama, knn,
                             R_to_rotvec, geodesic_deg)

NAME = "stage2_track"; INDEX = 2

def back_project(depth, mask, K, cam2world):
    """Back-project masked depth to 3D points in world coordinates.

    Args:
        depth: (H, W) depth image
        mask: (H, W) binary mask
        K: (3, 3) camera intrinsics
        cam2world: (4, 4) camera to world transform

    Returns:
        pts_world: (N, 3) 3D points in world coordinates
    """
    ys, xs = np.where(mask & (depth > 0))
    z = depth[ys, xs]
    x = (xs - K[0, 2]) * z / K[0, 0]
    y = (ys - K[1, 2]) * z / K[1, 1]
    pts_cam = np.stack([x, y, z], 1)
    return transform_points(pts_cam, cam2world)

def coarse_pose(prev_pts, cur_pts, ransac_thresh, iters, rng):
    """RANSAC rigid fit mapping prev_pts -> cur_pts (assumed corresponded).

    Samples 3 points, estimates rigid transform via Umeyama (with_scale=False),
    counts inliers, keeps the best hypothesis, and refits on inliers.

    Args:
        prev_pts: (N, 3) source points
        cur_pts: (N, 3) destination points (correspondence assumed)
        ransac_thresh: inlier distance threshold
        iters: number of RANSAC iterations
        rng: numpy random generator

    Returns:
        T_rel: (4, 4) best SE3 transform (prev_pts -> cur_pts)
        inlier_mask: (N,) boolean inlier mask
    """
    n = prev_pts.shape[0]
    best_inl = None
    best_T = np.eye(4)

    for _ in range(int(iters)):
        # Sample 3 random points
        sel = rng.choice(n, 3, replace=False)

        # Estimate rigid transform via Umeyama (no scale)
        _, R, t = umeyama(prev_pts[sel], cur_pts[sel], with_scale=False)
        T = se3(R, t)

        # Count inliers
        res = np.linalg.norm(transform_points(prev_pts, T) - cur_pts, axis=1)
        inl = res < ransac_thresh

        # Keep best hypothesis
        if best_inl is None or inl.sum() > best_inl.sum():
            best_inl, best_T = inl, T

    # Refit on inliers if we have enough
    if best_inl is not None and best_inl.sum() >= 3:
        _, R, t = umeyama(prev_pts[best_inl], cur_pts[best_inl], with_scale=False)
        best_T = se3(R, t)

    return best_T, best_inl
