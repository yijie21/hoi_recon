"""Deterministic synthetic HOI scene with ground-truth contact.

This is the backbone of `mock` mode. It generates a physically-sensible reach →
grasp → retract sequence: a spherical object in front of the camera and a hand
whose fingertips approach and lightly contact the object's near surface around the
middle of the clip. Everything is analytic and seedable, so:

  * stages 2/3 can inject *realistic perception error* on top of ground truth, and
  * stage 8 can re-derive ground truth and measure each stage's error.

All units are metres, camera at the origin looking down +z.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import se3, rotvec_to_R, transform_points, uv_sphere, vertex_normals


@dataclass
class MockHOI:
    T: int
    fps: float
    image_size: tuple             # (H, W)
    intrinsics: np.ndarray        # [3,3]
    extrinsics: np.ndarray        # [T,4,4] world->cam (identity: world==cam here)
    # object (ground truth)
    obj_verts: np.ndarray         # [No,3] canonical, centred at origin
    obj_faces: np.ndarray         # [Mo,3]
    obj_radius: float
    obj_poses: np.ndarray         # [T,4,4] object->world
    # hand (ground truth)
    hand_verts: np.ndarray        # [T,Nh,3] world
    hand_joints: np.ndarray       # [T,21,3] world
    contact_idx: np.ndarray       # [Nc] indices into hand verts that can touch
    # contact (ground truth)
    gt_contact_mask: np.ndarray   # [T,Nc] bool — candidate vert in contact this frame


def _canonical_hand(n: int, rng: np.random.Generator):
    """Procedural right hand point cloud, fingertips pointing +z (toward object)."""
    pts = []
    n_palm = n // 3
    pts.append(rng.uniform([-0.040, -0.012, -0.030], [0.040, 0.012, 0.005], (n_palm, 3)))
    n_fing = n - n_palm
    per = n_fing // 5
    base_x = np.linspace(-0.035, 0.035, 5)
    lengths = [0.045, 0.060, 0.065, 0.060, 0.050]  # thumb .. pinky
    for fi in range(5):
        cnt = per if fi < 4 else n_fing - per * 4
        z = rng.uniform(0.0, lengths[fi], cnt)
        x = base_x[fi] + rng.uniform(-0.006, 0.006, cnt)
        y = rng.uniform(-0.008, 0.008, cnt)
        pts.append(np.stack([x, y, z], 1))
    P = np.concatenate(pts, 0)
    contact_idx = np.where(P[:, 2] > 0.035)[0]          # fingertip region
    # 21 landmark joints: wrist + 4 along each finger
    joints = [[0.0, 0.0, -0.020]]
    for fi in range(5):
        for fr in (0.25, 0.5, 0.75, 1.0):
            joints.append([base_x[fi], 0.0, fr * lengths[fi]])
    return P.astype(np.float64), np.asarray(joints), contact_idx


def generate_mock_hoi(num_frames: int = 48, seed: int = 0,
                      image_size=(480, 640), fps: float = 30.0) -> MockHOI:
    rng = np.random.default_rng(seed)
    T = int(num_frames)
    H, W = image_size
    f = float(max(H, W))
    K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])
    extr = np.tile(np.eye(4), (T, 1, 1))  # world == camera frame in mock

    # --- object ---------------------------------------------------------
    R_obj = 0.040
    ov, of = uv_sphere(R_obj, nlat=12, nlon=18)
    t = np.linspace(0.0, 1.0, T)
    cx = 0.02 * np.sin(2 * np.pi * t)          # gentle object motion
    cy = 0.01 * np.sin(2 * np.pi * t + 0.7)
    cz = 0.60 + 0.0 * t
    centers = np.stack([cx, cy, cz], 1)
    obj_poses = np.zeros((T, 4, 4))
    for i in range(T):
        R = rotvec_to_R(np.array([0.0, 0.6 * t[i], 0.0]))  # slow spin about y
        obj_poses[i] = se3(R, centers[i])

    # --- hand -----------------------------------------------------------
    hand_canon, joints_canon, contact_idx = _canonical_hand(778, rng)
    max_fz = hand_canon[:, 2].max()
    # raised-cosine approach: gap large at ends, light press at middle
    bump = 0.5 * (1.0 + np.cos(2 * np.pi * (t - 0.5)))   # 1 at t=.5, 0 at ends
    gap = 0.05 * (1.0 - bump) - 0.004 * bump             # +5cm .. -4mm press
    near_surface_z = cz - R_obj
    root = np.stack([cx, cy, near_surface_z - max_fz + gap], 1)  # [T,3]

    hand_verts = hand_canon[None] + root[:, None, :]            # [T,Nh,3]
    hand_joints = joints_canon[None] + root[:, None, :]         # [T,21,3]

    # --- ground-truth contact ------------------------------------------
    cand = hand_verts[:, contact_idx, :]                       # [T,Nc,3]
    d = np.linalg.norm(cand - centers[:, None, :], axis=-1) - R_obj
    gt_contact_mask = (d >= -0.020) & (d <= 0.005)

    return MockHOI(
        T=T, fps=fps, image_size=(H, W), intrinsics=K, extrinsics=extr,
        obj_verts=ov, obj_faces=of, obj_radius=R_obj, obj_poses=obj_poses,
        hand_verts=hand_verts, hand_joints=hand_joints,
        contact_idx=contact_idx, gt_contact_mask=gt_contact_mask,
    )
