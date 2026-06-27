"""Deterministic synthetic egocentric RGB-D HOI scene (reach->grasp->move->place).
World frame fixed; the head-mounted camera moves (ego motion). Provides masks,
depth, and ground truth for every downstream stage and for error injection."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .geometry import se3, se3_inv, rotvec_to_R, transform_points, uv_sphere, vertex_normals
from .hand import procedural_hand, FINGERS


@dataclass
class EgoHOI:
    T: int; fps: float; image_size: tuple; intrinsics: np.ndarray
    cam_traj: np.ndarray; table_T: np.ndarray
    obj_verts: np.ndarray; obj_faces: np.ndarray; obj_poses_w: np.ndarray
    hand_verts_w: np.ndarray; hand_joints_w: np.ndarray; finger_idx: dict
    obj_mask: np.ndarray; hand_mask: np.ndarray; depth: np.ndarray
    stage_labels: np.ndarray


def _project(P_cam, K):
    z = np.clip(P_cam[:, 2], 1e-6, None)
    uv = (P_cam[:, :2] / z[:, None]) @ np.array([[K[0, 0], 0], [0, K[1, 1]]]).T
    uv = uv + np.array([K[0, 2], K[1, 2]])
    return uv, P_cam[:, 2]


def _splat(uv, z, H, W, rad=3):
    mask = np.zeros((H, W), bool)
    depth = np.zeros((H, W))
    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)
    for du in range(-rad, rad + 1):
        for dv in range(-rad, rad + 1):
            uu = np.clip(u + du, 0, W - 1)
            vv = np.clip(v + dv, 0, H - 1)
            sel = ok
            cur = depth[vv[sel], uu[sel]]
            zz = z[sel]
            take = (cur == 0) | (zz < cur)
            idx_v = vv[sel][take]
            idx_u = uu[sel][take]
            depth[idx_v, idx_u] = zz[take]
            mask[idx_v, idx_u] = True
    return mask, depth


def generate_ego_hoi(num_frames=48, seed=0, image_size=(480, 640), fps=30.0) -> EgoHOI:
    rng = np.random.default_rng(seed)
    T = int(num_frames)
    H, W = image_size
    f = float(max(H, W))
    K = np.array([[f, 0, W / 2.], [0, f, H / 2.], [0, 0, 1.]])
    t = np.linspace(0., 1., T, endpoint=False)

    # camera (head) trajectory cam->world: small sway + yaw, looking +z
    cam_traj = np.zeros((T, 4, 4))
    for i in range(T):
        R = rotvec_to_R(np.array([0.0, 0.15 * np.sin(2 * np.pi * t[i]), 0.0]))
        c = np.array([0.03 * np.sin(2 * np.pi * t[i]), 0.02 * np.sin(2 * np.pi * t[i] + 0.5), 0.0])
        cam_traj[i] = se3(R, c)
    table_T = se3(np.eye(3), np.array([0.0, 0.12, 0.60]))  # table plane in world

    # object in world: rests then is lifted/moved/placed
    R_obj = 0.040
    ov, of = uv_sphere(R_obj, nlat=12, nlon=18)
    cz = 0.60 + 0.0 * t
    lift = 0.05 * 0.5 * (1 - np.cos(2 * np.pi * np.clip((t - 0.3) / 0.4, 0, 1)))  # rises mid-clip
    centers = np.stack([0.02 * np.sin(2 * np.pi * t), 0.10 - lift, cz], 1)
    obj_poses_w = np.zeros((T, 4, 4))
    for i in range(T):
        obj_poses_w[i] = se3(rotvec_to_R(np.array([0., 0.6 * t[i], 0.])), centers[i])

    # hand in world: fingertips approach object near surface, press mid-clip
    hv, hj, fidx = procedural_hand(778, seed)
    bump = 0.5 * (1 + np.cos(2 * np.pi * (t - 0.5)))
    gap = 0.05 * (1 - bump) - 0.004 * bump
    max_fz = hv[:, 2].max()
    root = centers + np.stack([np.zeros(T), np.zeros(T), -R_obj - max_fz + gap], 1)
    hand_verts_w = hv[None] + root[:, None, :]
    hand_joints_w = hj[None] + root[:, None, :]

    # render masks + depth per frame (camera view)
    obj_mask = np.zeros((T, H, W), bool)
    hand_mask = np.zeros((T, H, W), bool)
    depth = np.zeros((T, H, W))
    for i in range(T):
        c2w = cam_traj[i]
        w2c = se3_inv(c2w)
        ow = transform_points(ov, obj_poses_w[i])
        oc = transform_points(ow, w2c)
        hc = transform_points(hand_verts_w[i], w2c)
        ouv, oz = _project(oc, K)
        huv, hz = _project(hc, K)
        om, od = _splat(ouv, oz, H, W)
        hm, hd = _splat(huv, hz, H, W)
        # hand occludes object where nearer: compute mask once, then apply consistently
        hand_nearer = om & hm & (hd < od)
        od[hand_nearer] = 0
        om[hand_nearer] = False
        obj_mask[i] = om
        hand_mask[i] = hm
        depth[i] = np.where(hd > 0, hd, od)

    # stage labels by clip fraction
    lab = np.array(["pre", "grasp", "move", "place", "post"])
    bins = np.clip((t * 5).astype(int), 0, 4)
    stage_labels = lab[bins]

    return EgoHOI(T, fps, (H, W), K, cam_traj, table_T, ov, of, obj_poses_w,
                  hand_verts_w, hand_joints_w, fidx, obj_mask, hand_mask, depth, stage_labels)
