"""Independent HAND metric vs HOT3D GT (UmeTrack) — the hand analogue of
gt_pose_eval_hot3d.py, for validating hand reprojection (ADR-0001 Q5).

kp2d and the hand mask are the optimizer's TARGETS, so scoring against them is circular.
This instead extracts the GT UmeTrack hand (the same mesh the adapter ray-casts into the
depth) per frame in the pinhole-camera frame, and reports symmetric hand chamfer (mm) +
2D projected chamfer (px) of a reconstructed MANO hand against it. Per frame we pick the
GT hand (left/right) nearest the reconstruction, so handedness is automatic.

Usage:
  gt_hand_eval_hot3d.py <clip_dir> <rc_input> <label>=<hand.npz|run_dir> [<label>=... ]
    hand source is either an npz with `hand_verts[T,778,3]` (run_hand_reproj out.npz) or a
    run dir (uses stage7_contact_optim hand_verts). Reports each label vs GT.
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from hand_tracking_toolkit import camera as htt_camera
from hand_tracking_toolkit import dataset as htt_dataset
from hand_tracking_toolkit.hand_models.umetrack_hand_model import forward_kinematics as umetrack_fk

STREAM = "214-1"; OUT = 1024; FOV_DEG = 90.0
R_FP = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])


def T_from(d):
    R = Rotation.from_quat(np.roll(d["quaternion_wxyz"], -1)).as_matrix()
    T = np.eye(4); T[:3, :3], T[:3, 3] = R, d["translation_xyz"]
    return T


def gt_hands(clip):
    """Per-frame list of GT hand vertex arrays (camera frame), one entry per visible side."""
    frames = sorted(glob.glob(f"{clip}/*.objects.json"))
    um_shape = htt_dataset.from_umetrack_hand_model_json(
        json.load(open(f"{clip}/__hand_shapes.json__"))["umetrack"])
    per_frame = []
    for fpath in frames:
        stem = fpath[:-len(".objects.json")]
        cams = json.load(open(f"{stem}.cameras.json"))[STREAM]
        T_wF = T_from(cams["T_world_from_camera"]); T_wP = T_wF.copy()
        T_wP[:3, :3] = T_wF[:3, :3] @ R_FP; T_Pw = np.linalg.inv(T_wP)
        hands = []
        for side, pc in htt_dataset.decode_hand_pose(json.load(open(f"{stem}.hands.json"))).items():
            if pc.umetrack is None:
                continue
            _, verts, _ = umetrack_fk(pc.umetrack, um_shape, requires_mesh=True)
            hands.append(verts.detach().numpy() @ T_Pw[:3, :3].T + T_Pw[:3, 3])
        per_frame.append(hands)
    return per_frame, frames


def load_hand(src):
    if src.endswith(".npz"):
        return np.load(src)["hand_verts"]
    z = np.load(f"{src.rstrip('/')}/stage7_contact_optim/arrays.npz")
    return z["hand_verts"]


def project(P, K):
    z = np.clip(P[:, 2], 1e-4, None)
    return np.stack([K[0, 0] * P[:, 0] / z + K[0, 2], K[1, 1] * P[:, 1] / z + K[1, 2]], 1)


def chamfer(A, B):
    d1 = cKDTree(B).query(A)[0]; d2 = cKDTree(A).query(B)[0]
    return (np.median(d1) + np.median(d2)) / 2


def main():
    clip = sys.argv[1].rstrip("/")
    K = np.load(f"{sys.argv[2].rstrip('/')}/intrinsics.npy")
    labels = [a.split("=", 1) for a in sys.argv[3:]]
    GT, _ = gt_hands(clip)
    print(f"clip {os.path.basename(clip)}: {len(GT)} frames, GT hands/frame "
          f"~{np.mean([len(h) for h in GT]):.1f}")
    for label, src in labels:
        hv = load_hand(src)
        T = min(len(GT), len(hv))
        ch3d, ch2d = [], []
        for t in range(T):
            if not GT[t]:
                continue
            # nearest GT hand by centroid (handedness auto)
            g = min(GT[t], key=lambda gg: np.linalg.norm(gg.mean(0) - hv[t].mean(0)))
            ch3d.append(chamfer(hv[t], g) * 1000)                    # mm
            ch2d.append(chamfer(project(hv[t], K), project(g, K)))   # px
        ch3d, ch2d = np.array(ch3d), np.array(ch2d)
        print(f"  {label:14s} hand_chamfer {np.median(ch3d):5.1f}/{np.percentile(ch3d,90):5.1f} mm"
              f"   reproj {np.median(ch2d):5.1f}/{np.percentile(ch2d,90):5.1f} px   (n={len(ch3d)})")


if __name__ == "__main__":
    main()
