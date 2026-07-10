"""Extract GT UmeTrack hand meshes into the rectified pinhole frame the pipeline uses,
and save them for overlay. make_rc_input.py already computes exactly these (to ray-cast
the GT depth) but discards them; this re-derives just the hand FK + the world->pinhole
transform (no image remap, no depth raycast), so it is fast.

GT hands are UmeTrack (toolkit-native, per-clip shape in __hand_shapes.json__), NOT MANO:
HOT3D's MANO thetas are unreliable and the only MANO_LEFT.pkl on this box is a fabricated
right-hand mirror. UmeTrack needs no licensed model files and is mocap-grade.

Usage: extract_gt_hands.py <rc_input_dir>   -> writes <rc_input_dir>/gt_hands.npz
  {faces:(Nf,3), verts: object[T] each (K_t, Nv, 3) hands in the rectified camera frame}
"""
import glob
import json
import os
import sys

import numpy as np
from hand_tracking_toolkit import dataset as htt_dataset
from hand_tracking_toolkit.hand_models.umetrack_hand_model import (
    forward_kinematics as umetrack_fk)
from scipy.spatial.transform import Rotation

STREAM = "214-1"
R_FP = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])


def T_from(d):
    R = Rotation.from_quat(np.roll(d["quaternion_wxyz"], -1)).as_matrix()
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, d["translation_xyz"]
    return T


def main():
    rc = sys.argv[1].rstrip("/")
    clip = json.load(open(f"{rc}/meta.json"))["clip"]
    frames = sorted(glob.glob(f"{clip}/*.objects.json"))
    um_shape = htt_dataset.from_umetrack_hand_model_json(
        json.load(open(f"{clip}/__hand_shapes.json__"))["umetrack"])

    faces = None
    per_frame = []
    for fpath in frames:
        stem = fpath[:-len(".objects.json")]
        T_wF = T_from(json.load(open(f"{stem}.cameras.json"))[STREAM]["T_world_from_camera"])
        T_wP = T_wF.copy()
        T_wP[:3, :3] = T_wF[:3, :3] @ R_FP
        T_Pw = np.linalg.inv(T_wP)                       # world -> rectified pinhole
        hands = []
        for side, pc in htt_dataset.decode_hand_pose(
                json.load(open(f"{stem}.hands.json"))).items():
            if pc.umetrack is None:
                continue
            _, verts, f = umetrack_fk(pc.umetrack, um_shape, requires_mesh=True)
            hv = verts.detach().numpy() @ T_Pw[:3, :3].T + T_Pw[:3, 3]   # camera frame
            hands.append(hv.astype(np.float32))
            if faces is None:
                faces = f.detach().numpy().astype(np.int64)
        per_frame.append(np.stack(hands) if hands else np.zeros((0, 0, 3), np.float32))

    verts = np.empty(len(per_frame), dtype=object)
    for i, h in enumerate(per_frame):
        verts[i] = h
    np.savez(f"{rc}/gt_hands.npz", faces=faces if faces is not None else np.zeros((0, 3), np.int64),
             verts=verts, allow_pickle=True)
    nz = sum(len(h) for h in per_frame)
    print(f"wrote {rc}/gt_hands.npz  frames={len(per_frame)} total-hand-instances={nz} "
          f"faces={None if faces is None else faces.shape}")


if __name__ == "__main__":
    main()
