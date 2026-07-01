"""ForeHOI output dir (output_4d/<name>/) -> workbench scene.npz.

ForeHOI emits frames.npz (poses[T,4,4], hand_verts[T,Hmax,778,3] NaN-padded, hand_count[T],
hand_faces[1538,3]) + object_mesh.glb (canonical). Object world = mverts@R.T + t per frame;
hand verts already in camera frame (metres, OpenCV).

Usage:
  python compare/adapters/forehoi_to_scene.py forehoi/output_4d/wild1 compare/scenes/forehoi.npz
"""
import sys, os
import numpy as np
import trimesh


def main(bundle, out_npz):
    d = np.load(os.path.join(bundle, "frames.npz"))
    poses = d["poses"].astype(np.float64)                       # [T,4,4]
    hv = d["hand_verts"].astype(np.float32)                     # [T,Hmax,778,3]
    T = poses.shape[0]
    # take first detected hand each frame; fill missing with the previous valid frame
    hand = hv[:, 0, :, :]
    for t in range(T):
        if not np.isfinite(hand[t]).all():
            hand[t] = hand[t - 1] if t else np.nan_to_num(hand[t])
    hfaces = d["hand_faces"].astype(np.int32)

    mesh = trimesh.load(os.path.join(bundle, "object_mesh.glb"), force="mesh")
    ov = np.asarray(mesh.vertices, np.float32)
    of = np.asarray(mesh.faces, np.int32)

    name = os.path.basename(bundle.rstrip("/"))
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez(out_npz,
             hand_verts=hand, hand_faces=hfaces,
             obj_verts=ov, obj_faces=of, obj_poses=poses,
             source=f"ForeHOI ({name}, {T}f)")
    print(f"wrote {out_npz}: hand {hand.shape}, obj mesh {ov.shape}/{of.shape}, T={T}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
