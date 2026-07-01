"""HORT output -> workbench scene.npz.

HORT (single image) emits per image:
  <out>/<name>.obj   hand MANO mesh (778 verts, 1552 faces), camera frame, metres
  <out>/<name>.json  object point cloud + palm/trans (object-local cloud)
Object world cloud = pointclouds_up + handpalm + objtrans  (cam_extr is identity).

Usage:
  python compare/adapters/hort_to_scene.py hort/out_demo/f0030 compare/scenes/hort.npz
"""
import sys, json, os
import numpy as np


def read_obj(path):
    V, F = [], []
    for ln in open(path):
        if ln.startswith("v "):
            V.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith("f "):
            F.append([int(p.split("/")[0]) - 1 for p in ln.split()[1:4]])
    return np.asarray(V, np.float32), np.asarray(F, np.int32)


def main(stem, out_npz):
    V, F = read_obj(stem + ".obj")
    j = json.load(open(stem + ".json"))
    pc = np.asarray(j["pointclouds_up"], np.float32)         # object-local
    palm = np.asarray(j["handpalm"], np.float32).reshape(3)
    trans = np.asarray(j["objtrans"], np.float32).reshape(3)
    obj_world = pc + palm + trans                            # camera frame, metres

    # HORT is single-frame -> T=1
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez(out_npz,
             hand_verts=V[None],            # [1,778,3]
             hand_faces=F,                  # [1552,3]
             obj_points=obj_world[None],    # [1,M,3]
             obj_point_colors=np.tile([90, 150, 230], (obj_world.shape[0], 1)).astype(np.uint8),
             source="HORT (feed-forward, 1 image)")
    print(f"wrote {out_npz}: hand {V.shape}, obj_points {obj_world.shape}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
