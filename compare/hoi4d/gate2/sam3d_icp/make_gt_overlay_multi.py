"""GT CAD overlay for ANY HOI4D rigid-category sequence: pose the
bbox-centered CAD by the per-frame objpose annotation (euler intrinsic
"XYZ", translation = box center) and splat it, shaded green, over the
sequence's align_rgb video.

Generalization check for the conventions established on kettle_N15
(RESULTS.md): frame indexing 1:1, euler XYZ, bbox-centered CAD, per-camera
intrinsics from camera_params. Frames whose objpose json is missing or
marked isEffective=0 are shown raw with a "no annotation" tag.

Usage: make_gt_overlay_multi.py <SEQ> [out.mp4]
  e.g. make_gt_overlay_multi.py ZY20210800002/H2/C5/N22/S263/s03/T3
"""
import json
import os
import sys

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

DS = "/workspace/datasets/hoi4d"
CAT = {1: "ToyCar", 2: "Mug", 5: "Bottle", 7: "Bowl", 12: "Kettle",
       13: "Knife", 20: "Chair"}
B = 2
N = 150000
FONT = cv2.FONT_HERSHEY_SIMPLEX


def main():
    seq = sys.argv[1].strip("/")
    cam, _, c_part, n_part = seq.split("/")[:4]
    cad_path = f"{DS}/HOI4D_CAD_Model_for_release/rigid/" \
               f"{CAT[int(c_part[1:])]}/{int(n_part[1:]):03d}.obj"
    out = sys.argv[2] if len(sys.argv) > 2 else \
        f"gt_overlay_{CAT[int(c_part[1:])].lower()}_{n_part}.mp4"

    cad = trimesh.load(cad_path, force="mesh")
    P, fidx = trimesh.sample.sample_surface(cad, N, seed=0)
    P = np.asarray(P) - cad.bounds.mean(0)      # HOI4D poses the bbox CENTER
    Nrm = np.asarray(cad.face_normals)[fidx]
    K = np.load(f"{DS}/camera_params/camera_params/{cam}/intrin.npy")
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    cap = cv2.VideoCapture(f"{DS}/HOI4D_release/{seq}/align_rgb/image.mp4")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Wd, Hd = W // B, H // B
    raw = out + ".raw.mp4"
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), 15, (Wd, Hd))
    t = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        sm = cv2.resize(img, (Wd, Hd))
        jp = f"{DS}/HOI4D_annotations/{seq}/objpose/{t}.json"
        anno = None
        if os.path.exists(jp):
            d = json.load(open(jp))
            if d.get("isEffective", 1) and d.get("dataList"):
                anno = d["dataList"][0]
        if anno is not None:
            r, c = anno["rotation"], anno["center"]
            R = Rotation.from_euler(
                "XYZ", [r["x"], r["y"], r["z"]]).as_matrix()
            X = P @ R.T + np.array([c["x"], c["y"], c["z"]])
            Nc = Nrm @ R.T
            z = X[:, 2]
            okz = z > 0.1
            u = np.clip((X[okz, 0] / z[okz] * fx + cx) / B, 0, Wd - 1).astype(int)
            v = np.clip((X[okz, 1] / z[okz] * fy + cy) / B, 0, Hd - 1).astype(int)
            lam = 0.35 + 0.65 * np.clip(
                -(Nc[okz] @ np.array([0.3, -0.5, -0.81])), 0, 1)
            col = (np.array([80, 220, 80])[None] * lam[:, None]).astype(np.uint8)
            order = np.argsort(z[okz])[::-1]
            lay = np.zeros((Hd, Wd, 3), np.uint8)
            cov = np.zeros((Hd, Wd), bool)
            lay[v[order], u[order]] = col[order]
            cov[v, u] = True
            cov = cv2.morphologyEx(cov.astype(np.uint8), cv2.MORPH_CLOSE,
                                   np.ones((3, 3), np.uint8)) > 0
            hole = cov & (lay.sum(2) == 0)
            lay[hole] = cv2.blur(lay, (3, 3))[hole]
            sm[cov] = (0.42 * sm[cov] + 0.58 * lay[cov]).astype(np.uint8)
        else:
            cv2.putText(sm, "no annotation", (12, 60), FONT, 0.7,
                        (0, 0, 255), 2, cv2.LINE_AA)
        label = f"{seq}  GT CAD overlay"
        cv2.putText(sm, label, (12, 30), FONT, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(sm, label, (12, 30), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(sm, f"frame {t}", (12, Hd - 14), FONT, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(sm)
        t += 1
    vw.release()
    os.system(f"ffmpeg -y -loglevel error -i {raw} -c:v libx264 -crf 21 "
              f"-pix_fmt yuv420p {out} && rm {raw}")
    print(f"wrote {out} ({t} frames)")


if __name__ == "__main__":
    main()
