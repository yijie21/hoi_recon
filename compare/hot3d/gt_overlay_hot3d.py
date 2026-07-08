"""HOT3D GT-overlay sanity check (parity with the HOI4D overlays in
compare/hoi4d/gate2/sam3d_icp): splat every annotated object's scanned GLB,
posed by the mocap GT trajectory, over the Aria RGB stream (214-1).

Data: BOP "clips" release (huggingface bop-benchmark/hot3d) extracted to
/workspace/datasets/hot3d/clips/<clip>/ — per frame: image_214-1.jpg,
cameras.json (fisheye624 calibration + T_world_from_camera), objects.json
(uid -> T_world_from_object), hands.json (MANO + amodal boxes, drawn here as
outlines). GLB units are MILLIMETERS. Camera model via Meta's
hand_tracking_toolkit (pip install git+https://github.com/facebookresearch/
hand_tracking_toolkit). Output frames are rotated 90 deg CW for upright
viewing (Aria RGB is stored rotated).

Usage: gt_overlay_hot3d.py <clip_dir> [out.mp4]
"""
import glob
import json
import os
import sys

import cv2
import numpy as np
import trimesh
from hand_tracking_toolkit import camera as htt_camera
from scipy.spatial.transform import Rotation

DS = "/workspace/datasets/hot3d"
STREAM = "214-1"
B = 2
N_PER_OBJ = 60000
PALETTE = [(80, 220, 80), (80, 160, 255), (255, 160, 80), (200, 80, 220),
           (80, 220, 220), (220, 220, 80), (160, 100, 255), (100, 255, 160)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def T_from(d):
    R = Rotation.from_quat(np.roll(d["quaternion_wxyz"], -1)).as_matrix()
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, d["translation_xyz"]
    return T


def load_object(uid):
    g = trimesh.load(f"{DS}/object_models/obj_{int(uid):06d}.glb")
    m = (trimesh.util.concatenate(list(g.geometry.values()))
         if isinstance(g, trimesh.Scene) else g)
    P, fidx = trimesh.sample.sample_surface(m, N_PER_OBJ, seed=0)
    return np.asarray(P) / 1000.0, np.asarray(m.face_normals)[fidx]


def main():
    clip = sys.argv[1].rstrip("/")
    name = os.path.basename(clip)
    out = sys.argv[2] if len(sys.argv) > 2 else f"gt_overlay_hot3d_{name}.mp4"
    frames = sorted(glob.glob(f"{clip}/*.objects.json"))
    names = json.load(open(f"{DS}/object_models_models_info.json"))

    objs = {}
    for uid in json.load(open(frames[0])):
        objs[uid] = load_object(uid)
    print(f"{name}: {len(frames)} frames, objects:",
          {u: names[u]["name"] for u in objs})

    img0 = cv2.imread(frames[0].replace(".objects.json", f".image_{STREAM}.jpg"))
    H, W = img0.shape[:2]
    Wd, Hd = W // B, H // B
    raw = out + ".raw.mp4"
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), 15, (Hd, Wd))

    for fi, fpath in enumerate(frames):
        stem = fpath[:-len(".objects.json")]
        img = cv2.imread(f"{stem}.image_{STREAM}.jpg")
        sm = cv2.resize(img, (Wd, Hd))
        cams = json.load(open(f"{stem}.cameras.json"))[STREAM]
        cam = htt_camera.from_json(cams)
        T_cw = np.linalg.inv(T_from(cams["T_world_from_camera"]))

        pts, cols, zs = [], [], []
        for k, (uid, (P, Nrm)) in enumerate(objs.items()):
            anno = json.load(open(fpath)).get(uid)
            if not anno:
                continue
            T_wo = T_from(anno[0]["T_world_from_object"])
            T_co = T_cw @ T_wo
            X = P @ T_co[:3, :3].T + T_co[:3, 3]
            Nc = Nrm @ T_co[:3, :3].T
            lam = 0.35 + 0.65 * np.clip(-(Nc @ np.array([0.3, -0.5, -0.81])), 0, 1)
            base = np.array(PALETTE[k % len(PALETTE)])
            pts.append(X)
            cols.append((base[None] * lam[:, None]).astype(np.uint8))
            zs.append(X[:, 2])
        if pts:
            X = np.concatenate(pts)
            C = np.concatenate(cols)
            z = np.concatenate(zs)
            # in front of camera AND inside the fisheye's valid solid angle
            ang = np.arccos(np.clip(z / np.linalg.norm(X, axis=1), -1, 1))
            ok = (z > 0.05) & (ang < 1.3)
            uv = np.asarray(cam.eye_to_window(X[ok]))
            inb = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            u = (uv[inb, 0] / B).astype(int)
            v = (uv[inb, 1] / B).astype(int)
            Ci, zi = C[ok][inb], z[ok][inb]
            order = np.argsort(zi)[::-1]
            lay = np.zeros((Hd, Wd, 3), np.uint8)
            cov = np.zeros((Hd, Wd), bool)
            lay[v[order], u[order]] = Ci[order]
            cov[v, u] = True
            cov = cv2.morphologyEx(cov.astype(np.uint8), cv2.MORPH_CLOSE,
                                   np.ones((3, 3), np.uint8)) > 0
            hole = cov & (lay.sum(2) == 0)
            lay[hole] = cv2.blur(lay, (3, 3))[hole]
            sm[cov] = (0.45 * sm[cov] + 0.55 * lay[cov]).astype(np.uint8)

        hands = json.load(open(f"{stem}.hands.json"))
        for side, hd in hands.items():
            box = (hd.get("boxes_amodal") or {}).get(STREAM)
            if box:
                x0, y0, x1, y1 = [int(b / B) for b in box]
                cv2.rectangle(sm, (x0, y0), (x1, y1), (255, 255, 255), 1)

        sm = cv2.rotate(sm, cv2.ROTATE_90_CLOCKWISE)
        label = f"HOT3D {name}  GT objects (mocap) + hand boxes"
        cv2.putText(sm, label, (10, 26), FONT, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(sm, label, (10, 26), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(sm, f"frame {fi}", (10, Wd - 12), FONT, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(sm)
    vw.release()
    os.system(f"ffmpeg -y -loglevel error -i {raw} -c:v libx264 -crf 21 "
              f"-pix_fmt yuv420p {out} && rm {raw}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
