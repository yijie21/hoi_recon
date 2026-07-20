"""Adapt a HOT3D BOP clip into render_and_compare pipeline input.

HOT3D Aria RGB is fisheye624 (stored rotated 90 deg) and the dataset has NO
depth sensor, so the pipeline's two hard inputs are synthesized here:

  1. RGB video: every frame is rectified onto a virtual UPRIGHT pinhole camera
     (axes = Aria camera rotated about its optical axis so the image needs no
     post-rotation), FOV_DEG horizontal, OUT x OUT px.
  2. GT metric depth: the posed BOP-eval GLBs of ALL annotated objects plus the
     GT UmeTrack hand meshes are ray-cast (open3d) from the same pinhole camera
     -> 16-bit millimetre PNGs, the RC_GT_DEPTH_DIR convention of the
     pipeline's `--depth gt` backend. Background has no return (0 -> NaN),
     like a real close-range depth sensor.

Also writes: intrinsics .npy (RC_GT_INTRINSICS), the target object's GT poses
in the per-frame pinhole camera frame (for later overlay/eval), and the SAM2
prompt pixel (GT centroid of the target in frame 0 — the "user click").

Usage: make_rc_input.py <clip_dir> <target_uid> <out_dir>
       [--start S --end E]   # optional inclusive frame sub-window (default: whole clip)
       [--out_res N]         # optional pinhole output size (default 1024)
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import open3d as o3d
import trimesh
from hand_tracking_toolkit import camera as htt_camera
from hand_tracking_toolkit import dataset as htt_dataset
from hand_tracking_toolkit.hand_models.umetrack_hand_model import (
    forward_kinematics as umetrack_fk)
from scipy.spatial.transform import Rotation

DS = "/workspace/datasets/hot3d"
STREAM = "214-1"
OUT = 1024                    # pinhole output size (square)
FOV_DEG = 90.0                # horizontal FOV of the virtual pinhole
FPS = 30.0


def T_from(d):
    R = Rotation.from_quat(np.roll(d["quaternion_wxyz"], -1)).as_matrix()
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, d["translation_xyz"]
    return T


def load_mesh(uid):
    # eval models: meters, the canonical the poses are defined in
    g = trimesh.load(f"{DS}/object_models_eval/obj_{int(uid):06d}.glb")
    return (trimesh.util.concatenate(list(g.geometry.values()))
            if isinstance(g, trimesh.Scene) else g)


def main():
    global OUT, FOV_DEG
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("target_uid")
    ap.add_argument("out_dir")
    ap.add_argument("--start", type=int, default=None,
                    help="first frame index (inclusive) of a sub-window; default 0")
    ap.add_argument("--end", type=int, default=None,
                    help="last frame index (inclusive) of a sub-window; default last frame")
    ap.add_argument("--out_res", type=int, default=OUT,
                    help="pinhole output size (square); default %(default)s")
    ap.add_argument("--fov", type=float, default=FOV_DEG,
                    help="pinhole horizontal FOV in degrees; default %(default)s")
    a = ap.parse_args()
    OUT, FOV_DEG = a.out_res, a.fov
    clip = a.clip.rstrip("/")
    target_uid = a.target_uid
    out_dir = a.out_dir
    os.makedirs(f"{out_dir}/depth_png", exist_ok=True)
    frames = sorted(glob.glob(f"{clip}/*.objects.json"))
    # optional inclusive frame sub-window. Indices match precompute_segments.py
    # (0-based into the same sorted *.objects.json list) so poses transfer 1:1.
    frame_start = a.start if a.start is not None else 0
    frame_end = a.end if a.end is not None else len(frames) - 1
    frames = frames[frame_start:frame_end + 1]

    f = 0.5 * OUT / np.tan(np.radians(FOV_DEG / 2))
    K = np.array([[f, 0, OUT / 2 - 0.5], [0, f, OUT / 2 - 0.5], [0, 0, 1.0]])
    np.save(f"{out_dir}/intrinsics.npy", K)

    # Pinhole camera = Aria RGB camera rotated about its optical axis so the
    # rectified image is upright (the stored image becomes upright under
    # cv2.ROTATE_90_CLOCKWISE, i.e. upright coords (u,v) = (H-1-y, x); the
    # matching camera-frame change is x_P=-y_F, y_P=x_F, z_P=z_F).
    R_FP = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])  # d_F = R_FP @ d_P

    # rectification maps depend only on the (fixed) fisheye calibration:
    cams0 = json.load(open(frames[0][:-len(".objects.json")] + ".cameras.json"))[STREAM]
    fish = htt_camera.from_json(cams0)
    uu, vv = np.meshgrid(np.arange(OUT), np.arange(OUT))
    d_P = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1],
                    np.ones_like(uu, np.float64)], -1).reshape(-1, 3)
    d_F = d_P @ R_FP.T
    uv = np.asarray(fish.eye_to_window(d_F))
    map_x = uv[:, 0].reshape(OUT, OUT).astype(np.float32)
    map_y = uv[:, 1].reshape(OUT, OUT).astype(np.float32)

    # meshes
    objs = {}
    for uid in json.load(open(frames[0])):
        m = load_mesh(uid)
        objs[uid] = o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor(np.asarray(m.vertices, np.float32)),
            o3d.core.Tensor(np.asarray(m.faces, np.uint32)))
    shp = json.load(open(f"{clip}/__hand_shapes.json__"))
    um_shape = htt_dataset.from_umetrack_hand_model_json(shp["umetrack"])

    # pinhole rays for ray casting (shared across frames; origin set per frame)
    dirs = (d_P / np.linalg.norm(d_P, axis=1, keepdims=True)).astype(np.float32)

    vw = cv2.VideoWriter(f"{out_dir}/rgb.raw.mp4",
                         cv2.VideoWriter_fourcc(*"mp4v"), FPS, (OUT, OUT))
    tgt_poses = np.zeros((len(frames), 4, 4))
    prompt = None
    for fi, fpath in enumerate(frames):
        stem = fpath[:-len(".objects.json")]
        img = cv2.imread(f"{stem}.image_{STREAM}.jpg")
        rect = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
        vw.write(rect)

        cams = json.load(open(f"{stem}.cameras.json"))[STREAM]
        T_wF = T_from(cams["T_world_from_camera"])
        T_wP = T_wF.copy()
        T_wP[:3, :3] = T_wF[:3, :3] @ R_FP
        T_Pw = np.linalg.inv(T_wP)

        scene = o3d.t.geometry.RaycastingScene()
        anno = json.load(open(fpath))
        for uid, mesh in objs.items():
            if uid not in anno:
                continue
            T_Po = T_Pw @ T_from(anno[uid][0]["T_world_from_object"])
            mm = mesh.clone()
            mm.transform(o3d.core.Tensor(T_Po.astype(np.float32)))
            scene.add_triangles(mm)
            if uid == target_uid:
                tgt_poses[fi] = T_Po
                if fi == 0:
                    c = T_Po[:3, 3]
                    p = (K @ (c / c[2]))[:2]
                    prompt = [float(p[0]), float(p[1])]
        for side, pc in htt_dataset.decode_hand_pose(
                json.load(open(f"{stem}.hands.json"))).items():
            if pc.umetrack is None:
                continue
            _, verts, faces = umetrack_fk(pc.umetrack, um_shape,
                                          requires_mesh=True)
            hv = verts.detach().numpy() @ T_Pw[:3, :3].T + T_Pw[:3, 3]
            scene.add_triangles(o3d.t.geometry.TriangleMesh(
                o3d.core.Tensor(hv.astype(np.float32)),
                o3d.core.Tensor(faces.detach().numpy().astype(np.uint32))))

        rays = np.concatenate([np.zeros_like(dirs), dirs], 1)
        hit = scene.cast_rays(o3d.core.Tensor(rays))
        t_hit = hit["t_hit"].numpy().reshape(OUT, OUT)
        z = t_hit * dirs[:, 2].reshape(OUT, OUT)     # range -> z-depth
        z[~np.isfinite(z)] = 0
        cv2.imwrite(f"{out_dir}/depth_png/{fi:05d}.png",
                    np.clip(z * 1000, 0, 65535).astype(np.uint16))
        if fi % 30 == 0:
            print(f"frame {fi}/{len(frames)}", flush=True)
    vw.release()
    os.system(f"ffmpeg -y -loglevel error -i {out_dir}/rgb.raw.mp4 -c:v libx264 "
              f"-crf 16 -pix_fmt yuv420p {out_dir}/rgb.mp4 && rm {out_dir}/rgb.raw.mp4")

    np.savez(f"{out_dir}/gt_target.npz", poses=tgt_poses, K=K,
             uid=int(target_uid))
    json.dump({"clip": clip, "target_uid": target_uid, "prompt": prompt,
               "fov_deg": FOV_DEG, "out": OUT,
               "frame_start": frame_start, "frame_end": frame_end},
              open(f"{out_dir}/meta.json", "w"), indent=1)
    print("prompt(px):", prompt)

    # debug frame 0: rectified RGB + GT depth-valid region tinted
    img = cv2.imread(f"{out_dir}/depth_png/00000.png", cv2.IMREAD_UNCHANGED)
    cap = cv2.VideoCapture(f"{out_dir}/rgb.mp4")
    _, rgb0 = cap.read()
    cap.release()
    vis = rgb0.copy()
    m = img > 0
    vis[m] = (0.5 * vis[m] + 0.5 * np.array([80, 220, 80])).astype(np.uint8)
    if prompt:
        cv2.circle(vis, (int(prompt[0]), int(prompt[1])), 8, (0, 0, 255), -1)
    cv2.imwrite(f"{out_dir}/debug_frame0.jpg", vis)
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
