"""Viser demo: backprojected GT-depth point cloud of the example clip
(kettle_N15, archived run render_and_compare/runs/kettle_gt).

Backprojects every frame's masked-valid sensor depth through the shared
intrinsics into the camera frame and serves an interactive viewer:
frame slider + play, point size, object-mask highlight / isolate, and a
camera frustum showing the RGB frame. This is step-1 evidence for
BEST_STRATEGY.md: the raw depth substrate the ICP registers against.

Run (rc5090 env): python viser_backproj_demo.py [--port 17860]
"""
import argparse
import os
import threading
import time

import cv2
import numpy as np
import viser

RUNS = "/workspace/code/hoi_recon/render_and_compare/runs"
BASE = os.path.join(RUNS, "kettle_gt")
STRIDE = 3          # pixel subsampling (3 -> ~180k pts/frame at 1080p)
Z_MIN, Z_MAX = 0.25, 5.0

# arms whose stage-4 registered object track can be overlaid on the cloud
ARMS = {
    "icpj3 (JOINT depth+silhouette)": "kettle_gt_icpj3",
    "icp4 (scale refit, rot free)": "kettle_gt_icp4",
    "icp5 (rot locked to tracker)": "kettle_gt_icp5",
    "icp2 (no scale refit)": "kettle_gt_icp2",
}
CANON_GLB = os.path.join(RUNS, "kettle_gt_icp2/stage3_object/sam3d/object.glb")


def load_all():
    arr = np.load(os.path.join(BASE, "stage0_preprocess/arrays.npz"))
    K = arr["intrinsics"].astype(np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    depth_dir = os.path.join(BASE, "stage0_preprocess/depth")
    frame_dir = os.path.join(BASE, "stage0_preprocess/frames")
    mask_dir = os.path.join(BASE, "stage1_detect_track/masks")
    n = len([f for f in os.listdir(depth_dir) if f.endswith(".npy")])

    pts, cols, obj, thumbs = [], [], [], []
    for i in range(n):
        g = np.load(os.path.join(depth_dir, f"{i:05d}.npy")).astype(np.float32)
        img = cv2.imread(os.path.join(frame_dir, f"{i:05d}.jpg"))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        m = np.load(os.path.join(mask_dir, f"{i:05d}.npy"))

        gs = g[::STRIDE, ::STRIDE]
        rs = rgb[::STRIDE, ::STRIDE]
        ms = m[::STRIDE, ::STRIDE]
        ys, xs = np.nonzero((gs > Z_MIN) & (gs < Z_MAX))
        z = gs[ys, xs]
        u, v = xs * STRIDE, ys * STRIDE
        P = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], 1)
        pts.append(P.astype(np.float32))
        cols.append(rs[ys, xs])
        obj.append(ms[ys, xs])
        thumbs.append(cv2.resize(rgb, (320, 180)))
        if i % 20 == 0:
            print(f"backprojected {i + 1}/{n}")
    print(f"done: {n} frames, ~{len(pts[0]):,} pts/frame")
    return K, pts, cols, obj, thumbs


def load_arms():
    """Stage-4 registered object per arm: verts already carry the global
    scale; poses are det=1 rigid per frame."""
    arms = {}
    for label, run in ARMS.items():
        a = np.load(os.path.join(RUNS, run, "stage4_align/arrays.npz"))
        c = a["obj_colors"]
        c = (c * 255).astype(np.uint8) if c.dtype.kind == "f" else c
        arms[label] = {"verts": a["obj_verts"].astype(np.float32),
                       "faces": a["obj_faces"].astype(np.int64),
                       "colors": c, "poses": a["obj_poses"].astype(np.float64)}
    return arms


def _wxyz(R):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=17860)
    args = ap.parse_args()

    K, pts, cols, obj, thumbs = load_all()
    T = len(pts)
    H, W = 1080, 1920

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("-y")   # OpenCV camera frame: y down, z fwd

    with server.gui.add_folder("Playback"):
        g_frame = server.gui.add_slider("frame", 0, T - 1, 1, 0)
        g_play = server.gui.add_checkbox("play", False)
        g_fps = server.gui.add_slider("fps", 1, 15, 1, 5)
    with server.gui.add_folder("Display"):
        g_size = server.gui.add_slider("point size (mm)", 1, 15, 1, 4)
        g_hl = server.gui.add_checkbox("highlight object mask", True)
        g_only = server.gui.add_checkbox("object points only", False)

    import trimesh
    arms = load_arms()
    with server.gui.add_folder("SAM-3D mesh"):
        g_reg = server.gui.add_checkbox("registered mesh in cloud", True)
        g_arm = server.gui.add_dropdown("arm", tuple(arms), tuple(arms)[0])
        g_canon = server.gui.add_checkbox("raw canonical mesh (aside)", False)

    obj_frame = server.scene.add_frame("/obj", show_axes=False)
    mesh_handles = {}
    for label, a in arms.items():
        tm = trimesh.Trimesh(a["verts"], a["faces"],
                             vertex_colors=a["colors"], process=False)
        mesh_handles[label] = server.scene.add_mesh_trimesh(
            f"/obj/{ARMS[label]}", tm, visible=False)

    canon_handle = None
    if os.path.exists(CANON_GLB):
        g = trimesh.load(CANON_GLB)
        tm = (trimesh.util.concatenate(list(g.geometry.values()))
              if isinstance(g, trimesh.Scene) else g)
        canon_handle = server.scene.add_mesh_trimesh(
            "/canonical", tm, position=(0.5, 0.0, 1.2), visible=False)

    frustum = server.scene.add_camera_frustum(
        "/camera", fov=2 * np.arctan(H / (2 * K[1, 1])), aspect=W / H,
        scale=0.12, image=thumbs[0])
    cloud = server.scene.add_point_cloud(
        "/cloud", points=pts[0], colors=cols[0],
        point_size=g_size.value / 1000.0, point_shape="circle")

    def redraw(_=None):
        i = int(g_frame.value)
        P, C, M = pts[i], cols[i].copy(), obj[i]
        if g_hl.value:
            C[M] = (255, 60, 40)
        if g_only.value:
            P, C = P[M], C[M]
        cloud.points, cloud.colors = P, C
        frustum.image = thumbs[i]
        for label, h in mesh_handles.items():
            h.visible = g_reg.value and label == g_arm.value
        pose = arms[g_arm.value]["poses"][i]
        obj_frame.position = pose[:3, 3]
        obj_frame.wxyz = _wxyz(pose[:3, :3])
        if canon_handle is not None:
            canon_handle.visible = g_canon.value

    g_frame.on_update(redraw)
    g_hl.on_update(redraw)
    g_only.on_update(redraw)
    g_reg.on_update(redraw)
    g_arm.on_update(redraw)
    g_canon.on_update(redraw)
    redraw()
    g_size.on_update(lambda _: setattr(cloud, "point_size",
                                       g_size.value / 1000.0))

    def player():
        while True:
            if g_play.value:
                g_frame.value = (int(g_frame.value) + 1) % T
            time.sleep(1.0 / g_fps.value)

    threading.Thread(target=player, daemon=True).start()
    print(f"viser ready on {args.host}:{args.port}")
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
