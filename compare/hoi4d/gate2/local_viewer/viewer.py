"""Standalone local viewer for VGGT-Omega 4D reconstructions of HOI4D clips.

Self-contained: needs only `numpy` and `viser` (pip install -r requirements.txt).
All data (VGGT-Omega depth/cameras/RGB, hand+object masks, GT sensor depth) is
embedded in data/<clip>.npz — no dataset or GPU required.

Run:
    python viewer.py                 # serves the first clip on port 8080
    python viewer.py --clip kettle_N22_S157_T1 --port 8080
Then open http://localhost:8080 — a "clip" dropdown switches clips live.

Scene: per-frame point cloud in VGGT-Omega's world frame (world = frame-0
camera), the camera trajectory (frusta + centres, current frame highlighted
with its RGB image), and an optional GT sensor-depth cloud drawn at a +1.5 m
x-offset. The GT cloud is in its OWN (real camera) coordinate frame — it is a
side-by-side reference, not an aligned overlay.
"""
import argparse, glob, os, time
import numpy as np


# ------------------------------------------------------------------ geometry
def R_to_wxyz(m):
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1) * 2
        return np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
                         (m[1, 0] - m[0, 1]) / s])
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        return np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s,
                         (m[0, 2] + m[2, 0]) / s])
    if m[1, 1] > m[2, 2]:
        s = np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        return np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s,
                         (m[1, 2] + m[2, 1]) / s])
    s = np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
    return np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                     (m[1, 2] + m[2, 1]) / s, 0.25 * s])


def unproject(depth, K, stride, zmax, extra_ok=None):
    """Depth map -> camera-frame points + the pixel-selection mask."""
    H, W = depth.shape
    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    z = depth[::stride, ::stride].astype(np.float32)
    ok = np.isfinite(z) & (z > 0.05) & (z < zmax)
    if extra_ok is not None:
        ok &= extra_ok[::stride, ::stride]
    ys, xs, z = ys[ok], xs[ok], z[ok]
    P = np.stack([(xs - K[0, 2]) / K[0, 0] * z, (ys - K[1, 2]) / K[1, 1] * z, z], 1)
    return P, ok


def to_world(P_cam, E):
    R, t = E[:, :3], E[:, 3]
    return (P_cam - t) @ R


def cam_center_c2w(E):
    R, t = E[:, :3], E[:, 3]
    return -R.T @ t, R.T


# ------------------------------------------------------------------ app
class ClipData:
    def __init__(self, path):
        z = np.load(path)
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.depth = z["depth"].astype(np.float32)
        self.conf = z["conf"].astype(np.float32)
        self.rgb = z["rgb"]
        self.K = z["K"].astype(np.float64)
        self.E = z["E"].astype(np.float64)
        self.hand, self.obj = z["hand"], z["obj"]
        self.gt_depth = z["gt_depth"].astype(np.float32)
        self.K_gt = z["K_gt"].astype(np.float64)
        self.idxs = z["frame_indices"]
        self.S, self.H, self.W = self.depth.shape
        self.conf_lo, self.conf_hi = np.percentile(self.conf, [2, 98])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    import viser

    root = os.path.dirname(os.path.abspath(__file__))
    paths = {os.path.splitext(os.path.basename(p))[0]: p
             for p in sorted(glob.glob(os.path.join(root, "data", "*.npz")))}
    if not paths:
        raise SystemExit("no data/*.npz found next to viewer.py")
    first = args.clip if args.clip in paths else next(iter(paths))

    server = viser.ViserServer(host="127.0.0.1", port=args.port, label="VGGT-Omega 4D")
    g_clip = server.gui.add_dropdown("clip", tuple(paths), initial_value=first)
    with server.gui.add_folder("Playback"):
        g_frame = server.gui.add_slider("frame", 0, 47, 1, 0)
        g_play = server.gui.add_checkbox("play", True)
        g_fps = server.gui.add_slider("fps", 1, 20, 1, 8)
    with server.gui.add_folder("Points"):
        g_stride = server.gui.add_slider("stride", 1, 8, 1, 2)
        g_zmax = server.gui.add_slider("max depth (m)", 0.5, 5.0, 0.1, 2.5)
        g_conf = server.gui.add_slider("min conf (pctl)", 0, 95, 1, 0)
        g_color = server.gui.add_dropdown("color", ("rgb", "hand/object tint", "confidence"),
                                          initial_value="rgb")
        g_size = server.gui.add_slider("point size", 0.001, 0.02, 0.001, 0.004)
    with server.gui.add_folder("Reference"):
        g_cams = server.gui.add_checkbox("camera trajectory", True)
        g_gt = server.gui.add_checkbox("GT cloud (own frame, +1.5m x)", False)
    g_info = server.gui.add_markdown("")

    state = {"clip": None, "data": None, "key": None, "cam_nodes": [],
             "gt_node": None, "cams_visible": True}

    def load_clip(name):
        server.scene.reset()
        d = ClipData(paths[name])
        state.update({"clip": name, "data": d, "key": None, "gt_node": None,
                      "cams_visible": True, "cam_nodes": []})
        g_frame.value = 0
        # slider range fix-up for shorter clips
        try:
            g_frame.max = d.S - 1
        except Exception:
            pass
        nodes = []
        centers = []
        for s in range(d.S):
            c, R_c2w = cam_center_c2w(d.E[s])
            centers.append(c)
            nodes.append(server.scene.add_camera_frustum(
                f"/cams/f{s}", fov=2 * np.arctan(d.H / 2 / d.K[s][1, 1]),
                aspect=d.W / d.H, scale=0.02, wxyz=R_to_wxyz(R_c2w), position=c,
                color=(140, 140, 140)))
        nodes.append(server.scene.add_point_cloud(
            "/cams/centres", np.array(centers, np.float32),
            colors=np.tile([[255, 160, 40]], (len(centers), 1)), point_size=0.005))
        state["cam_nodes"] = nodes

    def refresh():
        d = state["data"]
        s = min(int(g_frame.value), d.S - 1)
        key = (state["clip"], s, int(g_stride.value), round(float(g_zmax.value), 2),
               int(g_conf.value), g_color.value, bool(g_gt.value),
               round(float(g_size.value), 4))
        if key == state["key"]:
            return
        state["key"] = key
        st = int(g_stride.value)
        conf_min = np.percentile(d.conf[s], g_conf.value) if g_conf.value > 0 else -1e9
        Pc, ok = unproject(d.depth[s], d.K[s], st, float(g_zmax.value),
                           d.conf[s] >= conf_min)
        P = to_world(Pc, d.E[s])
        if g_color.value == "rgb":
            C = d.rgb[s][::st, ::st][ok]
        elif g_color.value == "confidence":
            cn = d.conf[s][::st, ::st][ok]
            cn = np.clip((cn - d.conf_lo) / max(d.conf_hi - d.conf_lo, 1e-6), 0, 1)
            C = np.stack([255 * (1 - cn), 255 * cn, np.zeros_like(cn)], 1).astype(np.uint8)
        else:
            C = d.rgb[s][::st, ::st][ok].astype(np.float32)
            gray = C.mean(1, keepdims=True) * 0.55
            C = np.tile(gray, (1, 3))
            hm = d.hand[s][::st, ::st][ok]; om = d.obj[s][::st, ::st][ok]
            C[om] = C[om] * 0.4 + np.array([200, 60, 50]) * 0.6
            C[hm] = C[hm] * 0.4 + np.array([60, 190, 90]) * 0.6
            C = C.clip(0, 255).astype(np.uint8)
        server.scene.add_point_cloud("/cloud", P.astype(np.float32), colors=C,
                                     point_size=float(g_size.value))
        c, R_c2w = cam_center_c2w(d.E[s])
        server.scene.add_camera_frustum(
            "/current_cam", fov=2 * np.arctan(d.H / 2 / d.K[s][1, 1]),
            aspect=d.W / d.H, scale=0.06, wxyz=R_to_wxyz(R_c2w), position=c,
            color=(255, 90, 40), image=d.rgb[s][::4, ::4])
        if g_gt.value:
            Pg, okg = unproject(d.gt_depth[s], d.K_gt, max(2, st), float(g_zmax.value))
            Pg[:, 0] += 1.5
            Cg = d.rgb[s][::max(2, st), ::max(2, st)][okg]
            state["gt_node"] = server.scene.add_point_cloud(
                "/gt_cloud", Pg.astype(np.float32), colors=Cg,
                point_size=float(g_size.value))
        elif state["gt_node"] is not None:
            state["gt_node"].remove()
            state["gt_node"] = None
        if bool(g_cams.value) != state["cams_visible"]:
            state["cams_visible"] = bool(g_cams.value)
            for n in state["cam_nodes"]:
                n.visible = state["cams_visible"]
        g_info.content = (f"**{state['clip']}** — frame {s + 1}/{d.S} "
                          f"(source #{int(d.idxs[s])}), {P.shape[0]:,} pts")

    load_clip(first)
    print(f"open http://localhost:{args.port}   clips: {', '.join(paths)}")
    last = time.time()
    while True:
        if g_clip.value != state["clip"]:
            load_clip(g_clip.value)
        d = state["data"]
        if g_play.value and time.time() - last > 1.0 / float(g_fps.value):
            g_frame.value = (int(g_frame.value) + 1) % d.S
            last = time.time()
        refresh()
        time.sleep(0.015)


if __name__ == "__main__":
    main()
