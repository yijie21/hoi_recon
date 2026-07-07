"""Interactive 4D visualization of VGGT-Omega reconstructions on HOI4D clips.

Two phases (so the viewer never needs the GPU):

  extract : run VGGT-Omega once per clip on the Gate-2 frame set and cache the
            full reconstruction — dense depth, confidence, cameras (world =
            frame-0 camera, OpenCV convention), preprocessed RGB, and
            hand/object masks resized to model resolution —
            to <clip>/gate2/vggt_recon.npz  (~60-90 MB per clip).

  serve   : viser player for one clip: per-frame point cloud in the VGGT world
            frame, camera frusta + trajectory, time slider / autoplay,
            point stride + depth clip + confidence filter, colour modes
            (RGB / hand-object tint / confidence heat), and an optional GT
            reference cloud (HOI4D aligned depth + real intrinsics, static
            camera) drawn at an x-offset — NOTE the GT reference lives in its
            own coordinate frame (real camera), not VGGT's world frame; it is
            a side-by-side reference, not an aligned overlay.

Usage (extract needs the gate2 env + GPU; serve needs viser only):
  CUDA_VISIBLE_DEVICES=0 python visualize_vggt.py extract [--clips a,b] [--n-frames 48]
  python visualize_vggt.py serve --clip kettle_N22_S157_T1 [--port 8090]

View over SSH from your machine:
  ssh -p $VAST_TCP_PORT_22 -L 8090:127.0.0.1:8090 <user>@$PUBLIC_IPADDR
  then open http://localhost:8090
"""
import argparse, glob, json, os, sys, time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate2_extract import CLIPS_ROOT, VGGT_ROOT, frame_indices, load_vggt


# ------------------------------------------------------------------ extract
def extract_clip(model, clip, n_frames):
    import torch
    sys.path.insert(0, VGGT_ROOT)
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    n_rgb = len(glob.glob(os.path.join(clip, "rgb", "*.jpg")))
    idxs = frame_indices(n_rgb, n_frames)
    paths = [os.path.join(clip, "rgb", f"{i:06d}.jpg") for i in idxs]
    images = load_and_preprocess_images(paths, image_resolution=512).cuda()
    with torch.inference_mode():
        pred = model(images)
    H, W = pred["images"].shape[-2:]
    E, K = encoding_to_camera(pred["pose_enc"], (H, W))
    depth = pred["depth"].squeeze().float().cpu().numpy()
    if depth.ndim == 4:
        depth = depth[..., 0]
    conf = pred["depth_conf"].squeeze().float().cpu().numpy()
    rgb = (pred["images"].squeeze().permute(0, 2, 3, 1).float().cpu().numpy()
           * 255).clip(0, 255).astype(np.uint8)

    hands, objs = [], []
    for i in idxs:
        hm, om = [cv2.imread(os.path.join(clip, "masks", f"frame_{i:06d}_masks", f"{n}.png"),
                             cv2.IMREAD_GRAYSCALE) for n in ("hand", "object")]
        for src, dst in ((hm, hands), (om, objs)):
            m = (src > 127).astype(np.uint8) if src is not None else np.zeros((1080, 1920), np.uint8)
            dst.append(cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 0)

    out = os.path.join(clip, "gate2", "vggt_recon.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(
        out,
        frame_indices=idxs,
        depth=depth.astype(np.float16),
        conf=conf.astype(np.float16),
        rgb=rgb,
        K=K.squeeze(0).float().cpu().numpy(),
        E=E.squeeze(0).float().cpu().numpy(),
        hand=np.stack(hands), obj=np.stack(objs))
    print(f"[extract] {os.path.basename(clip)}: {len(idxs)} frames at {H}x{W} -> {out}",
          flush=True)


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


def unproject_frame(depth, K, E, stride, zmax, conf=None, conf_min=None):
    """Depth map -> world-frame points. E = [R|t] world-to-cam (OpenCV)."""
    H, W = depth.shape
    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    z = depth[::stride, ::stride].astype(np.float32)
    ok = np.isfinite(z) & (z > 0.05) & (z < zmax)
    if conf is not None and conf_min is not None:
        ok &= conf[::stride, ::stride].astype(np.float32) >= conf_min
    ys, xs, z = ys[ok], xs[ok], z[ok]
    Xc = np.stack([(xs - K[0, 2]) / K[0, 0] * z, (ys - K[1, 2]) / K[1, 1] * z, z], 1)
    R, t = E[:, :3], E[:, 3]
    return (Xc - t) @ R, ok


def cam_center_c2w(E):
    R, t = E[:, :3], E[:, 3]
    return -R.T @ t, R.T


# ------------------------------------------------------------------ serve
def serve(clip, port):
    import viser
    z = np.load(os.path.join(clip, "gate2", "vggt_recon.npz"))
    depth = z["depth"].astype(np.float32)
    conf = z["conf"].astype(np.float32)
    rgb, K, E = z["rgb"], z["K"].astype(np.float64), z["E"].astype(np.float64)
    hand, obj = z["hand"], z["obj"]
    idxs = z["frame_indices"]
    S, H, W = depth.shape
    conf_lo, conf_hi = np.percentile(conf, [2, 98])
    K_gt = np.load(os.path.join(clip, "intrin.npy")).astype(np.float64)

    server = viser.ViserServer(host="127.0.0.1", port=port, label=os.path.basename(clip))
    with server.gui.add_folder("Playback"):
        g_frame = server.gui.add_slider("frame", 0, S - 1, 1, 0)
        g_play = server.gui.add_checkbox("play", True)
        g_fps = server.gui.add_slider("fps", 1, 15, 1, 6)
    with server.gui.add_folder("Points"):
        g_stride = server.gui.add_slider("stride", 1, 8, 1, 3)
        g_zmax = server.gui.add_slider("max depth (m)", 0.5, 5.0, 0.1, 2.5)
        g_conf = server.gui.add_slider("min confidence", float(conf_lo), float(conf_hi),
                                       float((conf_hi - conf_lo) / 50), float(conf_lo))
        g_color = server.gui.add_dropdown("color", ("rgb", "hand/object tint", "confidence"),
                                          initial_value="rgb")
        g_size = server.gui.add_slider("point size", 0.001, 0.02, 0.001, 0.004)
    with server.gui.add_folder("Reference"):
        g_cams = server.gui.add_checkbox("camera trajectory", True)
        g_gt = server.gui.add_checkbox("GT cloud (own frame, offset)", False)
    g_info = server.gui.add_markdown("")

    # static: all-camera trajectory (thin frusta + centres)
    cam_nodes, centers = [], []
    for s in range(S):
        c, R_c2w = cam_center_c2w(E[s])
        centers.append(c)
        cam_nodes.append(server.scene.add_camera_frustum(
            f"/cams/f{s}", fov=2 * np.arctan(H / 2 / K[s][1, 1]), aspect=W / H,
            scale=0.02, wxyz=R_to_wxyz(R_c2w), position=c,
            color=(140, 140, 140)))
    centers = np.array(centers)
    cam_nodes.append(server.scene.add_point_cloud(
        "/cams/centres", centers,
        colors=np.tile([[255, 160, 40]], (len(centers), 1)), point_size=0.005))

    state = {"key": None, "cams_visible": True, "gt_node": None}

    def gt_cloud(s):
        i = int(idxs[s])
        g = cv2.imread(os.path.join(clip, "depth", f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)
        g = g.astype(np.float32) / 1000.0
        img = cv2.imread(os.path.join(clip, "rgb", f"{i:06d}.jpg"))[:, :, ::-1]
        st = max(2, int(g_stride.value) * 2)
        ys, xs = np.mgrid[0:g.shape[0]:st, 0:g.shape[1]:st]
        zz = g[::st, ::st]
        ok = (zz > 0.25) & (zz < g_zmax.value)
        ys, xs, zz = ys[ok], xs[ok], zz[ok]
        P = np.stack([(xs - K_gt[0, 2]) / K_gt[0, 0] * zz,
                      (ys - K_gt[1, 2]) / K_gt[1, 1] * zz, zz], 1)
        P[:, 0] += 1.5                     # side-by-side offset; own coordinate frame
        return P, img[::st, ::st][ok]

    def refresh():
        s = int(g_frame.value)
        key = (s, int(g_stride.value), round(float(g_zmax.value), 2),
               round(float(g_conf.value), 3), g_color.value, bool(g_gt.value),
               round(float(g_size.value), 4))
        if key == state["key"]:
            return
        state["key"] = key
        st = int(g_stride.value)
        P, ok = unproject_frame(depth[s], K[s], E[s], st, float(g_zmax.value),
                                conf[s], float(g_conf.value))
        if g_color.value == "rgb":
            C = rgb[s][::st, ::st][ok]
        elif g_color.value == "confidence":
            cn = conf[s][::st, ::st][ok]
            cn = np.clip((cn - conf_lo) / max(conf_hi - conf_lo, 1e-6), 0, 1)
            C = np.stack([255 * (1 - cn), 255 * cn, np.zeros_like(cn)], 1).astype(np.uint8)
        else:
            C = rgb[s][::st, ::st][ok].astype(np.float32)
            gray = C.mean(1, keepdims=True) * 0.55
            C = np.tile(gray, (1, 3))
            hm = hand[s][::st, ::st][ok]; om = obj[s][::st, ::st][ok]
            C[om] = C[om] * 0.4 + np.array([200, 60, 50]) * 0.6
            C[hm] = C[hm] * 0.4 + np.array([60, 190, 90]) * 0.6
            C = C.clip(0, 255).astype(np.uint8)
        server.scene.add_point_cloud("/cloud", P.astype(np.float32), colors=C,
                                     point_size=float(g_size.value))
        c, R_c2w = cam_center_c2w(E[s])
        server.scene.add_camera_frustum(
            "/current_cam", fov=2 * np.arctan(H / 2 / K[s][1, 1]), aspect=W / H,
            scale=0.06, wxyz=R_to_wxyz(R_c2w), position=c, color=(255, 90, 40),
            image=rgb[s][::4, ::4])
        if g_gt.value:
            Pg, Cg = gt_cloud(s)
            state["gt_node"] = server.scene.add_point_cloud(
                "/gt_cloud", Pg.astype(np.float32), colors=Cg,
                point_size=float(g_size.value))
        elif state["gt_node"] is not None:
            state["gt_node"].remove()
            state["gt_node"] = None
        if bool(g_cams.value) != state["cams_visible"]:
            state["cams_visible"] = bool(g_cams.value)
            for n in cam_nodes:
                n.visible = state["cams_visible"]
        g_info.content = (f"**{os.path.basename(clip)}** — frame {s + 1}/{S} "
                          f"(source #{int(idxs[s])}), {P.shape[0]:,} pts, "
                          f"depth res {H}x{W}")

    print(f"viser on http://127.0.0.1:{port}  (clip {os.path.basename(clip)}, {S} frames)")
    last = time.time()
    while True:
        if g_play.value and time.time() - last > 1.0 / float(g_fps.value):
            g_frame.value = (int(g_frame.value) + 1) % S
            last = time.time()
        refresh()
        time.sleep(0.02)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["extract", "serve"])
    ap.add_argument("--clips", default="")
    ap.add_argument("--clip", default="")
    ap.add_argument("--n-frames", type=int, default=48)
    ap.add_argument("--port", type=int, default=8090)
    args = ap.parse_args()

    if args.mode == "extract":
        clips = ([os.path.join(CLIPS_ROOT, c) for c in args.clips.split(",") if c]
                 or sorted(d for d in glob.glob(os.path.join(CLIPS_ROOT, "*"))
                           if os.path.isdir(d)))
        model = load_vggt()
        for c in clips:
            extract_clip(model, c, args.n_frames)
    else:
        if not args.clip:
            done = [os.path.basename(os.path.dirname(os.path.dirname(p))) for p in
                    glob.glob(os.path.join(CLIPS_ROOT, "*", "gate2", "vggt_recon.npz"))]
            print("extracted clips:", ", ".join(sorted(done)) or "(none)")
            return
        serve(os.path.join(CLIPS_ROOT, args.clip), args.port)


if __name__ == "__main__":
    main()
