"""Gate-2 extraction — sample per-frame depth from feed-forward 4D models.

For each clip: pick 48 evenly-spaced frames, sample fixed pixel populations
(object / hand eroded masks; static background) at FULL resolution, and record
each source's predicted depth at those pixels alongside GT.

Sources:
  moge : cached MoGe depth from the b6 kill test (half-res, fov-conditioned)
         — the previous-generation baseline, same frames, same pixels.
  d4rt : OpenD4RT 48CLIP (unofficial D4RT), query interface: (u,v,t_src) ->
         xyz at t_tgt in camera t_cam. We query the diagonal t_src=t_tgt=t_cam
         = t (per-frame depth in that frame's own camera). Video resized to
         256x256, queries normalised by ORIGINAL dims — exactly the repo's
         WorldTrack eval convention.
  vggt : VGGT-Omega 1B-512. Dense per-frame depth head; "balanced"
         preprocessing is a pure resize for our aspect (0.5625 in [0.5,2] —
         no crop), so full-res pixel (x,y) maps by pure scaling.

Output per clip+source: <clip>/gate2/<source>_samples.npz with, per region
r in {obj, hand, bg}: r_t (local frame idx), r_zp (predicted), r_zg (GT, m),
plus frame_indices and meta. Evaluation happens separately in gate2_eval.py
(CPU) so metrics can be iterated without re-running models.

Usage:
  python gate2_extract.py --source moge|d4rt|vggt [--clips name1,name2] [--n-frames 48]
Run from any cwd; repo paths are hardcoded for this box.
"""
import argparse, glob, json, os, sys
import numpy as np
import cv2

CLIPS_ROOT = "/workspace/hoi4d/clips"
D4RT_ROOT = "/workspace/code/Open-d4rt"
D4RT_CKPT = f"{D4RT_ROOT}/checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG"
VGGT_ROOT = "/workspace/code/vggt-omega"
VGGT_CKPT = f"{VGGT_ROOT}/weights/vggt_omega_1b_512.pt"

ZMIN, ZMAX, STATIC_TOL = 0.25, 5.0, 0.05
ERODE_FULL, DILATE_FULL = 5, 35          # kill_test full-res morphology
N_OBJ, N_HAND, N_BG = 1500, 1000, 2500
SEED = 0


def frame_indices(T, n):
    return np.unique(np.round(np.linspace(0, T - 1, min(n, T))).astype(int))


def load_clip_fullres(clip, idxs):
    """GT depth + region masks + sampled pixel coords at full res."""
    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ERODE_FULL + 1,) * 2)
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DILATE_FULL + 1,) * 2)
    gts, objs, hands, dyns = [], [], [], []
    for i in idxs:
        g = cv2.imread(os.path.join(clip, "depth", f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)
        gts.append(g.astype(np.float32) / 1000.0)
        hm, om = [cv2.imread(os.path.join(clip, "masks", f"frame_{i:06d}_masks", f"{n}.png"),
                             cv2.IMREAD_GRAYSCALE) for n in ("hand", "object")]
        hm = (hm > 127) if hm is not None else np.zeros(g.shape, bool)
        om = (om > 127) if om is not None else np.zeros(g.shape, bool)
        objs.append(cv2.erode(om.astype(np.uint8), ke) > 0)
        hands.append(cv2.erode(hm.astype(np.uint8), ke) > 0)
        dyns.append(cv2.dilate((hm | om).astype(np.uint8), kd) > 0)
    gt = np.stack(gts)
    valid = (gt > ZMIN) & (gt < ZMAX)
    gt_nan = np.where(valid, gt, np.nan)
    static = np.abs(gt_nan - np.nanmedian(gt_nan, 0)[None]) < STATIC_TOL

    rng = np.random.default_rng(SEED)
    samples = {"obj": [], "hand": [], "bg": []}
    for s, i in enumerate(idxs):
        pops = {"obj": (objs[s] & valid[s], N_OBJ),
                "hand": (hands[s] & valid[s], N_HAND),
                "bg": (valid[s] & static[s] & ~dyns[s], N_BG)}
        for name, (m, cap) in pops.items():
            ys, xs = np.nonzero(m)
            if ys.size == 0:
                continue
            if ys.size > cap:
                pick = rng.choice(ys.size, cap, replace=False)
                ys, xs = ys[pick], xs[pick]
            samples[name].append((np.full(ys.size, s, np.int32), ys.astype(np.int32),
                                  xs.astype(np.int32), gt[s][ys, xs]))
    out = {}
    for name, chunks in samples.items():
        t = np.concatenate([c[0] for c in chunks]); y = np.concatenate([c[1] for c in chunks])
        x = np.concatenate([c[2] for c in chunks]); zg = np.concatenate([c[3] for c in chunks])
        out[name] = (t, y, x, zg)
    H, W = gt.shape[1:]
    return out, (H, W)


# ------------------------------------------------------------------ sources
def extract_moge(clip, idxs, samples, hw):
    d = np.load(os.path.join(clip, "kill_test", "moge_depth.npy")).astype(np.float32)
    Hm, Wm = d.shape[1:]
    H, W = hw
    out = {}
    for name, (t, y, x, zg) in samples.items():
        ym = np.clip((y * Hm) // H, 0, Hm - 1)
        xm = np.clip((x * Wm) // W, 0, Wm - 1)
        zp = d[idxs[t], ym, xm]
        out[name] = zp
    return out, {"res": [int(Hm), int(Wm)]}


def extract_d4rt(model_pack, clip, idxs, samples, hw):
    import torch
    model, chunk = model_pack
    H, W = hw
    frames = []
    for i in idxs:
        img = cv2.imread(os.path.join(clip, "rgb", f"{i:06d}.jpg"))[:, :, ::-1]
        frames.append(cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA))
    video = (torch.from_numpy(np.stack(frames).copy()).float().permute(0, 3, 1, 2)
             .unsqueeze(0).cuda() / 255.0)
    aspect = torch.tensor([[1.0]], dtype=torch.float32, device="cuda")

    sys.path.insert(0, D4RT_ROOT)
    from src.eval.tasks import _encode_model_memory, _run_model_for_queries
    with torch.no_grad():
        memory = _encode_model_memory(model=model, video_b=video, aspect_b=aspect)
        out = {}
        for name, (t, y, x, zg) in samples.items():
            q = {"u": torch.from_numpy(x / max(W - 1, 1)).cuda().float(),
                 "v": torch.from_numpy(y / max(H - 1, 1)).cuda().float(),
                 "t_src": torch.from_numpy(t.astype(np.int64)).cuda(),
                 "t_tgt": torch.from_numpy(t.astype(np.int64)).cuda(),
                 "t_cam": torch.from_numpy(t.astype(np.int64)).cuda()}
            pred = _run_model_for_queries(model=model, video_b=video, aspect_b=aspect,
                                          query=q, chunk_size=chunk, memory_b=memory)
            out[name] = pred["xyz_3d"][:, 2].numpy().astype(np.float32)
    return out, {"res": [256, 256]}


def load_d4rt():
    import torch
    sys.path.insert(0, D4RT_ROOT)
    from src.model import build_model
    from src.core import load_checkpoint, load_yaml_config
    cfg = load_yaml_config(f"{D4RT_CKPT}/model.yaml")
    model = build_model(cfg["model"]).eval()
    payload = load_checkpoint(f"{D4RT_CKPT}/opend4rt.ckpt", map_location="cpu")
    sd = payload
    for key in ("state_dict", "model", "module", "network", "net"):
        if isinstance(payload, dict) and isinstance(payload.get(key), dict):
            sd = payload[key]
            break
    res = model.load_state_dict(sd, strict=False)
    print(f"d4rt loaded: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
    return (model.cuda(), 4096)


def load_vggt():
    import torch
    sys.path.insert(0, VGGT_ROOT)
    from vggt_omega.models import VGGTOmega
    model = VGGTOmega().cuda().eval()
    sd = torch.load(VGGT_CKPT, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    res = model.load_state_dict(sd, strict=False)
    print(f"vggt-omega loaded: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
    return model


def extract_vggt(model, clip, idxs, samples, hw):
    import torch
    sys.path.insert(0, VGGT_ROOT)
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    paths = [os.path.join(clip, "rgb", f"{i:06d}.jpg") for i in idxs]
    images = load_and_preprocess_images(paths, image_resolution=512).cuda()
    with torch.inference_mode():
        pred = model(images)
    depth = pred["depth"]
    d = depth.squeeze().float().cpu().numpy()          # [S,H',W']
    if d.ndim == 4:
        d = d[..., 0]
    S, Hv, Wv = d.shape
    H, W = hw
    out = {}
    for name, (t, y, x, zg) in samples.items():
        yv = np.clip((y * Hv) // H, 0, Hv - 1)
        xv = np.clip((x * Wv) // W, 0, Wv - 1)
        out[name] = d[t, yv, xv].astype(np.float32)
    return out, {"res": [int(Hv), int(Wv)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["moge", "d4rt", "vggt"])
    ap.add_argument("--clips", default="")
    ap.add_argument("--n-frames", type=int, default=48)
    args = ap.parse_args()

    if args.clips:
        clips = [os.path.join(CLIPS_ROOT, c) for c in args.clips.split(",")]
    else:
        clips = sorted(d for d in glob.glob(os.path.join(CLIPS_ROOT, "*"))
                       if os.path.isdir(d))
    model_pack = None
    if args.source == "d4rt":
        model_pack = load_d4rt()
    elif args.source == "vggt":
        model_pack = load_vggt()

    for clip in clips:
        name = os.path.basename(clip)
        n_rgb = len(glob.glob(os.path.join(clip, "rgb", "*.jpg")))
        idxs = frame_indices(n_rgb, args.n_frames)
        samples, hw = load_clip_fullres(clip, idxs)
        if args.source == "moge":
            zp, meta = extract_moge(clip, idxs, samples, hw)
        elif args.source == "d4rt":
            zp, meta = extract_d4rt(model_pack, clip, idxs, samples, hw)
        else:
            zp, meta = extract_vggt(model_pack, clip, idxs, samples, hw)
        outdir = os.path.join(clip, "gate2")
        os.makedirs(outdir, exist_ok=True)
        payload = {"frame_indices": idxs, "meta": json.dumps(meta | {"hw": list(hw)})}
        for r, (t, y, x, zg) in samples.items():
            payload[f"{r}_t"] = t; payload[f"{r}_y"] = y; payload[f"{r}_x"] = x
            payload[f"{r}_zg"] = zg; payload[f"{r}_zp"] = zp[r]
        np.savez_compressed(os.path.join(outdir, f"{args.source}_samples.npz"), **payload)
        print(f"[{args.source}] {name}: frames={len(idxs)} "
              f"obj={samples['obj'][0].size} hand={samples['hand'][0].size} "
              f"bg={samples['bg'][0].size}", flush=True)


if __name__ == "__main__":
    main()
