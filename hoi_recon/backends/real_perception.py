"""Concrete real-backend implementations (GPU).

Wired and verified backends live here; stage code calls into these. Each loader is
lazy + cached so models are built once per process. Heavy third-party imports stay
inside the functions so `mock` mode never imports them.

Currently implemented:
  * MoGe-2 geometry  -> metric depth + camera intrinsics (stage0)        [VERIFIED]
  * SAM 2 video seg  -> object masks (stage1)                            [VERIFIED]
HaMeR (stage2) and SAM-3D-Objects (stage3) are wired in their stage modules.
"""
from __future__ import annotations

import os

import numpy as np

from . import BackendNotAvailable, require_ckpt

_CACHE = {}

# depth backend -> (relative checkpoint path, moge module version)
DEPTH_MODELS = {
    "moge": ("moge/moge-2-vitl-normal/model.pt", "v2"),
}


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def list_frames(frames_dir):
    import glob
    return sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))


def _load_moge(cfg):
    key = cfg.backend.depth
    if key in _CACHE:
        return _CACHE[key]
    if key not in DEPTH_MODELS:
        raise BackendNotAvailable(f"unknown depth backend '{key}'")
    rel, ver = DEPTH_MODELS[key]
    ckpt = require_ckpt(cfg.paths.checkpoints, rel,
                        f"hf download Ruicheng/moge-2-vitl-normal --local-dir "
                        f"{cfg.paths.checkpoints}/moge/moge-2-vitl-normal")
    import torch
    if ver == "v2":
        from moge.model.v2 import MoGeModel
    else:
        from moge.model.v1 import MoGeModel
    model = MoGeModel.from_pretrained(ckpt).to(_device()).eval()
    _CACHE[key] = model
    return model


def run_stage0_geometry(cfg, frame_paths, out_dir):
    """MoGe-2 over frames -> metric depth maps + camera intrinsics.

    Returns dict(intrinsics[3,3] px, extrinsics[T,4,4], depth_dir, depth_paths,
    image_size(H,W)). Camera extrinsics fall back to identity (static-camera
    assumption) unless a SLAM backend (VIPE) is wired — logged by the caller.
    """
    import torch
    import cv2
    model = _load_moge(cfg)
    dev = _device()
    depth_dir = os.path.join(out_dir, "depth")
    os.makedirs(depth_dir, exist_ok=True)

    Ks, depth_paths, HW = [], [], None
    for i, p in enumerate(frame_paths):
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        HW = (H, W)
        t = torch.tensor(img / 255.0, dtype=torch.float32, device=dev).permute(2, 0, 1)
        with torch.no_grad():
            out = model.infer(t)
        depth = out["depth"].float().cpu().numpy()
        mask = out["mask"].cpu().numpy().astype(bool) if "mask" in out else np.isfinite(depth)
        depth = np.where(mask, depth, 0.0).astype(np.float16)
        dp = os.path.join(depth_dir, f"{i:05d}.npy")
        np.save(dp, depth)
        depth_paths.append(dp)
        Kn = out["intrinsics"].cpu().numpy()        # normalized
        K = Kn.copy()
        K[0] *= W
        K[1] *= H
        Ks.append(K)

    intrinsics = np.median(np.stack(Ks, 0), axis=0)  # robust across frames
    T = len(frame_paths)
    extrinsics = np.tile(np.eye(4), (T, 1, 1))       # identity (static-cam fallback)
    return {"intrinsics": intrinsics, "extrinsics": extrinsics,
            "depth_dir": depth_dir, "depth_paths": depth_paths, "image_size": HW}


# ==========================================================================
# Stage 1 — hand detection (YOLO) + object segmentation (SAM 2)
# ==========================================================================
def detect_hands(cfg, frame_paths):
    """WiLoR's YOLO hand detector -> per-frame boxes. Returns
    (boxes[T,2,4] xyxy, valid[T,2]); slot 0=left, 1=right."""
    from ultralytics import YOLO
    ckpt = require_ckpt(cfg.paths.checkpoints, "wilor/detector.pt",
                        "hf download rolpotamias/WiLoR --local-dir "
                        f"{cfg.paths.checkpoints}/wilor")
    model = YOLO(ckpt)
    T = len(frame_paths)
    boxes = np.full((T, 2, 4), np.nan)
    valid = np.zeros((T, 2), bool)
    conf = np.zeros((T, 2))
    for i, p in enumerate(frame_paths):
        res = model(p, conf=0.3, verbose=False)[0]
        for b in res.boxes:
            slot = int(b.cls.item()) % 2          # YOLO class -> hand side slot
            c = float(b.conf.item())
            if c > conf[i, slot]:
                boxes[i, slot] = b.xyxy[0].cpu().numpy()
                valid[i, slot] = True
                conf[i, slot] = c
    return boxes, valid


def _load_sam2(cfg):
    if "sam2" in _CACHE:
        return _CACHE["sam2"]
    ckpt = require_ckpt(cfg.paths.checkpoints,
                        "sam2/sam2.1-hiera-large/sam2.1_hiera_large.pt",
                        "hf download facebook/sam2.1-hiera-large --local-dir "
                        f"{cfg.paths.checkpoints}/sam2/sam2.1-hiera-large")
    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_l.yaml", ckpt, device=_device())
    _CACHE["sam2"] = predictor
    return predictor


def _object_prompt(boxes, valid, image_size):
    """Heuristic SAM2 prompt for the interacting object: centre of the detected
    hand box (the held object sits in the grasp), else image centre. Replace with
    a real interacting-object detector or a user click for production."""
    H, W = image_size
    v = valid[0] if valid.shape[0] else np.zeros(2, bool)
    if v.any():
        b = boxes[0][v][0]
        return float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2)
    return W / 2.0, H / 2.0


def segment_object(cfg, frames_dir, frame_paths, prompt_xy, out_dir):
    """SAM 2 video segmentation from a single point prompt on frame 0.
    Returns (masks_dir, mask_paths)."""
    import torch
    predictor = _load_sam2(cfg)
    masks_dir = os.path.join(out_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    T = len(frame_paths)
    mask_paths = [None] * T
    with torch.inference_mode(), torch.autocast(_device(), dtype=torch.bfloat16):
        state = predictor.init_state(video_path=frames_dir)
        predictor.add_new_points_or_box(
            state, frame_idx=0, obj_id=1,
            points=np.array([prompt_xy], np.float32),
            labels=np.array([1], np.int32))
        for fidx, _ids, logits in predictor.propagate_in_video(state):
            m = (logits[0] > 0).squeeze().cpu().numpy().astype(bool)
            mp = os.path.join(masks_dir, f"{fidx:05d}.npy")
            np.save(mp, m)
            mask_paths[fidx] = mp
    return masks_dir, mask_paths


# ==========================================================================
# Stage 3 — model-free object: SAM mask + MoGe depth -> mesh + 6D trajectory
# ==========================================================================
def _backproject(depth, mask, K):
    sel = mask & (depth > 0)
    ys, xs = np.where(sel)
    z = depth[ys, xs]
    x = (xs - K[0, 2]) / K[0, 0] * z
    y = (ys - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], 1)


def _interp_nan_rows(a):
    x = np.arange(a.shape[0])
    out = a.copy()
    for c in range(a.shape[1]):
        col = a[:, c]
        ok = ~np.isnan(col)
        out[:, c] = np.interp(x, x[ok], col[ok]) if ok.any() else 0.0
    return out


def run_object_depthlift(cfg, frame_paths, mask_paths, depth_paths, K, max_pts=1500):
    """Lift the masked object to 3D per frame, build a canonical convex-hull mesh
    from the best frame, and track translation across the clip (R=I).

    Returns dict(verts[No,3], faces[Mo,3], poses[T,4,4], radius). This is the
    working model-free object branch; SAM-3D-Objects can replace it for sharper,
    rotation-aware geometry.
    """
    import trimesh
    T = len(frame_paths)
    rng = np.random.default_rng(0)
    clouds, centroids = [], np.full((T, 3), np.nan)
    for i in range(T):
        if mask_paths[i] is None:
            clouds.append(None); continue
        m = np.load(mask_paths[i])
        d = np.load(depth_paths[i]).astype(np.float32)
        pc = _backproject(d, m, K)
        if len(pc) < 50:
            clouds.append(None); continue
        zmed = np.median(pc[:, 2])                       # drop depth outliers
        pc = pc[np.abs(pc[:, 2] - zmed) < 0.5 * zmed + 0.1]
        clouds.append(pc)
        centroids[i] = pc.mean(0)
    sizes = [0 if c is None else len(c) for c in clouds]
    if max(sizes) < 50:
        raise BackendNotAvailable("object segmentation produced no usable depth "
                                  "points; check the SAM2 object prompt / masks")
    a = int(np.argmax(sizes))
    pa = clouds[a]
    if len(pa) > max_pts:
        pa = pa[rng.choice(len(pa), max_pts, replace=False)]
    ca = pa.mean(0)
    canon = pa - ca
    hull = trimesh.convex.convex_hull(canon)
    verts = np.asarray(hull.vertices, np.float64)
    faces = np.asarray(hull.faces, np.int64)
    radius = float(np.linalg.norm(canon, axis=1).mean())
    centroids = _interp_nan_rows(centroids)
    poses = np.tile(np.eye(4), (T, 1, 1))
    poses[:, :3, 3] = centroids
    return {"verts": verts, "faces": faces, "poses": poses,
            "radius": np.array(radius), "anchor_frame": a}


# ==========================================================================
# Stage 2 — MANO-free hand fallback: lift the hand region to a corresponded
# point cloud via MoGe depth. Lets the full pipeline run without MANO/license.
# ==========================================================================
def run_hand_depthlift(cfg, frame_paths, hand_boxes, hand_valid, depth_paths, K,
                       grid=28, depth_band=0.12):
    """Lift the detected hand box to a fixed GxG grid of 3D points each frame, so
    vertices are temporally corresponded (grid cell = same box-relative location).
    Returns the stage-2 contract (verts[T,G*G,3], joints[T,21,3], contact_idx, ...).
    Coarse vs MANO, but unblocks an end-to-end real run."""
    import cv2
    T = len(frame_paths)
    G = grid
    Nh = G * G
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    gj, gi = np.meshgrid((np.arange(G) + 0.5) / G, (np.arange(G) + 0.5) / G)
    verts = np.zeros((T, Nh, 3))
    joints = np.zeros((T, 21, 3))
    # fixed 21-joint and contact subsets on the grid (lower-central = fingers)
    jsel = np.linspace(0, Nh - 1, 21).astype(int)
    rows = (np.arange(Nh) // G) / G
    cols = (np.arange(Nh) % G) / G
    contact_idx = np.where((rows > 0.35) & (cols > 0.15) & (cols < 0.85))[0]

    prev = None
    for i in range(T):
        slot = 1 if hand_valid[i, 1] else (0 if hand_valid[i, 0] else None)
        if slot is None:
            verts[i] = prev if prev is not None else 0.0
            joints[i] = verts[i][jsel]
            continue
        H, W = np.load(depth_paths[i]).shape
        x0, y0, x1, y1 = hand_boxes[i, slot]
        x0, x1 = np.clip([x0, x1], 0, W - 1)
        y0, y1 = np.clip([y0, y1], 0, H - 1)
        d = np.load(depth_paths[i]).astype(np.float32)
        crop = d[int(y0):int(y1) + 1, int(x0):int(x1) + 1]
        if crop.size < 9:
            verts[i] = prev if prev is not None else 0.0
            joints[i] = verts[i][jsel]; continue
        dg = cv2.resize(crop, (G, G), interpolation=cv2.INTER_NEAREST).reshape(-1)
        valid = dg > 0
        zmed = np.median(dg[valid]) if valid.any() else 1.0
        dg = np.where(valid & (np.abs(dg - zmed) < depth_band), dg, zmed)  # gate to hand
        u = x0 + gj.reshape(-1) * (x1 - x0)
        v = y0 + gi.reshape(-1) * (y1 - y0)
        X = (u - cx) / fx * dg
        Y = (v - cy) / fy * dg
        pc = np.stack([X, Y, dg], 1)
        verts[i] = pc
        joints[i] = pc[jsel]
        prev = pc
    return {"betas": np.zeros(10), "orient": np.zeros((T, 3)), "pose": np.zeros((T, 45)),
            "transl": verts[:, :, :].mean(1), "joints": joints, "verts": verts,
            "contact_idx": contact_idx, "hand_faces": None}


# ==========================================================================
# Stage 2 — hand reconstruction (HaMeR -> MANO).  Requires MANO (license-gated).
# ==========================================================================
def _cam_crop_to_full(cam_bbox, box_center, box_size, img_size, focal):
    """HaMeR weak-perspective crop cam -> full-image translation (numpy)."""
    img_w, img_h = float(img_size[0]), float(img_size[1])
    cx, cy = box_center
    bs = box_size * cam_bbox[0] + 1e-9
    tz = 2.0 * focal / bs
    tx = (2.0 * (cx - img_w / 2.0) / bs) + cam_bbox[1]
    ty = (2.0 * (cy - img_h / 2.0) / bs) + cam_bbox[2]
    return np.array([tx, ty, tz], np.float64)


def _mano_fingertip_idx(verts, tips=(744, 320, 443, 554, 672), rad=0.025):
    """Contact candidate vertices = MANO surface within `rad` of the 5 fingertips."""
    tip_pts = verts[list(tips)]
    d = np.min(np.linalg.norm(verts[:, None, :] - tip_pts[None], axis=2), axis=1)
    idx = np.where(d < rad)[0]
    return idx if len(idx) >= 20 else np.argsort(d)[:120]


def run_hand(cfg, frame_paths, hand_boxes, hand_valid):
    """HaMeR per-frame MANO reconstruction using stage-1 hand boxes (no detectron2).

    Returns dict(betas, orient, pose, transl, joints[T,21,3], verts[T,778,3],
    contact_idx, hand_faces). Hard-gated on the MANO model (license).
    """
    import sys
    import torch
    import cv2

    # MANO (the license-gated blocker) is checked first — it's the action you own.
    mano_dir = os.path.join(cfg.paths.checkpoints, "mano")
    if not os.path.exists(os.path.join(mano_dir, "MANO_RIGHT.pkl")):
        raise BackendNotAvailable(
            "MANO model required for HaMeR but missing. It is LICENSE-GATED and "
            "cannot be auto-downloaded: register at https://mano.is.tue.mpg.de, "
            f"accept the license, then place MANO_RIGHT.pkl at {mano_dir}/MANO_RIGHT.pkl\n"
            "  (MANO-free alternative that runs now: --hand depthlift)")
    ckpt = require_ckpt(cfg.paths.checkpoints,
                        "hamer/hamer_ckpts/checkpoints/hamer.ckpt",
                        "download HaMeR weights (see scripts/download_checkpoints.sh)")

    repo = os.path.abspath(os.path.join(cfg.paths.third_party, "hamer"))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        from hamer.configs import get_config
        from hamer.models import HAMER
        from hamer.datasets.vitdet_dataset import ViTDetDataset
    except Exception as e:
        raise BackendNotAvailable(
            f"HaMeR import failed: {e}\n  install deps (no detectron2 needed): "
            "pip install pytorch-lightning smplx==0.1.28 yacs einops timm webdataset; "
            "MANO .pkl loading also needs chumpy (numpy<1.24) — see README real-mode notes.")

    dev = _device()
    cfg_yaml = os.path.join(os.path.dirname(ckpt), os.pardir, "model_config.yaml")
    model_cfg = get_config(cfg_yaml, update_cachedir=True)
    model_cfg.defrost()
    if model_cfg.MODEL.BACKBONE.TYPE == "vit" and "BBOX_SHAPE" not in model_cfg.MODEL:
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
    model_cfg.MANO.MODEL_PATH = mano_dir
    for mean in (os.path.join(cfg.paths.checkpoints, "hamer", "hamer_ckpts", "mano_mean_params.npz"),
                 os.path.join(cfg.paths.checkpoints, "hamer", "data", "mano_mean_params.npz"),
                 os.path.join(cfg.paths.third_party, "WiLoR", "mano_data", "mano_mean_params.npz")):
        if os.path.exists(mean):
            model_cfg.MANO.MEAN_PARAMS = mean
            break
    model_cfg.freeze()

    model = HAMER.load_from_checkpoint(ckpt, strict=False, cfg=model_cfg).to(dev).eval()
    focal, img_sz_model = model_cfg.EXTRA.FOCAL_LENGTH, model_cfg.MODEL.IMAGE_SIZE
    faces = np.asarray(model.mano.faces, np.int64) if hasattr(model.mano, "faces") else None

    T = len(frame_paths)
    verts_all = np.zeros((T, 778, 3))
    joints_all = np.zeros((T, 21, 3))
    ref = None
    for i, p in enumerate(frame_paths):
        img = cv2.imread(p)[:, :, ::-1].copy()                  # BGR->RGB
        slot = 1 if hand_valid[i, 1] else (0 if hand_valid[i, 0] else None)
        if slot is None:
            if i > 0:
                verts_all[i], joints_all[i] = verts_all[i - 1], joints_all[i - 1]
            continue
        box = hand_boxes[i, slot][None, :]
        right = np.array([1 if slot == 1 else 0])
        ds = ViTDetDataset(model_cfg, img, box, right, rescale_factor=2.0)
        batch = torch.utils.data.default_collate([ds[0]])
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            out = model(batch)
        v = out["pred_vertices"][0].cpu().numpy()
        j = out["pred_keypoints_3d"][0].cpu().numpy()
        s = 2.0 * float(right[0]) - 1.0                         # mirror left hands
        v[:, 0] *= s
        j[:, 0] *= s
        pc = out["pred_cam"][0].cpu().numpy()
        bc = batch["box_center"][0].cpu().numpy()
        bs = float(batch["box_size"][0].cpu().numpy())
        isz = batch["img_size"][0].cpu().numpy()
        sf = focal / img_sz_model * float(max(isz))
        t = _cam_crop_to_full(pc, bc, bs, isz, sf)
        verts_all[i], joints_all[i] = v + t, j + t
        ref = i if ref is None else ref

    if ref is None:
        raise BackendNotAvailable("no hands detected in any frame (stage1)")
    contact_idx = _mano_fingertip_idx(verts_all[ref])
    return {"betas": np.zeros(10), "orient": np.zeros((T, 3)), "pose": np.zeros((T, 45)),
            "transl": joints_all[:, 0, :], "joints": joints_all, "verts": verts_all,
            "contact_idx": contact_idx, "hand_faces": faces}
