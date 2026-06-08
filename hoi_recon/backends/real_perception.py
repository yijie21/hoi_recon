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

from . import BackendNotAvailable, require_ckpt, require_repo

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
    """Stage-0 geometry: per-frame metric depth + camera intrinsics (+ extrinsics).

    Dispatches on cfg.backend.depth:
      * 'moge'                 -> MoGe-2 depth + intrinsics; identity extrinsics
      * 'da3'/'depth_anything_3' -> Depth-Anything-3: metric depth + intrinsics
                                  + real camera poses (extrinsics), one joint pass

    Returns dict(intrinsics[3,3] px, extrinsics[T,4,4] world->cam, depth_dir,
    depth_paths, image_size(H,W), camera_source).
    """
    if cfg.backend.depth == "vggt":
        return _vggt_geometry(cfg, frame_paths, out_dir)
    if cfg.backend.depth in ("da3", "depth_anything_3"):
        return _da3_geometry(cfg, frame_paths, out_dir)
    return _moge_geometry(cfg, frame_paths, out_dir)


def _save_depth(depth_dir, i, depth):
    os.makedirs(depth_dir, exist_ok=True)
    dp = os.path.join(depth_dir, f"{i:05d}.npy")
    np.save(dp, depth.astype(np.float16))
    return dp


def _da3_geometry(cfg, frame_paths, out_dir):
    """Depth-Anything-3 — metric depth + intrinsics + camera extrinsics in one
    multi-view pass (ViPE's camera-pose capability is merged into DA3). Depth is
    resized to the original frame resolution and intrinsics scaled to match, so it
    stays aligned with the (original-resolution) SAM2 masks used downstream."""
    import torch
    import cv2
    require_repo(cfg.paths.third_party, "Depth-Anything-3",
                 "git clone https://github.com/ByteDance-Seed/Depth-Anything-3")
    try:
        from depth_anything_3.api import DepthAnything3
    except Exception as e:
        raise BackendNotAvailable(
            f"Depth-Anything-3 not importable ({e}). Install it: "
            "cd third_party/Depth-Anything-3 && pip install -e .")
    name = cfg.backend.get("da3_model", "da3metric-large") \
        if hasattr(cfg.backend, "get") else "da3metric-large"
    model = DepthAnything3(model_name=name).to(_device()).eval()
    with torch.no_grad():
        pred = model.inference(list(frame_paths), process_res=504)
    depth = np.asarray(pred.depth, np.float32)          # (N,hp,wp) metric
    intr = np.asarray(pred.intrinsics, np.float64)      # (N,3,3) at processed res
    ext34 = np.asarray(pred.extrinsics, np.float64)     # (N,3,4) world->cam
    N, hp, wp = depth.shape
    H, W = cv2.imread(frame_paths[0]).shape[:2]         # original frame size

    depth_dir = os.path.join(out_dir, "depth")
    depth_paths = []
    for i in range(N):
        d = cv2.resize(depth[i], (W, H), interpolation=cv2.INTER_NEAREST)
        depth_paths.append(_save_depth(depth_dir, i, d))
    K = np.median(intr, axis=0).copy()
    K[0] *= W / wp                                      # rescale to original res
    K[1] *= H / hp
    extr = np.tile(np.eye(4), (N, 1, 1))
    extr[:, :3, :4] = ext34
    return {"intrinsics": K, "extrinsics": extr, "depth_dir": depth_dir,
            "depth_paths": depth_paths, "image_size": (H, W), "camera_source": "da3"}


def _vggt_geometry(cfg, frame_paths, out_dir):
    """VGGT consistent geometry: ONE globally-consistent camera trajectory +
    temporally-consistent depth for the whole clip (vs per-frame monocular MoGe +
    static-camera assumption). Runs VGGT in the sam3d-objects env (subprocess; its
    numpy<2/torch pins match), maps the padded-518 depth/intrinsics back to the
    original frame resolution, and returns real per-frame extrinsics.

    NOTE: VGGT is up-to-scale (monocular). Depth is returned in VGGT units; the
    metric scale is resolved downstream (the render-and-compare optimizer fits it
    from the MANO hand). This is the geometry foundation of the redesigned pipeline.
    """
    import subprocess
    import cv2
    require_repo(cfg.paths.third_party, "vggt",
                 "git clone https://github.com/facebookresearch/vggt")
    from ..logging_utils import log
    geo_npz = os.path.join(out_dir, "vggt", "geo.npz")
    os.makedirs(os.path.dirname(geo_npz), exist_ok=True)
    if not os.path.exists(geo_npz):
        repo = os.path.abspath(os.path.join(cfg.paths.third_party, "vggt"))
        env_name = (cfg.backend.get("sam3d_env", "sam3d-objects")
                    if hasattr(cfg.backend, "get") else "sam3d-objects")
        conda = os.environ.get("CONDA_EXE", "conda")
        ckpt = os.path.join(cfg.paths.checkpoints, "vggt", "model.pt")
        frames_dir = os.path.dirname(os.path.abspath(frame_paths[0]))
        cmd = [conda, "run", "--no-capture-output", "-n", env_name, "python",
               os.path.join(repo, "vggt_geom.py"), "--frames_dir", frames_dir,
               "--mode", "pad", "--out", os.path.abspath(geo_npz)]
        if os.path.exists(ckpt):
            cmd += ["--ckpt", ckpt]
        log(f"camera/depth: running VGGT (env '{env_name}') for consistent geometry...")
        r = subprocess.run(cmd, cwd=repo)
        if r.returncode != 0 or not os.path.exists(geo_npz):
            raise BackendNotAvailable(f"VGGT subprocess failed (exit {r.returncode}).")
    else:
        log(f"camera/depth: reusing cached VGGT geometry {geo_npz}")

    g = np.load(geo_npz)
    extr34 = g["extrinsic"]                              # (N,3,4) world->cam (OpenCV)
    intr = g["intrinsic"]                                # (N,3,3) padded-518
    dproc = g["depth"].astype(np.float32)               # (N,518,518)
    x0, y0, nw, nh = [int(v) for v in g["content_rect"]]
    oh, ow = [int(v) for v in g["orig_hw"]]
    sel = g["sel_idx"]
    sx, sy = nw / ow, nh / oh

    # intrinsics: padded-518 -> original frame (median over frames)
    K = np.eye(3)
    K[0, 0] = np.median(intr[:, 0, 0]) / sx
    K[1, 1] = np.median(intr[:, 1, 1]) / sy
    K[0, 2] = np.median((intr[:, 0, 2] - x0) / sx)
    K[1, 2] = np.median((intr[:, 1, 2] - y0) / sy)

    T = len(frame_paths)
    depth_dir = os.path.join(out_dir, "depth")
    os.makedirs(depth_dir, exist_ok=True)
    # map each VGGT frame's content depth back to the original resolution
    depth_paths = [None] * T
    sel_list = list(sel)
    for k, fi in enumerate(sel_list):
        d = dproc[k][y0:y0 + nh, x0:x0 + nw]            # content region (nh,nw)
        d = cv2.resize(d, (ow, oh), interpolation=cv2.INTER_NEAREST)
        depth_paths[int(fi)] = _save_depth(depth_dir, int(fi), d)
    # if subsampled, fill gaps by nearest selected frame
    if len(sel_list) < T:
        last = None
        for i in range(T):
            if depth_paths[i] is not None:
                last = depth_paths[i]
            elif last is not None:
                depth_paths[i] = last

    extr = np.tile(np.eye(4), (T, 1, 1))
    for k, fi in enumerate(sel_list):
        extr[int(fi), :3, :4] = extr34[k]
    return {"intrinsics": K, "extrinsics": extr, "depth_dir": depth_dir,
            "depth_paths": depth_paths, "image_size": (oh, ow),
            "camera_source": "vggt"}


def _moge_geometry(cfg, frame_paths, out_dir):
    """MoGe-2 over frames -> metric depth + intrinsics; identity camera extrinsics
    (static-camera assumption; use --depth da3 for real camera motion)."""
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
            "depth_dir": depth_dir, "depth_paths": depth_paths,
            "image_size": HW, "camera_source": "identity"}


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


def _hand_occluder_dir(stage_dir):
    """Locate per-frame SAM2 HAND masks to use as a don't-care occluder region in
    the silhouette losses (the hand covering the object must not penalize the
    object pose). Looked up at <run_root>/hand_masks — generated by
    scripts/object_confidence.py (cached) — relative to the stage dir the caller
    runs in. Returns the absolute dir or None (losses then run without exclusion;
    graceful for fresh runs that have not produced hand masks yet)."""
    d = os.path.join(os.path.dirname(os.path.abspath(stage_dir)), "hand_masks")
    return d if os.path.isdir(d) and any(f.endswith(".npy") for f in os.listdir(d)) else None


def run_joint_optimizer(cfg, run_dir, s2, s6, frame_paths, mask_paths, K):
    """Joint differentiable hand+object render-and-compare (PyTorch3D + MANO, sam3d
    env). Optimizes MANO articulation + object 6D against keypoint-reprojection /
    silhouette / photometric + contact + non-penetration. Returns
    (hand_verts[T,778,3], hand_joints[T,21,3] or None, obj_poses[T,4,4]).
    Requires the threaded MANO params from stage 2."""
    import subprocess
    from ..logging_utils import log
    if s2.get("mano_pose") is None:
        raise BackendNotAvailable("joint optimizer needs MANO params (re-run stage 2)")
    repo = require_repo(cfg.paths.third_party, "sam-3d-objects", "")
    mano_dir = _resolve_mano_dir(cfg.paths.checkpoints)
    jo_dir = os.path.join(run_dir, "jo"); os.makedirs(jo_dir, exist_ok=True)
    out_npz = os.path.join(jo_dir, "out.npz")
    if not os.path.exists(out_npz):
        hnpz = os.path.join(jo_dir, "hand.npz")
        # hand_side (1=right, 0=left per frame): the optimizer must mirror its
        # right-hand MANO layer output on left-hand frames (HaMeR params are always
        # right-hand). Default to right for stage-2 bundles predating this key.
        hand_side = s2.get("hand_side")
        if hand_side is None:
            hand_side = np.ones(len(s2["verts"]), np.int64)
        # kp2d: HaMeR's 21 2D keypoints (full-image px, OpenPose joint order, already
        # un-mirrored for left hands) — drives the keypoint reprojection loss that
        # registers the hand to the image (the hand's primary image-space evidence).
        kp2d = s2.get("kp2d")
        if kp2d is None:
            kp2d = np.zeros((len(s2["verts"]), 21, 2))
        np.savez(hnpz, mano_global=s2["mano_global"], mano_pose=s2["mano_pose"],
                 mano_betas=s2["mano_betas"], verts=s2["verts"], joints=s2["joints"],
                 contact_idx=s2["contact_idx"], hand_faces=s2["hand_faces"],
                 hand_side=hand_side, kp2d=kp2d)
        onpz = os.path.join(jo_dir, "obj.npz")
        np.savez(onpz, verts=np.asarray(s6["obj_verts"]), faces=s6["obj_faces"],
                 vertex_colors=s6["obj_colors"], poses=s6["obj_poses"])
        Kp = os.path.join(jo_dir, "K.npy"); np.save(Kp, np.asarray(K))
        env_name = (cfg.backend.get("sam3d_env", "sam3d-objects")
                    if hasattr(cfg.backend, "get") else "sam3d-objects")
        conda = os.environ.get("CONDA_EXE", "conda")
        masks_dir = os.path.dirname(os.path.abspath(
            mask_paths[next(i for i, p in enumerate(mask_paths) if p)]))
        frames_dir = os.path.dirname(os.path.abspath(frame_paths[0]))
        cmd = [conda, "run", "--no-capture-output", "-n", env_name, "python",
               os.path.join(repo, "joint_opt.py"), "--hand", os.path.abspath(hnpz),
               "--obj", os.path.abspath(onpz), "--frames_dir", frames_dir,
               "--masks_dir", masks_dir, "--K", os.path.abspath(Kp),
               "--mano_dir", mano_dir, "--out", os.path.abspath(out_npz), "--iters", "400",
               "--w_kp2d", "3.0", "--kp_sigma", "60.0"]
        occl = _hand_occluder_dir(run_dir)
        if occl:
            cmd += ["--occluder_dir", occl]
        log("joint optimizer: differentiable MANO articulation + object "
            "(silhouette/photometric/contact)...")
        r = subprocess.run(cmd, cwd=repo)
        if r.returncode != 0 or not os.path.exists(out_npz):
            raise BackendNotAvailable(f"joint optimizer failed (exit {r.returncode}).")
    else:
        log(f"joint optimizer: reusing cached {out_npz}")
    d = np.load(out_npz)
    hj = d["hand_joints"] if "hand_joints" in d.files else None
    return d["hand_verts"], hj, d["obj_poses"]


def run_object_pose_render_compare(cfg, run_dir, frame_paths, mask_paths, K,
                                   verts, faces, colors, init_poses):
    """Differentiable render-and-compare object 6D refinement (PyTorch3D, sam3d env).

    Renders the textured SAM-3D mesh into each frame and optimizes the per-frame
    6D pose against silhouette IoU + a PHOTOMETRIC term (rendered texture vs RGB) —
    the photometric term recovers the spin-about-axis DOF the silhouette tracker
    can't see. Initialized from `init_poses` (the silhouette tracker). Cached.
    """
    import subprocess
    from ..logging_utils import log
    repo = require_repo(cfg.paths.third_party, "sam-3d-objects",
                        "git clone https://github.com/facebookresearch/sam-3d-objects")
    rc_dir = os.path.join(run_dir, "rc")
    os.makedirs(rc_dir, exist_ok=True)
    out_npz = os.path.join(rc_dir, "poses.npz")
    if not os.path.exists(out_npz):
        mesh_npz = os.path.join(rc_dir, "mesh.npz")
        np.savez(mesh_npz, verts=np.asarray(verts), faces=np.asarray(faces),
                 vertex_colors=np.asarray(colors))
        init_npz = os.path.join(rc_dir, "init.npz"); np.savez(init_npz, poses=init_poses)
        K_npz = os.path.join(rc_dir, "K.npy"); np.save(K_npz, np.asarray(K))
        env_name = (cfg.backend.get("sam3d_env", "sam3d-objects")
                    if hasattr(cfg.backend, "get") else "sam3d-objects")
        conda = os.environ.get("CONDA_EXE", "conda")
        masks_dir = os.path.dirname(os.path.abspath(mask_paths[
            next(i for i, p in enumerate(mask_paths) if p)]))
        frames_dir = os.path.dirname(os.path.abspath(frame_paths[0]))
        cmd = [conda, "run", "--no-capture-output", "-n", env_name, "python",
               os.path.join(repo, "render_compare.py"), "--mesh", os.path.abspath(mesh_npz),
               "--frames_dir", frames_dir, "--masks_dir", masks_dir,
               "--K", os.path.abspath(K_npz), "--init_poses", os.path.abspath(init_npz),
               "--out", os.path.abspath(out_npz), "--iters", "200"]
        occl = _hand_occluder_dir(run_dir)
        if occl:
            cmd += ["--occluder_dir", occl]
        log(f"object pose: differentiable render-and-compare (silhouette+photometric, "
            f"env '{env_name}')...")
        r = subprocess.run(cmd, cwd=repo)
        if r.returncode != 0 or not os.path.exists(out_npz):
            raise BackendNotAvailable(f"render-compare failed (exit {r.returncode}).")
    else:
        log(f"object pose: reusing cached render-compare poses {out_npz}")
    return np.load(out_npz)["poses"]


def run_object_pose_foundationpose(cfg, run_dir, frame_paths, mask_paths,
                                   depth_paths, K, verts, faces):
    """Per-frame object 6D pose via FoundationPose (model-based RGB-D tracker).

    Runs FoundationPose in the sam3d-objects env (reused — it has torch/pytorch3d/
    nvdiffrast/kaolin) as a subprocess: register on the largest-mask frame with the
    SAM-3D mesh + MoGe depth + SAM2 mask, then track bidirectionally. Returns
    poses[T,4,4] (object->camera). Cached per run. NOTE: FoundationPose expects
    reliable sensor depth; on monocular MoGe depth its translation can drift.
    """
    import subprocess
    import trimesh
    from ..logging_utils import log
    T = len(frame_paths)
    fp_dir = os.path.join(run_dir, "fp")
    os.makedirs(fp_dir, exist_ok=True)
    out_npz = os.path.join(fp_dir, "poses.npz")
    if not os.path.exists(out_npz):
        mesh_path = os.path.join(fp_dir, "mesh.obj")
        trimesh.Trimesh(np.asarray(verts), np.asarray(faces), process=False).export(mesh_path)
        K_path = os.path.join(fp_dir, "K.npy"); np.save(K_path, np.asarray(K))
        areas = [int(np.load(p).sum()) if p else 0 for p in mask_paths]
        rf = int(np.argmax(areas))
        if areas[rf] < 50:
            raise BackendNotAvailable("no usable mask to register FoundationPose")
        repo = require_repo(cfg.paths.third_party, "FoundationPose",
                            "git clone https://github.com/NVlabs/FoundationPose")
        env_name = (cfg.backend.get("sam3d_env", "sam3d-objects")
                    if hasattr(cfg.backend, "get") else "sam3d-objects")
        conda = os.environ.get("CONDA_EXE", "conda")
        depth_dir = os.path.dirname(os.path.abspath(depth_paths[0]))
        frames_dir = os.path.dirname(os.path.abspath(frame_paths[0]))
        masks_dir = os.path.dirname(os.path.abspath(mask_paths[rf]))
        cmd = [conda, "run", "--no-capture-output", "-n", env_name, "python",
               os.path.join(repo, "fp_track.py"),
               "--mesh", os.path.abspath(mesh_path),
               "--frames_dir", frames_dir, "--depth_dir", depth_dir,
               "--K", os.path.abspath(K_path),
               "--mask", os.path.abspath(mask_paths[rf]),
               "--register_frame", str(rf), "--out", os.path.abspath(out_npz),
               # give FP its best shot on monocular MoGe depth: clamp the z-smear
               # inside the object mask to a thin band around the near surface
               # (sensor-depth is what FP's stability assumes; this approximates it).
               "--masks_dir", masks_dir, "--clean_depth", "120"]
        log(f"object pose: running FoundationPose (register on frame {rf}, env "
            f"'{env_name}'); first run loads the refiner/scorer nets...")
        r = subprocess.run(cmd, cwd=repo)
        if r.returncode != 0 or not os.path.exists(out_npz):
            raise BackendNotAvailable(
                f"FoundationPose subprocess failed (exit {r.returncode}).")
    else:
        log(f"object pose: reusing cached FoundationPose poses {out_npz}")
    return np.load(out_npz)["poses"]


def _kabsch(A, B):
    """Least-squares rotation R (3x3) mapping centered point set A onto B (rows are
    points): B-mean(B) ~= (A-mean(A)) @ R.T."""
    Ac = A - A.mean(0)
    Bc = B - B.mean(0)
    U, S, Vt = np.linalg.svd(Ac.T @ Bc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def couple_object_to_hand(obj_poses, hand_joints, hand_valid, obj_radius,
                          grasp_pad=0.05):
    """Make a grasped object inherit the hand's rigid 6D motion (so it ROTATES with
    the wrist, not just translates).

    A held object is rigid w.r.t. the hand, so once grasped its pose is the hand's
    palm rigid transform — relative to the grasp-onset frame — applied to its rest
    pose. The depth-lift track only gives translation (R=I); this adds the missing
    rotation by Procrustes-fitting the rigid palm (wrist + finger MCP joints) frame
    to frame. Frames before grasp onset stay static at the rest pose. Detection
    gaps after onset hold the previous pose. Returns poses[T,4,4].
    """
    T = obj_poses.shape[0]
    PALM = [0, 5, 9, 13, 17]                  # wrist + finger MCPs = rigid palm
    centroids = obj_poses[:, :3, 3]
    palm = hand_joints[:, PALM, :]
    palm_c = palm.mean(1)
    valid = (hand_valid.any(1) if getattr(hand_valid, "ndim", 1) == 2
             else np.asarray(hand_valid, bool))
    r = float(obj_radius) if obj_radius and float(obj_radius) > 0 else 0.05
    grasp = valid & (np.linalg.norm(palm_c - centroids, axis=1) < 2.0 * r + grasp_pad)
    if not grasp.any():
        return obj_poses                      # never detected a grasp -> unchanged
    t0 = int(np.argmax(grasp))
    A0 = palm[t0]
    poses = np.tile(np.eye(4), (T, 1, 1))
    Rprev = np.eye(3)
    for t in range(T):
        if t >= t0 and grasp[t]:
            Rprev = _kabsch(A0, palm[t])       # hand-driven rotation (vs grasp onset)
            R = Rprev
        elif t >= t0:
            R = Rprev                          # detection gap: hold last rotation
        else:
            R = np.eye(3)                      # pre-grasp: object at rest orientation
        poses[t, :3, :3] = R
        # KEEP the image-grounded depth-lift translation (it reprojects onto the
        # object to within a few px); only inherit ROTATION from the hand. Rotating
        # about the (centered) object's centroid leaves the centroid where the
        # object actually is, so the object both follows the wrist AND stays
        # registered to the video. (Earlier this used a palm-derived translation,
        # which made the object drift off the real object by 50-230 px.)
        poses[t, :3, 3] = centroids[t]
    return poses


def run_object_sam3d(cfg, run_dir, frame_paths, mask_paths, depth_paths, K):
    """SAM-3D-Objects textured mesh for the interacting object, placed on the
    metric depth-lift 6D track.

    SAM-3D Objects (a separate torch-2.5.1 conda env) reconstructs a full textured
    mesh from ONE RGB frame + its object mask. We run it as a subprocess (its deps
    conflict with this env's torch) and load the resulting verts/faces/vertex
    colors. That mesh is canonical + normalized, so we:
      (a) orient it from SAM-3D's y-up canonical frame into the camera frame,
      (b) scale it to the object's METRIC size measured from depth at the anchor
          frame (projected mask bbox -> metres), and
      (c) reuse the depth-lift centroid trajectory for the 6D poses — that is what
          keeps the object co-located with the hand.
    Returns dict(verts, faces, vertex_colors, poses, radius, anchor_frame).
    """
    import subprocess
    # depth-lift gives the metric 6D translation track that aligns with the hand.
    track = run_object_depthlift(cfg, frame_paths, mask_paths, depth_paths, K)

    # anchor frame = largest visible object mask (best single view for image->3D)
    areas = [int(np.load(mp).sum()) if mp else 0 for mp in mask_paths]
    bi = int(np.argmax(areas))
    if areas[bi] < 50:
        raise BackendNotAvailable("no usable object mask for SAM-3D (check stage1)")

    # metric extent of the visible object at the anchor frame (projected bbox -> m)
    m = np.load(mask_paths[bi]).astype(bool)
    d = np.load(depth_paths[bi]).astype(np.float32)
    ys, xs = np.where(m)
    zsel = d[ys, xs]; zsel = zsel[zsel > 0]
    z_med = float(np.median(zsel)) if zsel.size else 0.3
    h_px = float(ys.max() - ys.min() + 1)
    w_px = float(xs.max() - xs.min() + 1)
    metric_extent = float(max(h_px * z_med / K[1, 1], w_px * z_med / K[0, 0]))

    repo = require_repo(cfg.paths.third_party, "sam-3d-objects",
                        "git clone https://github.com/facebookresearch/sam-3d-objects")
    out_npz = os.path.join(run_dir, "sam3d", "object.npz")
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    from ..logging_utils import log
    # SAM-3D is deterministic per (frame, seed) and slow (~13GB weights); cache it
    # independently of stage --force (which recomputes only the cheap stage logic).
    # Delete this run's sam3d/object.npz to regenerate the mesh.
    if os.path.exists(out_npz):
        log(f"object: reusing cached SAM-3D mesh {out_npz} (delete it to regenerate)")
    else:
        env_name = (cfg.backend.get("sam3d_env", "sam3d-objects")
                    if hasattr(cfg.backend, "get") else "sam3d-objects")
        conda = os.environ.get("CONDA_EXE", "conda")
        cmd = [conda, "run", "--no-capture-output", "-n", env_name, "python",
               os.path.join(repo, "sam3d_infer.py"), "--no-texture",
               "--image", os.path.abspath(frame_paths[bi]),
               "--mask", os.path.abspath(mask_paths[bi]),
               "--out", os.path.abspath(out_npz)]
        log(f"object: running SAM-3D-Objects on frame {bi} in env '{env_name}' "
            f"(this loads ~13GB of weights; first run is slow)...")
        r = subprocess.run(cmd, cwd=repo)
        if r.returncode != 0 or not os.path.exists(out_npz):
            raise BackendNotAvailable(
                f"SAM-3D subprocess failed (exit {r.returncode}). Build its env first: "
                "see third_party/sam-3d-objects/doc/setup.md (conda env 'sam3d-objects').")

    data = np.load(out_npz)
    verts = data["verts"].astype(np.float64)
    faces = data["faces"].astype(np.int64)
    colors = data["vertex_colors"].astype(np.uint8)

    # orient SAM-3D y-up canonical mesh into the camera frame (OpenCV: y-down,
    # z-forward), center at origin, and scale to the metric extent from depth.
    verts = verts @ np.diag([1.0, -1.0, -1.0]).T
    verts -= verts.mean(0)
    cur_extent = float((verts.max(0) - verts.min(0)).max())
    verts *= metric_extent / max(cur_extent, 1e-9)
    radius = float(np.linalg.norm(verts - verts.mean(0), axis=1).mean())

    log(f"object: SAM-3D mesh {verts.shape[0]} verts, scaled to "
        f"{metric_extent*100:.1f}cm extent; using depth-lift 6D track")
    return {"verts": verts, "faces": faces, "vertex_colors": colors,
            "poses": track["poses"], "radius": np.array(radius),
            "anchor_frame": bi}


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


def _patch_numpy_for_chumpy():
    """Restore the numpy scalar aliases (np.bool/int/float/...) removed in
    numpy>=1.24 so chumpy 0.70 imports under this env's numpy 2.x.

    chumpy is pulled in implicitly when smplx unpickles the official MANO .pkl
    (its arrays are chumpy objects); chumpy's __init__ does `from numpy import
    bool, int, float, ...`, which reads these as numpy module attributes. Setting
    them back *before* chumpy is first imported makes that import succeed without
    downgrading numpy (which MoGe/SAM2 need at >=2). Idempotent; no-op once set.
    """
    import numpy as np
    for name, typ in {"bool": bool, "int": int, "float": float, "complex": complex,
                      "object": object, "str": str, "unicode": str, "long": int}.items():
        if name not in np.__dict__:        # dict check avoids numpy's __getattr__ FutureWarning
            np.__dict__[name] = typ


def _resolve_mano_dir(ckpt_root):
    """Return the directory that actually contains MANO_RIGHT.pkl.

    Supports both the flat layout (checkpoints/mano/MANO_RIGHT.pkl) and the
    official MANO_v1_2 archive layout, which extracts to
    checkpoints/mano/mano_v1_2/models/MANO_RIGHT.pkl. The returned dir is what
    HaMeR/smplx is pointed at via MANO.MODEL_PATH, so it must hold the .pkl
    directly. Returns None if no MANO model is found.
    """
    candidates = [
        os.path.join(ckpt_root, "mano"),
        os.path.join(ckpt_root, "mano", "models"),
        os.path.join(ckpt_root, "mano", "mano_v1_2", "models"),
    ]
    for d in candidates:
        if os.path.exists(os.path.join(d, "MANO_RIGHT.pkl")):
            return d
    return None


def _hand_metric_anchor(depth_path, box, K):
    """Metric 3D anchor for the hand from MoGe depth + real intrinsics.

    HaMeR returns a correctly-shaped, metric MANO hand but places it with a
    *fabricated* focal length, so its absolute depth is wrong by metres and varies
    wildly frame to frame (the hand 'flies away' from the object, which lives in
    the MoGe metric frame). We discard HaMeR's depth and instead backproject the
    hand-box centre at the box's foreground depth using the SAME metric map +
    intrinsics the object is reconstructed from — so hand and object share one
    metric camera frame. Returns the metric hand-centroid position (3,) or None.
    """
    d = np.load(depth_path).astype(np.float32)
    H, W = d.shape
    x0, y0, x1, y1 = box
    x0, x1 = np.clip([x0, x1], 0, W - 1)
    y0, y1 = np.clip([y0, y1], 0, H - 1)
    cx_px, cy_px = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    # central sub-window of the box -> stable hand depth (avoids box-edge background)
    hw, hh = 0.25 * (x1 - x0), 0.25 * (y1 - y0)
    sub = d[int(cy_px - hh):int(cy_px + hh) + 1, int(cx_px - hw):int(cx_px + hw) + 1]
    pos = sub[sub > 0]
    if pos.size < 4:
        crop = d[int(y0):int(y1) + 1, int(x0):int(x1) + 1]
        pos = crop[crop > 0]
        if pos.size < 4:
            return None
    z = float(np.median(pos))
    if not np.isfinite(z) or z <= 0:
        return None
    X = (cx_px - K[0, 2]) / K[0, 0] * z
    Y = (cy_px - K[1, 2]) / K[1, 1] * z
    return np.array([X, Y, z], np.float64)


def _mano_fingertip_idx(verts, tips=(744, 320, 443, 554, 672), rad=0.025):
    """Contact candidate vertices = MANO surface within `rad` of the 5 fingertips."""
    tip_pts = verts[list(tips)]
    d = np.min(np.linalg.norm(verts[:, None, :] - tip_pts[None], axis=2), axis=1)
    idx = np.where(d < rad)[0]
    return idx if len(idx) >= 20 else np.argsort(d)[:120]


def run_hand(cfg, frame_paths, hand_boxes, hand_valid, depth_paths=None, K=None):
    """HaMeR per-frame MANO reconstruction using stage-1 hand boxes (no detectron2).

    Returns dict(betas, orient, pose, transl, joints[T,21,3], verts[T,778,3],
    contact_idx, hand_faces). Hard-gated on the MANO model (license).

    When depth_paths + K are supplied (real run with MoGe metric depth), the hand
    is placed by anchoring its centroid to the metric depth at the hand box (see
    _hand_metric_anchor), so it shares the object's metric camera frame. Without
    them it falls back to HaMeR's weak-perspective translation (depth unreliable).
    """
    import sys
    import torch
    import cv2

    _patch_numpy_for_chumpy()   # let chumpy (MANO .pkl unpickling) import on numpy 2.x

    # MANO (the license-gated blocker) is checked first — it's the action you own.
    mano_dir = _resolve_mano_dir(cfg.paths.checkpoints)
    if mano_dir is None:
        flat = os.path.join(cfg.paths.checkpoints, "mano")
        raise BackendNotAvailable(
            "MANO model required for HaMeR but missing. It is LICENSE-GATED and "
            "cannot be auto-downloaded: register at https://mano.is.tue.mpg.de, "
            f"accept the license, then place MANO_RIGHT.pkl at {flat}/MANO_RIGHT.pkl "
            f"(or extract MANO_v1_2 to {flat}/mano_v1_2/models/).\n"
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
    # Drop the training-time backbone init weights ('hamer_training_data/
    # vitpose_backbone.pth'): they aren't shipped with the demo weights and the
    # full hamer.ckpt already contains the trained backbone (loaded below via
    # load_from_checkpoint). HaMeR's own load_hamer() pops this for the same reason.
    if "PRETRAINED_WEIGHTS" in model_cfg.MODEL.BACKBONE:
        model_cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
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
    # MANO articulation params (rotmats) + 2D keypoints — kept so the render-and-
    # compare optimizer (Phase 2) can re-articulate the hand and fit 2D keypoints.
    glob_all = np.tile(np.eye(3), (T, 1, 1))                    # global_orient (T,3,3)
    pose_all = np.tile(np.eye(3), (T, 15, 1, 1))               # hand_pose (T,15,3,3)
    betas_all = np.zeros((T, 10))
    kp2d_all = np.zeros((T, 21, 2))                            # full-image px
    side_all = np.zeros(T, np.int64)                          # 1=right,0=left
    # Single-hand assumption: decide the side ONCE by majority vote over the clip.
    # YOLO occasionally flips the side label on isolated frames (e.g. 1 'right'
    # among 162 'left' detections); honouring those per frame mirrors the MANO
    # estimate on exactly those frames and corrupts the track. The box from the
    # minority slot is still used (it is the same physical hand, mislabelled) —
    # only the side flag fed to HaMeR is forced to the dominant one.
    dom = 1 if hand_valid[:, 1].sum() >= hand_valid[:, 0].sum() else 0
    ref = None
    for i, p in enumerate(frame_paths):
        img = cv2.imread(p)[:, :, ::-1].copy()                  # BGR->RGB
        slot = dom if hand_valid[i, dom] else ((1 - dom) if hand_valid[i, 1 - dom] else None)
        if slot is None:
            if i > 0:
                verts_all[i], joints_all[i] = verts_all[i - 1], joints_all[i - 1]
                glob_all[i], pose_all[i], betas_all[i] = glob_all[i-1], pose_all[i-1], betas_all[i-1]
                kp2d_all[i], side_all[i] = kp2d_all[i-1], side_all[i-1]
            continue
        box = hand_boxes[i, slot][None, :]
        right = np.array([dom])
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

        anchor = None
        if depth_paths is not None and K is not None:
            anchor = _hand_metric_anchor(depth_paths[i], hand_boxes[i, slot], K)
        if anchor is not None:
            # keep HaMeR's metric shape + articulation; place its centroid at the
            # metric depth point (same frame as the object) instead of HaMeR's
            # fabricated-focal depth.
            c = v.mean(0)
            verts_all[i], joints_all[i] = v - c + anchor, j - c + anchor
        else:
            pc = out["pred_cam"][0].cpu().numpy()
            pc[1] *= s                  # crop-cam tx is in mirrored-crop coords for left hands
            bc = batch["box_center"][0].cpu().numpy()
            bs = float(batch["box_size"][0].cpu().numpy())
            isz = batch["img_size"][0].cpu().numpy()
            sf = focal / img_sz_model * float(max(isz))
            t = _cam_crop_to_full(pc, bc, bs, isz, sf)
            verts_all[i], joints_all[i] = v + t, j + t

        # MANO articulation params + 2D keypoints (full-image px) for the optimizer
        mp = out["pred_mano_params"]
        glob_all[i] = mp["global_orient"][0].reshape(3, 3).cpu().numpy()
        pose_all[i] = mp["hand_pose"][0].reshape(15, 3, 3).cpu().numpy()
        betas_all[i] = mp["betas"][0].reshape(-1)[:10].cpu().numpy()
        side_all[i] = int(right[0])
        bc = batch["box_center"][0].cpu().numpy()
        bs = float(batch["box_size"][0].cpu().numpy())
        kp = out["pred_keypoints_2d"][0].cpu().numpy()          # crop-normalized
        kp[:, 0] *= s                                           # un-mirror left-hand crops
        kp2d_all[i] = bc[None, :] + kp * bs                     # -> full-image px
        ref = i if ref is None else ref

    if ref is None:
        raise BackendNotAvailable("no hands detected in any frame (stage1)")
    # Back-fill leading frames with no detection (hand not yet tracked during the
    # approach) using the first valid pose, so the hand rests at its entry point
    # instead of collapsing to the camera origin (0,0,0).
    for i in range(ref):
        verts_all[i], joints_all[i] = verts_all[ref], joints_all[ref]
        glob_all[i], pose_all[i], betas_all[i] = glob_all[ref], pose_all[ref], betas_all[ref]
        kp2d_all[i], side_all[i] = kp2d_all[ref], side_all[ref]
    contact_idx = _mano_fingertip_idx(verts_all[ref])
    return {"betas": np.zeros(10), "orient": np.zeros((T, 3)), "pose": np.zeros((T, 45)),
            "transl": joints_all[:, 0, :], "joints": joints_all, "verts": verts_all,
            "contact_idx": contact_idx, "hand_faces": faces,
            # MANO params (rotmats) + 2D keypoints for the render-and-compare optimizer
            "mano_global": glob_all, "mano_pose": pose_all, "mano_betas": betas_all,
            "kp2d": kp2d_all, "hand_side": side_all}
