"""Real perception backend adapters (stages 0-3).

These are deliberately thin: they validate that the third-party repo + weights are
present, then call into them. The actual model-call bodies are marked TODO and
raise `BackendNotAvailable` until wired, because the exact API differs per repo and
per checkpoint version. Each function documents the output contract that the
matching mock path already fulfils, so wiring is a fill-in-the-blank.
"""
from __future__ import annotations

from . import BackendNotAvailable, require_repo, require_ckpt

# ---- repo registry (name in third_party/, clone URL hint) ------------------
REPOS = {
    "hamer":          ("hamer",              "https://github.com/geopavlakos/hamer"),
    "wilor":          ("WiLoR",              "https://github.com/rolpotamias/WiLoR"),
    "dyn_hamr":       ("Dyn-HaMR",           "https://github.com/ZhengdiYu/Dyn-HaMR"),
    "hawor":          ("HaWoR",              "https://github.com/ThunderVVV/HaWoR"),
    "sam2":           ("sam2",               "https://github.com/facebookresearch/sam2"),
    "cotracker":      ("co-tracker",         "https://github.com/facebookresearch/co-tracker"),
    "sam3d":          ("sam-3d-objects",     "https://github.com/facebookresearch/sam-3d-objects"),
    "bundlesdf":      ("BundleSDF",          "https://github.com/NVlabs/BundleSDF"),
    "foundationpose": ("FoundationPose",     "https://github.com/NVlabs/FoundationPose"),
    "moge":           ("MoGe",               "https://github.com/microsoft/MoGe"),
    "depth_anything_v2": ("Depth-Anything-V2", "https://github.com/DepthAnything/Depth-Anything-V2"),
    "vipe":           ("vipe",               "https://github.com/nv-tlabs/vipe"),
}


def _repo(cfg, key):
    name, url = REPOS[key]
    return require_repo(cfg.paths.third_party, name, f"git clone {url}")


# --------------------------------------------------------------------------
# Stage 0: camera + depth
# --------------------------------------------------------------------------
def run_camera(cfg, frames):
    """-> dict(intrinsics[3,3], extrinsics[T,4,4], image_size(H,W), fps)."""
    _repo(cfg, cfg.backend.camera)
    raise BackendNotAvailable(
        f"camera backend '{cfg.backend.camera}' repo present but adapter not wired. "
        "Fill backends.perception.run_camera() to call VIPE/DROID-SLAM and return "
        "intrinsics + per-frame world->cam extrinsics.")


def run_depth(cfg, frames, out_dir):
    """-> dict(depth_paths[List[str]]) metric depth per frame as .npy."""
    _repo(cfg, cfg.backend.depth)
    require_ckpt(cfg.paths.checkpoints, f"{cfg.backend.depth}/weights.pt",
                 f"download {cfg.backend.depth} weights")
    raise BackendNotAvailable(
        f"depth backend '{cfg.backend.depth}' adapter not wired (run_depth()).")


# --------------------------------------------------------------------------
# Stage 1: detection / segmentation / tracking
# --------------------------------------------------------------------------
def run_detect_track(cfg, frames, out_dir):
    """-> dict(hand_boxes[T,2,4], hand_valid[T,2], object_box[T,4],
              object_mask_paths, object_amodal_paths, point_tracks)."""
    _repo(cfg, "wilor")
    _repo(cfg, "sam2")
    raise BackendNotAvailable("detect/track adapter not wired (run_detect_track()).")


# --------------------------------------------------------------------------
# Stage 2: hand reconstruction
# --------------------------------------------------------------------------
def run_hand(cfg, frames, detections, camera):
    """-> dict(betas[10], orient[T,3], pose[T,45], transl[T,3],
              joints[T,21,3], verts[T,778,3], contact_idx[Nc])."""
    _repo(cfg, cfg.backend.hand)            # hamer / wilor / hawor
    if cfg.backend.hand in ("hamer", "wilor"):
        _repo(cfg, "dyn_hamr")             # temporal world-space stabilization
    require_ckpt(cfg.paths.checkpoints, "mano/MANO_RIGHT.pkl",
                 "register at https://mano.is.tue.mpg.de and place MANO_RIGHT.pkl")
    raise BackendNotAvailable(
        f"hand backend '{cfg.backend.hand}' adapter not wired (run_hand()).")


# --------------------------------------------------------------------------
# Stage 3: object shape + 6D pose
# --------------------------------------------------------------------------
def run_object(cfg, frames, detections, depth, camera):
    """-> dict(verts[No,3] canonical, faces[Mo,3], poses[T,4,4], radius/scale)."""
    _repo(cfg, cfg.backend.object)          # sam3d / bundlesdf / foundationpose
    raise BackendNotAvailable(
        f"object backend '{cfg.backend.object}' adapter not wired (run_object()).")
