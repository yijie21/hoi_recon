"""Configuration: YAML defaults merged with CLI overrides.

A tiny attribute-accessible dict is used instead of a heavy schema lib so the
config stays trivially serializable into each run directory for reproducibility.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Dict

import yaml

# Repo root = parent of the `hoi_recon` package dir. Used to anchor checkpoint /
# third-party paths so the pipeline finds weights no matter what CWD it is
# launched from (e.g. `python -m hoi_recon.cli` run from outside the repo).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_DEFAULTS: Dict[str, Any] = {
    "mock": True,            # synthetic scene + injected per-stage error (no weights)
    "seed": 0,
    "num_frames": 48,        # used in mock mode if no video is given
    "video": None,           # path to input RGB video (real mode / mock frame count)
    "backend": {
        "hand": "hamer",     # hamer | wilor | hawor
        "object": "sam3d",   # sam3d | bundlesdf | foundationpose
        "depth": "moge",     # moge | depth_anything_v2 | metric3d
        "camera": "vipe",    # vipe | droid_slam
    },
    "contact": {
        "dist_thresh_m": 0.02,      # 2 cm validity gate (CHOIR)
        "normal_thresh_deg": 60.0,  # surface-normal compatibility cone
        "knn_k": 50,
    },
    "optim": {               # stage7 contact-aware optimization (normalized energies)
        "iters": 200,
        "lr": 0.02,
        "w_contact": 1.0,
        "w_pen": 0.3,
        "w_temporal": 0.2,
        "w_anchor": 0.02,
    },
    "smoothing": {           # stage5 temporal smoothing
        "window": 5,
    },
    "paths": {
        "third_party": os.path.join(_REPO_ROOT, "third_party"),
        "checkpoints": os.path.join(_REPO_ROOT, "checkpoints"),
    },
    "force": False,          # recompute stages even if cached
}


class Config(dict):
    """dict with attribute access and recursive wrapping."""

    def __getattr__(self, k: str) -> Any:
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Config(v) if isinstance(v, dict) else v

    def __setattr__(self, k: str, v: Any) -> None:
        self[k] = v


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if v is None:
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(yaml_path: str | None = None, overrides: dict | None = None) -> Config:
    cfg = copy.deepcopy(_DEFAULTS)
    if yaml_path:
        with open(yaml_path) as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return Config(cfg)


def save_config(cfg: Config, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(dict(cfg), f, sort_keys=False)
