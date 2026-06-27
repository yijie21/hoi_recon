"""Config: YAML defaults + CLI overrides. Attribute-accessible dict.
Adapted from render_and_compare/hoi_recon/config.py (self-contained vendor)."""
from __future__ import annotations
import copy, os
from typing import Any, Dict
import yaml

# Method root = parent of the `egoaero` package dir, so runs/ checkpoints/
# third_party/ resolve under egoaero/ regardless of CWD.
_METHOD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_DEFAULTS: Dict[str, Any] = {
    "mock": True,
    "seed": 0,
    "num_frames": 48,
    "video": None,
    "backend": {                      # real-mode drivers (gated; see backends/real.py)
        "hand": "hawor",
        "object": "bundlesdf",
        "mesh": "sam3d",
        "camera": "orbslam3",
        "segment": "sam3",
        "mllm": "none",
    },
    "track": {                        # App A defaults (DOCUMENTED — see ASSUMPTIONS.md)
        "ransac_thresh_m": 0.005,
        "ransac_iters": 200,
        "memory_topk": 4,
        "q_weights": {"area": 1.0, "depth": 0.5, "view": 0.5, "occ": 1.0},
        "q_insert_thresh": 0.3,
        "sel_weights": {"overlap": 1.0, "rot": 0.5, "quality": 0.5},
        "graph_weights": {"feat": 1.0, "geo": 1.0, "sdf": 0.0, "mask": 0.2, "pose": 0.1},
        "graph_iters": 10,
        "drift_sigma_m": 0.01,        # mock: injected per-frame translation drift
        "drift_sigma_deg": 3.0,       # mock: injected per-frame rotation drift
    },
    "hand": {
        "depth_bias_m": 0.03,         # mock: injected monocular global depth bias
        "corr_neighborhood_px": 7,
    },
    "ego": {                          # §2.1.4
        "smooth_window": 5,
    },
    "contact": {                      # App C — constants verbatim; weights documented
        "contact_gap_m": 0.0005,      # 0.5 mm
        "thenar_gap_m": 0.0018,       # 1.8 mm
        "opp_gap_m": 0.0005,          # DOCUMENTED default (paper gives thumb/thenar only)
        "max_global_trans_m": 0.034,  # 34 mm
        "max_finger_disp_m": 0.015,   # 15 mm
        "max_pushback_m": 0.008,      # 8 mm
        "smooth_window": 9,
        "boundary_frames": 6,
        "pen_eps_m": 0.002,           # DOCUMENTED penetration threshold default
        "contact_mask_radius_factor": 4.0,  # DOCUMENTED default: contact-mask zone = factor * contact_gap
        "region_weights": {"thumb": 1.0, "opp": 1.0, "hukou": 0.5},  # DOCUMENTED
        "rotation_enabled": False,    # whole-hand rotation disabled (0 deg)
    },
    "quality": {                      # App E — paper gives NO constants; all DOCUMENTED defaults
        "eps_g_m": 0.004,             # contact-gap recoverability threshold (4 mm)
        "eps_delta_m": 0.012,         # per-finger correction-budget threshold (12 mm)
        "delta_max_m": 0.015,         # budget normalizer (= contact.max_finger_disp_m)
        "alpha": 1.0,                 # R_after weight in Q
        "beta": 0.5,                  # B_repair weight in Q
        "gamma": 1.0,                 # U_unresolved weight in Q
        "pen_ref_mm": 50000.0,        # R_after penetration normalizer (mock-scale)
        "gap_ref_mm": 40.0,           # R_after contact-gap normalizer
        "obj_move_thresh_m_per_frame": 0.01,  # object-moving threshold for U_unresolved (per-frame translation)
        "q_accept": 0.6,              # Q >= -> accept
        "q_repairable": 0.3,          # q_repairable <= Q < q_accept -> repairable_accept
    },
    "paths": {
        "third_party": os.path.join(_METHOD_ROOT, "third_party"),
        "checkpoints": os.path.join(_METHOD_ROOT, "checkpoints"),
    },
    "force": False,
}

class Config(dict):
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
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(dict(cfg), f, sort_keys=False)
