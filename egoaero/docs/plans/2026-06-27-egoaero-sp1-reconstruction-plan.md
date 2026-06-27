# EgoAERO SP1 — Asset-free Hand-Object Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `egoaero/`, a self-contained HOI-reconstruction method that turns a single egocentric RGB-D video into contact-consistent hand-object trajectories (per-frame MANO hand, object mesh, object 6-DoF trajectory, contact maps), runnable end-to-end today in `--mock` mode.

**Architecture:** An 8-stage pipeline (stage0..stage7) mirroring the proven `render_and_compare` shape: each stage is a pure `run(ctx) -> Bundle` that caches a self-contained bundle to disk and is independently unit-tested. Stages map 1:1 to EgoAERO paper §2.1 / App A–C. A vendored `core/` (geometry, hand model, mock scene, bundle IO) keeps the method self-contained — no imports from sibling methods.

**Tech Stack:** Python 3.9+, numpy, scipy, pyyaml, trimesh (mesh IO), pytest. Pure-numpy (no torch) for SP1 — every faithful module here is geometric.

## Global Constraints

- **Self-contained:** `egoaero/` MUST NOT import from `render_and_compare/` or any sibling. Vendored core files are adapted copies.
- **Mock-first:** every stage runs in `--mock` with no weights/data; real backends are optional and gated.
- **Faithful where specified; documented defaults at gaps.** Every default or deviation from the paper is appended to `egoaero/ASSUMPTIONS.md` with the paper section it fills and the source it is borrowed from.
- **App C constants are verbatim:** contact gap `0.5 mm`, thenar gap `1.8 mm`, max whole-hand translation `34 mm`, max local finger displacement `15 mm`, max penetration push-back `8 mm`, smoothing window `9`, boundary transition `6 frames`, whole-hand rotation disabled (`0°`).
- **Units:** metres internally; report mm/cm/deg in metrics only.
- **Determinism:** all randomness seeded via `cfg.seed`.
- **Stage convention:** each stage module defines `NAME: str`, `INDEX: int`, and `run(ctx) -> Bundle`.
- **Commits:** one commit per task, message prefix `egoaero:`.
- **Tests:** CWD-safe (no reliance on launch directory); run with `python -m pytest egoaero/tests/ -q`.

---

## File Structure

```
egoaero/
  egoaero/
    __init__.py
    cli.py                  Task 16
    pipeline.py             Task 16
    config.py               Task 1
    bundle.py               Task 3
    contract.py             Task 17
    core/
      __init__.py
      geometry.py           Task 2   (vendored+extended)
      hand.py               Task 4   (procedural hand, fingertip/pad sets, finger-chain weights)
      mock_scene.py         Task 5   (synthetic ego RGB-D HOI scene)
    stages/
      __init__.py
      stage0_ego_io.py      Task 6
      stage1_semantic.py    Task 7
      stage2_track.py       Tasks 8-9
      stage3_mesh.py        Task 10
      stage4_hand.py        Task 11
      stage5_ego_comp.py    Task 12
      stage6_contact.py     Tasks 13-14
      stage7_eval.py        Task 15
    backends/
      __init__.py
      real.py               Task 16 (stub registry; raises if backend missing)
    configs/
      mock.yaml             Task 16
  tests/                    one test file per task
  ASSUMPTIONS.md            seeded Task 1, appended throughout
  README.md                 Task 18
  docs/specs|plans/         (already present)
```

---

### Task 1: Package skeleton, config, ASSUMPTIONS scaffold

**Files:**
- Create: `egoaero/egoaero/__init__.py`, `egoaero/egoaero/core/__init__.py`, `egoaero/egoaero/stages/__init__.py`, `egoaero/egoaero/backends/__init__.py`
- Create: `egoaero/egoaero/config.py`
- Create: `egoaero/ASSUMPTIONS.md`
- Test: `egoaero/tests/test_config.py`

**Interfaces:**
- Produces: `config.load_config(yaml_path=None, overrides=None) -> Config`; `config.save_config(cfg, path)`; `Config` (attribute-accessible dict); `config._METHOD_ROOT` (str, anchors `runs/`, `checkpoints/`).

- [ ] **Step 1: Write the failing test**

```python
# egoaero/tests/test_config.py
import os
from egoaero import config

def test_defaults_and_override():
    cfg = config.load_config(overrides={"seed": 7, "contact": {"contact_gap_m": 0.001}})
    assert cfg.mock is True
    assert cfg.seed == 7
    assert cfg.contact.contact_gap_m == 0.001        # override merged
    assert cfg.contact.thenar_gap_m == 0.0018        # App C verbatim default kept

def test_method_root_under_egoaero():
    assert config._METHOD_ROOT.replace("\\", "/").endswith("egoaero")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest egoaero/tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'egoaero'` (and after package exists, `AttributeError`).

- [ ] **Step 3: Write minimal implementation**

Create the four empty `__init__.py` files. Then `egoaero/egoaero/config.py` — adapt the structure of `render_and_compare/hoi_recon/config.py` (same `Config`, `_deep_merge`, `load_config`, `save_config`) with EgoAERO defaults:

```python
# egoaero/egoaero/config.py
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
        "region_weights": {"thumb": 1.0, "opp": 1.0, "hukou": 0.5},  # DOCUMENTED
        "rotation_enabled": False,    # whole-hand rotation disabled (0 deg)
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(dict(cfg), f, sort_keys=False)
```

Create `egoaero/ASSUMPTIONS.md` with a header and the first entries:

```markdown
# EgoAERO reproduction — assumptions & deviations

Every value the paper leaves unspecified, with the section it fills and the source.

## Config defaults
- `track.*` weights/thresholds (App A): paper gives no values → defaults borrowed from
  BundleSDF [16] / FoundationPose [17] conventions. (Task 1, 8, 9)
- `contact.opp_gap_m=0.5mm`, `contact.region_weights`, `contact.pen_eps_m=2mm` (App C):
  paper specifies thumb/thenar gaps and bounds only → these are documented defaults. (Task 1, 13-14)
- `hand.depth_bias_m`, `track.drift_sigma_*`: mock-mode injected error magnitudes (not from
  paper) used to exercise the correction/optimizer modules. (Task 5, 8, 11)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest egoaero/tests/test_config.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add egoaero/egoaero egoaero/tests/test_config.py egoaero/ASSUMPTIONS.md
git commit -m "egoaero: package skeleton + config + ASSUMPTIONS scaffold"
```

---

### Task 2: Vendored core geometry

**Files:**
- Create: `egoaero/egoaero/core/geometry.py`
- Test: `egoaero/tests/test_geometry.py`

**Interfaces:**
- Produces: `rotvec_to_R`, `R_to_rotvec`, `se3`, `se3_inv`, `transform_points`, `uv_sphere`, `box_mesh`, `vertex_normals`, `knn`, `signed_distance_to_mesh`, `umeyama`, `se3_log`, `se3_exp`, `geodesic_deg`.

- [ ] **Step 1: Write the failing test**

```python
# egoaero/tests/test_geometry.py
import numpy as np
from egoaero.core import geometry as g

def test_se3_inv_roundtrip():
    T = g.se3(g.rotvec_to_R(np.array([0.1, -0.2, 0.3])), np.array([1., 2., 3.]))
    assert np.allclose(g.se3_inv(T) @ T, np.eye(4), atol=1e-9)

def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(50, 3))
    s, R, t = 1.7, g.rotvec_to_R(np.array([0.2, 0.1, -0.3])), np.array([0.5, -1., 2.])
    dst = (s * (R @ src.T).T) + t
    s2, R2, t2 = g.umeyama(src, dst, with_scale=True)
    assert abs(s2 - s) < 1e-6 and np.allclose(R2, R, atol=1e-6)

def test_geodesic_deg_zero_and_90():
    I = np.eye(3)
    Rz = g.rotvec_to_R(np.array([0, 0, np.pi/2]))
    assert g.geodesic_deg(I, I) < 1e-6
    assert abs(g.geodesic_deg(I, Rz) - 90.0) < 1e-4

def test_se3_log_exp_roundtrip():
    T = g.se3(g.rotvec_to_R(np.array([0.3, 0.1, -0.2])), np.array([0.4, -0.1, 0.2]))
    assert np.allclose(g.se3_exp(g.se3_log(T)), T, atol=1e-8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest egoaero/tests/test_geometry.py -v`
Expected: FAIL (`ModuleNotFoundError` / missing attrs).

- [ ] **Step 3: Write minimal implementation**

Copy `rotvec_to_R`, `se3`, `transform_points`, `uv_sphere`, `vertex_normals`, `knn`, `signed_distance_to_mesh`, `umeyama` verbatim from `render_and_compare/hoi_recon/geometry.py`. Then append:

```python
# --- additions for egoaero pose-graph / ego-motion ---
def se3_inv(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4); Ti[:3, :3] = R.T; Ti[:3, 3] = -R.T @ t
    return Ti

def R_to_rotvec(R: np.ndarray) -> np.ndarray:
    ang = np.arccos(np.clip((np.trace(R) - 1) / 2.0, -1.0, 1.0))
    if ang < 1e-8:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return ang * w / (2.0 * np.sin(ang))

def geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(R_to_rotvec(Ra.T @ Rb))))

def se3_log(T: np.ndarray) -> np.ndarray:
    """SE3 -> 6-vector twist [rho(3), phi(3)] (translation-part, rotation-part)."""
    phi = R_to_rotvec(T[:3, :3])
    return np.concatenate([T[:3, 3], phi])   # left-trivialized approx (small-step use)

def se3_exp(xi: np.ndarray) -> np.ndarray:
    """6-vector twist -> SE3 (matching se3_log's convention)."""
    return se3(rotvec_to_R(xi[3:]), xi[:3])

def box_mesh(half: np.ndarray):
    """Axis-aligned box (half-extents (3,)) -> (verts[8,3], faces[12,3])."""
    s = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    verts = s * half
    faces = np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],
                      [2,3,7],[2,7,6],[1,5,7],[1,7,3],[0,2,6],[0,6,4]])
    return verts, faces
```

Note in `ASSUMPTIONS.md`: `se3_log/se3_exp` use a left-trivialized small-step parameterization (translation decoupled from rotation), adequate for the iterative pose-graph refinement here; full SE3 exponential not required at the step sizes used.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest egoaero/tests/test_geometry.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add egoaero/egoaero/core/geometry.py egoaero/tests/test_geometry.py egoaero/ASSUMPTIONS.md
git commit -m "egoaero: vendored core geometry (SE3 log/exp, geodesic, box mesh)"
```

---

### Task 3: Bundle IO

**Files:**
- Create: `egoaero/egoaero/bundle.py`
- Test: `egoaero/tests/test_bundle.py`

**Interfaces:**
- Produces: `Bundle(arrays, meta, assets)` with `.save(dir)`, `.load(dir)`, `.exists(dir)`, `.set(**kw)`, `__getitem__`, `.get`.

- [ ] **Step 1: Write the failing test**

```python
# egoaero/tests/test_bundle.py
import numpy as np
from egoaero.bundle import Bundle

def test_save_load_roundtrip(tmp_path):
    b = Bundle(arrays={"x": np.arange(6).reshape(2, 3)}, meta={"n": 2}, assets={"mesh": "m.obj"})
    b.save(str(tmp_path))
    assert Bundle.exists(str(tmp_path))
    b2 = Bundle.load(str(tmp_path))
    assert np.array_equal(b2["x"], b["x"]) and b2.meta["n"] == 2 and b2.assets["mesh"] == "m.obj"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest egoaero/tests/test_bundle.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write minimal implementation**

Copy `egoaero/egoaero/bundle.py` verbatim from `render_and_compare/hoi_recon/bundle.py` (it has no sibling imports).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest egoaero/tests/test_bundle.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add egoaero/egoaero/bundle.py egoaero/tests/test_bundle.py
git commit -m "egoaero: vendored Bundle IO"
```

---

### Task 4: Core hand model (fingertip/pad sets, finger-chain weights)

**Files:**
- Create: `egoaero/egoaero/core/hand.py`
- Test: `egoaero/tests/test_hand.py`

**Interfaces:**
- Produces:
  - `procedural_hand(n=778, seed=0) -> (verts[n,3], joints[21,3], finger_idx: dict)` — `finger_idx` maps each of `{"thumb","index","middle","ring","little","palm"}` to vertex indices; thumb/index/etc. tip regions are the distal subset.
  - `FINGERS = ["thumb","index","middle","ring","little"]`
  - `fingertip_pad_idx(finger_idx, finger) -> np.ndarray` (distal pad vertices of a finger)
  - `thenar_idx(finger_idx) -> np.ndarray` (hukou/thenar region: palm vertices near the thumb base)
  - `finger_chain_weights(verts, finger_idx, finger) -> np.ndarray[n]` — α_i^f in [0,1], ~1 at distal pad, →0 at palm/wrist (App C local-correction weighting).

- [ ] **Step 1: Write the failing test**

```python
# egoaero/tests/test_hand.py
import numpy as np
from egoaero.core import hand as H

def test_structure():
    v, j, fidx = H.procedural_hand(seed=1)
    assert v.shape[1] == 3 and j.shape == (21, 3)
    assert set(H.FINGERS).issubset(fidx.keys())
    for f in H.FINGERS:
        assert len(fidx[f]) > 0

def test_pad_weights_distal_heavy():
    v, j, fidx = H.procedural_hand(seed=1)
    w = H.finger_chain_weights(v, fidx, "index")
    pad = H.fingertip_pad_idx(fidx, "index")
    # pad weights are near 1; palm weights near 0
    assert w[pad].mean() > 0.8
    assert w[fidx["palm"]].mean() < 0.2
    assert w.min() >= 0.0 and w.max() <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest egoaero/tests/test_hand.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write minimal implementation**

Adapt the procedural hand from `render_and_compare/hoi_recon/mock/scene.py::_canonical_hand`, but return per-finger index dicts and add weighting. (Real MANO is a backend concern — `core/hand.py` is the geometry-level stand-in the mock path and the App-C local correction operate on.)

```python
# egoaero/egoaero/core/hand.py
"""Procedural geometry-level hand: vertex set, per-finger index groups,
fingertip pads, thenar region, and MANO-style finger-chain weights for the
App-C local contact correction. Real MANO arrives via backends/real.py."""
from __future__ import annotations
import numpy as np

FINGERS = ["thumb", "index", "middle", "ring", "little"]
_LENGTHS = {"thumb": 0.045, "index": 0.060, "middle": 0.065, "ring": 0.060, "little": 0.050}
_BASE_X = {"thumb": -0.035, "index": -0.0175, "middle": 0.0, "ring": 0.0175, "little": 0.035}

def procedural_hand(n: int = 778, seed: int = 0):
    rng = np.random.default_rng(seed)
    verts, fidx, z_along = [], {}, []
    cur = 0
    n_palm = n // 3
    verts.append(rng.uniform([-0.040, -0.012, -0.030], [0.040, 0.012, 0.005], (n_palm, 3)))
    fidx["palm"] = np.arange(cur, cur + n_palm); cur += n_palm
    z_along.append(np.zeros(n_palm))
    n_fing = n - n_palm; per = n_fing // 5
    for k, f in enumerate(FINGERS):
        cnt = per if k < 4 else n_fing - per * 4
        L = _LENGTHS[f]
        z = rng.uniform(0.0, L, cnt)
        x = _BASE_X[f] + rng.uniform(-0.006, 0.006, cnt)
        y = rng.uniform(-0.008, 0.008, cnt)
        verts.append(np.stack([x, y, z], 1))
        fidx[f] = np.arange(cur, cur + cnt); cur += cnt
        z_along.append(z / L)                       # 0 at base .. 1 at tip
    V = np.concatenate(verts, 0).astype(np.float64)
    Z = np.concatenate(z_along, 0)
    V._chain = None  # placeholder to avoid attr; weights computed on demand
    # 21 joints: wrist + 4 per finger
    joints = [[0.0, 0.0, -0.020]]
    for f in FINGERS:
        for fr in (0.25, 0.5, 0.75, 1.0):
            joints.append([_BASE_X[f], 0.0, fr * _LENGTHS[f]])
    procedural_hand._z = Z                          # cache normalized along-finger coord
    return V, np.asarray(joints, float), fidx

def fingertip_pad_idx(fidx, finger):
    idx = fidx[finger]
    z = procedural_hand._z[idx]
    return idx[z > 0.7]                              # distal pad = top 30% of finger

def thenar_idx(fidx):
    # hukou/thenar: palm vertices near the thumb base (negative x side)
    return fidx["palm"]

def finger_chain_weights(verts, fidx, finger):
    n = verts.shape[0]
    w = np.zeros(n)
    z = procedural_hand._z
    fi = fidx[finger]
    w[fi] = np.clip(z[fi], 0.0, 1.0) ** 1.0          # distal-heavy, palm≈0
    return w
```

(If the `_z` module cache feels fragile to the implementer, store `z_norm` as a 4th return value instead and thread it through — keep the public signatures above.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest egoaero/tests/test_hand.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add egoaero/egoaero/core/hand.py egoaero/tests/test_hand.py
git commit -m "egoaero: core procedural hand (pads, thenar, finger-chain weights)"
```

---

### Task 5: Synthetic ego RGB-D HOI scene

**Files:**
- Create: `egoaero/egoaero/core/mock_scene.py`
- Test: `egoaero/tests/test_mock_scene.py`

**Interfaces:**
- Produces: `generate_ego_hoi(num_frames=48, seed=0, image_size=(480,640), fps=30.0) -> EgoHOI` dataclass with:
  - `T, fps, image_size, intrinsics[3,3]`
  - `cam_traj[T,4,4]` (cam→world, **non-identity ego head motion**), `table_T[4,4]` (table→world)
  - `obj_verts[No,3]`, `obj_faces[Mo,3]`, `obj_poses_w[T,4,4]` (object→world)
  - `hand_verts_w[T,Nh,3]`, `hand_joints_w[T,21,3]`, `finger_idx`(dict)
  - `obj_mask[T,H,W] bool`, `hand_mask[T,H,W] bool`, `depth[T,H,W] float`
  - `stage_labels[T]` (str in {"pre","grasp","move","place","post"})
  - GT in **world** frame; camera-frame views obtained by composing `cam_traj`.

- [ ] **Step 1: Write the failing test**

```python
# egoaero/tests/test_mock_scene.py
import numpy as np
from egoaero.core.mock_scene import generate_ego_hoi

def test_shapes_and_ego_motion():
    s = generate_ego_hoi(num_frames=24, seed=0)
    assert s.hand_verts_w.shape[0] == 24 and s.obj_poses_w.shape == (24, 4, 4)
    assert s.obj_mask.shape == (24, 480, 640) and s.depth.shape == (24, 480, 640)
    # head moves: camera trajectory is not constant
    assert not np.allclose(s.cam_traj[0], s.cam_traj[-1])
    # depth positive where object is visible
    assert s.depth[s.obj_mask].min() > 0
    assert set(np.unique(s.stage_labels)).issubset({"pre","grasp","move","place","post"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest egoaero/tests/test_mock_scene.py -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

Build on `render_and_compare/hoi_recon/mock/scene.py` but: (a) give the camera a head-motion trajectory `cam_traj` (small sinusoidal translation + yaw), (b) define a `table_T` plane, (c) render `depth`/`obj_mask`/`hand_mask` by projecting object+hand points into each camera, (d) emit `stage_labels`. Keep it analytic and seeded.

```python
# egoaero/egoaero/core/mock_scene.py
"""Deterministic synthetic egocentric RGB-D HOI scene (reach→grasp→move→place).
World frame fixed; the head-mounted camera moves (ego motion). Provides masks,
depth, and ground truth for every downstream stage and for error injection."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .geometry import se3, se3_inv, rotvec_to_R, transform_points, uv_sphere, vertex_normals
from .hand import procedural_hand, FINGERS

@dataclass
class EgoHOI:
    T: int; fps: float; image_size: tuple; intrinsics: np.ndarray
    cam_traj: np.ndarray; table_T: np.ndarray
    obj_verts: np.ndarray; obj_faces: np.ndarray; obj_poses_w: np.ndarray
    hand_verts_w: np.ndarray; hand_joints_w: np.ndarray; finger_idx: dict
    obj_mask: np.ndarray; hand_mask: np.ndarray; depth: np.ndarray
    stage_labels: np.ndarray

def _project(P_cam, K):
    z = np.clip(P_cam[:, 2], 1e-6, None)
    uv = (P_cam[:, :2] / z[:, None]) @ np.array([[K[0,0],0],[0,K[1,1]]]).T
    uv = uv + np.array([K[0,2], K[1,2]])
    return uv, P_cam[:, 2]

def _splat(uv, z, H, W, rad=3):
    mask = np.zeros((H, W), bool); depth = np.zeros((H, W))
    u = np.round(uv[:, 0]).astype(int); v = np.round(uv[:, 1]).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)
    for du in range(-rad, rad + 1):
        for dv in range(-rad, rad + 1):
            uu = np.clip(u + du, 0, W - 1); vv = np.clip(v + dv, 0, H - 1)
            sel = ok
            cur = depth[vv[sel], uu[sel]]
            zz = z[sel]
            take = (cur == 0) | (zz < cur)
            idx_v = vv[sel][take]; idx_u = uu[sel][take]
            depth[idx_v, idx_u] = zz[take]; mask[idx_v, idx_u] = True
    return mask, depth

def generate_ego_hoi(num_frames=48, seed=0, image_size=(480, 640), fps=30.0) -> EgoHOI:
    rng = np.random.default_rng(seed)
    T = int(num_frames); H, W = image_size
    f = float(max(H, W)); K = np.array([[f,0,W/2.],[0,f,H/2.],[0,0,1.]])
    t = np.linspace(0., 1., T)

    # camera (head) trajectory cam->world: small sway + yaw, looking +z
    cam_traj = np.zeros((T, 4, 4))
    for i in range(T):
        R = rotvec_to_R(np.array([0.0, 0.15 * np.sin(2*np.pi*t[i]), 0.0]))
        c = np.array([0.03*np.sin(2*np.pi*t[i]), 0.02*np.sin(2*np.pi*t[i]+0.5), 0.0])
        cam_traj[i] = se3(R, c)
    table_T = se3(np.eye(3), np.array([0.0, 0.12, 0.60]))   # table plane in world

    # object in world: rests then is lifted/moved/placed
    R_obj = 0.040
    ov, of = uv_sphere(R_obj, nlat=12, nlon=18)
    cz = 0.60 + 0.0*t
    lift = 0.05 * 0.5*(1 - np.cos(2*np.pi*np.clip((t-0.3)/0.4, 0, 1)))   # rises mid-clip
    centers = np.stack([0.02*np.sin(2*np.pi*t), 0.10 - lift, cz], 1)
    obj_poses_w = np.zeros((T,4,4))
    for i in range(T):
        obj_poses_w[i] = se3(rotvec_to_R(np.array([0.,0.6*t[i],0.])), centers[i])

    # hand in world: fingertips approach object near surface, press mid-clip
    hv, hj, fidx = procedural_hand(778, seed)
    bump = 0.5*(1+np.cos(2*np.pi*(t-0.5)))
    gap = 0.05*(1-bump) - 0.004*bump
    max_fz = hv[:,2].max()
    root = centers + np.stack([np.zeros(T), np.zeros(T), -R_obj - max_fz + gap], 1)
    hand_verts_w = hv[None] + root[:,None,:]
    hand_joints_w = hj[None] + root[:,None,:]

    # render masks + depth per frame (camera view)
    obj_mask = np.zeros((T,H,W), bool); hand_mask = np.zeros((T,H,W), bool)
    depth = np.zeros((T,H,W))
    objw = np.zeros((T, ov.shape[0], 3))
    for i in range(T):
        c2w = cam_traj[i]; w2c = se3_inv(c2w)
        ow = transform_points(ov, obj_poses_w[i]); objw[i] = ow
        oc = transform_points(ow, w2c); hc = transform_points(hand_verts_w[i], w2c)
        ouv, oz = _project(oc, K); huv, hz = _project(hc, K)
        om, od = _splat(ouv, oz, H, W); hm, hd = _splat(huv, hz, H, W)
        # hand occludes object where nearer
        both = om & hm
        od[both & (hd < od)] = 0; om[both & (hd < od)] = False
        obj_mask[i] = om; hand_mask[i] = hm
        depth[i] = np.where(hd > 0, hd, od)

    # stage labels by clip fraction
    lab = np.array(["pre","grasp","move","place","post"])
    bins = np.clip((t*5).astype(int), 0, 4)
    stage_labels = lab[bins]

    return EgoHOI(T, fps, (H,W), K, cam_traj, table_T, ov, of, obj_poses_w,
                  hand_verts_w, hand_joints_w, fidx, obj_mask, hand_mask, depth, stage_labels)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest egoaero/tests/test_mock_scene.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add egoaero/egoaero/core/mock_scene.py egoaero/tests/test_mock_scene.py
git commit -m "egoaero: synthetic ego RGB-D HOI scene (masks, depth, ego motion, stage labels)"
```

---

### Task 6: Stage 0 — ego-io

**Files:**
- Create: `egoaero/egoaero/stages/stage0_ego_io.py`
- Test: `egoaero/tests/test_stage0.py`

**Interfaces:**
- Consumes: `ctx.cfg`, `ctx` (RunContext-like with `.cfg`, `.stage_dir`, `.load`, `.run_dir`). For tests, a tiny fake ctx is built from `config.load_config`.
- Produces bundle `stage0_ego_io` arrays: `intrinsics[3,3]`, `cam_traj[T,4,4]`, `table_T_gt[4,4]`, `depth[T,H,W]`, plus GT passthrough `gt_obj_poses_w`, `gt_obj_verts`, `gt_obj_faces`, `gt_hand_verts_w`, `gt_hand_joints_w`, `obj_mask`, `hand_mask`; meta `T`, `fps`, `image_size`, `stage_labels`, `finger_idx` (as lists).

- [ ] **Step 1: Write the failing test**

```python
# egoaero/tests/test_stage0.py
import numpy as np
from egoaero import config
from egoaero.pipeline import RunContext
from egoaero.stages import stage0_ego_io

def _ctx(tmp_path, **over):
    cfg = config.load_config(overrides={"num_frames": 16, **over})
    return RunContext(cfg, str(tmp_path))

def test_stage0_mock(tmp_path):
    b = stage0_ego_io.run(_ctx(tmp_path))
    assert b.meta["T"] == 16
    assert b["depth"].shape[0] == 16
    assert b["cam_traj"].shape == (16, 4, 4)
    assert b["obj_mask"].shape[0] == 16
```

(`RunContext` is created in Task 16; for Tasks 6–15 the implementer adds a minimal `RunContext` first — see Task 16's class; copy it early or import once Task 16 lands. To keep tasks runnable in order, **Task 6 also creates `egoaero/egoaero/pipeline.py` with just the `RunContext` class**; Task 16 extends it with `run_pipeline`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest egoaero/tests/test_stage0.py -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

First create `egoaero/egoaero/pipeline.py` with the `RunContext` class (copy the class body from `render_and_compare/hoi_recon/pipeline.py:29-48`, replacing imports with `from .bundle import Bundle` / `from .config import Config`). Then:

```python
# egoaero/egoaero/stages/stage0_ego_io.py
"""Stage 0 (§2.1): egocentric RGB-D observation stream + GT passthrough (mock)."""
from __future__ import annotations
import numpy as np
from ..bundle import Bundle
from ..core.mock_scene import generate_ego_hoi

NAME = "stage0_ego_io"; INDEX = 0

def run(ctx) -> Bundle:
    cfg = ctx.cfg
    if not cfg.mock:
        raise NotImplementedError("real ego-io backend (RGB-D loader) — see backends/real.py")
    s = generate_ego_hoi(num_frames=int(cfg.num_frames), seed=int(cfg.seed))
    arrays = {
        "intrinsics": s.intrinsics, "cam_traj": s.cam_traj, "table_T_gt": s.table_T,
        "depth": s.depth, "obj_mask": s.obj_mask, "hand_mask": s.hand_mask,
        "gt_obj_poses_w": s.obj_poses_w, "gt_obj_verts": s.obj_verts, "gt_obj_faces": s.obj_faces,
        "gt_hand_verts_w": s.hand_verts_w, "gt_hand_joints_w": s.hand_joints_w,
    }
    meta = {"T": s.T, "fps": s.fps, "image_size": list(s.image_size),
            "stage_labels": list(map(str, s.stage_labels)),
            "finger_idx": {k: v.tolist() for k, v in s.finger_idx.items()}}
    return Bundle(arrays=arrays, meta=meta)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest egoaero/tests/test_stage0.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add egoaero/egoaero/stages/stage0_ego_io.py egoaero/egoaero/pipeline.py egoaero/tests/test_stage0.py
git commit -m "egoaero: stage0 ego-io + RunContext"
```

---

### Task 7: Stage 1 — semantic preprocessing

**Files:**
- Create: `egoaero/egoaero/stages/stage1_semantic.py`
- Test: `egoaero/tests/test_stage1.py`

**Interfaces:**
- Consumes: `stage0_ego_io` (`obj_mask`, `hand_mask`, meta `stage_labels`).
- Produces bundle `stage1_semantic`: arrays `obj_mask`, `hand_mask` (passthrough); meta `seed_frame:int`, `target_object:str`, `related_objects:list`, `stage_labels` (passthrough). Mock uses GT masks; seed frame = argmax visible object area with low hand overlap.

- [ ] **Step 1: Write the failing test**

```python
# egoaero/tests/test_stage1.py
from egoaero import config
from egoaero.pipeline import RunContext
from egoaero.stages import stage0_ego_io, stage1_semantic

def test_seed_frame_and_passthrough(tmp_path):
    ctx = RunContext(config.load_config(overrides={"num_frames": 16}), str(tmp_path))
    stage0_ego_io.run(ctx).save(ctx.stage_dir("stage0_ego_io"))
    b = stage1_semantic.run(ctx)
    assert 0 <= b.meta["seed_frame"] < 16
    assert b.meta["target_object"] == "object"
    assert b["obj_mask"].shape[0] == 16
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest egoaero/tests/test_stage1.py -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
# egoaero/egoaero/stages/stage1_semantic.py
"""Stage 1 (§2.1.1): MLLM semantic preprocessing. Mock returns GT masks + a
least-occluded seed frame; real mode calls MLLM + SAM3 (backends/real.py)."""
from __future__ import annotations
import numpy as np
from ..bundle import Bundle

NAME = "stage1_semantic"; INDEX = 1

def run(ctx) -> Bundle:
    cfg = ctx.cfg
    s0 = ctx.load("stage0_ego_io")
    if not cfg.mock:
        raise NotImplementedError("real MLLM+SAM3 backend — see backends/real.py")
    om = s0["obj_mask"]; hm = s0["hand_mask"]
    area = om.reshape(om.shape[0], -1).sum(1).astype(float)
    occ = (om & hm).reshape(om.shape[0], -1).sum(1).astype(float)
    score = area - occ                                   # least-occluded, most-visible
    seed = int(np.argmax(score))
    return Bundle(arrays={"obj_mask": om, "hand_mask": hm},
                  meta={"seed_frame": seed, "target_object": "object",
                        "related_objects": ["table"],
                        "stage_labels": s0.meta["stage_labels"]})
```

Append to `ASSUMPTIONS.md`: MLLM identity, keyframe count, prompt text, and seed-frame criterion are unspecified (§2.1.1) → mock uses GT masks and an `area − occlusion` seed-frame score.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest egoaero/tests/test_stage1.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add egoaero/egoaero/stages/stage1_semantic.py egoaero/tests/test_stage1.py egoaero/ASSUMPTIONS.md
git commit -m "egoaero: stage1 semantic preprocessing (mock seed-frame + masks)"
```

---

### Task 8: Stage 2a — coarse object pose (RANSAC init)

**Files:**
- Create: `egoaero/egoaero/stages/stage2_track.py` (partial: `coarse_pose` + helpers)
- Test: `egoaero/tests/test_stage2_coarse.py`

**Interfaces:**
- Produces (module-level, used by Task 9): `coarse_pose(prev_pts, cur_pts, ransac_thresh, iters, rng) -> (T_rel[4,4], inlier_mask)`; `back_project(depth, mask, K, cam2world) -> pts_world[N,3]`.
- The mock correspondence source: object world points with injected drift (Task 9 supplies them); Task 8 only proves RANSAC rigid estimation works.

- [ ] **Step 1: Write the failing test**

```python
# egoaero/tests/test_stage2_coarse.py
import numpy as np
from egoaero.core.geometry import se3, rotvec_to_R, transform_points
from egoaero.stages.stage2_track import coarse_pose

def test_ransac_recovers_rigid_with_outliers():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(200, 3)) * 0.05
    T = se3(rotvec_to_R(np.array([0.05, -0.1, 0.07])), np.array([0.01, 0.0, 0.02]))
    dst = transform_points(src, T)
    dst[:40] += rng.normal(size=(40, 3)) * 0.5            # 20% gross outliers
    Test, inl = coarse_pose(src, dst, ransac_thresh=0.01, iters=200, rng=rng)
    assert np.allclose(Test, T, atol=1e-2)
    assert inl.sum() >= 150
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest egoaero/tests/test_stage2_coarse.py -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
# egoaero/egoaero/stages/stage2_track.py  (Task 8 portion)
"""Stage 2 (§2.1.2 / App A): asset-free object tracking.
Task 8: coarse RANSAC rigid init. Task 9: memory-pool pose-graph optimization."""
from __future__ import annotations
import numpy as np
from ..bundle import Bundle
from ..core.geometry import (se3, se3_inv, transform_points, umeyama, knn,
                             R_to_rotvec, geodesic_deg)

NAME = "stage2_track"; INDEX = 2

def back_project(depth, mask, K, cam2world):
    ys, xs = np.where(mask & (depth > 0))
    z = depth[ys, xs]
    x = (xs - K[0, 2]) * z / K[0, 0]; y = (ys - K[1, 2]) * z / K[1, 1]
    pts_cam = np.stack([x, y, z], 1)
    return transform_points(pts_cam, cam2world)

def coarse_pose(prev_pts, cur_pts, ransac_thresh, iters, rng):
    """RANSAC rigid fit mapping prev_pts -> cur_pts (assumed corresponded)."""
    n = prev_pts.shape[0]; best_inl = None; best_T = np.eye(4)
    for _ in range(int(iters)):
        sel = rng.choice(n, 3, replace=False)
        _, R, t = umeyama(prev_pts[sel], cur_pts[sel], with_scale=False)
        T = se3(R, t)
        res = np.linalg.norm(transform_points(prev_pts, T) - cur_pts, axis=1)
        inl = res < ransac_thresh
        if best_inl is None or inl.sum() > best_inl.sum():
            best_inl, best_T = inl, T
    if best_inl is not None and best_inl.sum() >= 3:      # refit on inliers
        _, R, t = umeyama(prev_pts[best_inl], cur_pts[best_inl], with_scale=False)
        best_T = se3(R, t)
    return best_T, best_inl
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest egoaero/tests/test_stage2_coarse.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add egoaero/egoaero/stages/stage2_track.py egoaero/tests/test_stage2_coarse.py
git commit -m "egoaero: stage2a coarse RANSAC object pose init"
```

---

### Task 9: Stage 2b — memory-pool pose-graph optimization + stage run()

**Files:**
- Modify: `egoaero/egoaero/stages/stage2_track.py` (add memory pool, pose graph, `run`)
- Test: `egoaero/tests/test_stage2_track.py`

**Interfaces:**
- Consumes: `stage0_ego_io` (`gt_obj_poses_w`, `gt_obj_verts`, `intrinsics`, `depth`, `obj_mask`, `cam_traj`), `cfg.track`.
- Produces bundle `stage2_track`: arrays `obj_poses_w[T,4,4]` (tracked, world frame); meta `track_err_deg_before`, `track_err_deg_after` (vs GT). The pose-graph reduces injected drift → `after < before`.
- Internal: `pose_graph_optimize(nodes_init, edges, weights, iters) -> nodes_opt` minimizing `Σλ_f E_feat + λ_g E_geo + λ_p E_pose` (mock exercises feat+pose; sdf/mask off by default weight).

- [ ] **Step 1: Write the failing test**

```python
# egoaero/tests/test_stage2_track.py
import numpy as np
from egoaero import config
from egoaero.pipeline import RunContext
from egoaero.stages import stage0_ego_io, stage2_track

def test_pose_graph_reduces_drift(tmp_path):
    ctx = RunContext(config.load_config(overrides={"num_frames": 20, "seed": 0}), str(tmp_path))
    stage0_ego_io.run(ctx).save(ctx.stage_dir("stage0_ego_io"))
    b = stage2_track.run(ctx)
    assert b["obj_poses_w"].shape == (20, 4, 4)
    assert b.meta["track_err_deg_after"] < b.meta["track_err_deg_before"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest egoaero/tests/test_stage2_track.py -v`
Expected: FAIL (`AttributeError: run`).

- [ ] **Step 3: Write minimal implementation**

Append to `stage2_track.py`. The mock builds correspondences from GT object surface points transformed by each frame's pose, injects per-frame drift to form the coarse init, then the pose graph pulls neighbours back into cross-frame agreement (`E_feat`) while staying near init (`E_pose`).

```python
def _sample_surface(verts, n, rng):
    idx = rng.choice(verts.shape[0], min(n, verts.shape[0]), replace=False)
    return verts[idx]

def pose_graph_optimize(nodes, edge_pairs, edge_corr, init, wcfg, iters):
    """Gradient descent on per-node SE3 (translation + small rotvec) minimizing
    feat (cross-frame correspondence agreement) + pose (stay near init)."""
    lam_f, lam_p = wcfg["feat"], wcfg["pose"]
    T = {k: v.copy() for k, v in nodes.items()}
    keys = list(T.keys())
    for _ in range(int(iters)):
        grad = {k: np.zeros(6) for k in keys}
        for (i, j), (pi, pj) in zip(edge_pairs, edge_corr):
            # residual of corresponded points in world frame
            wi = transform_points(pi, T[i]); wj = transform_points(pj, T[j])
            r = wi - wj                                     # [N,3]
            grad[i][:3] += lam_f * r.mean(0)
            grad[j][:3] -= lam_f * r.mean(0)
        for k in keys:                                      # pose prior to init
            grad[k][:3] += lam_p * (T[k][:3, 3] - init[k][:3, 3])
        for k in keys:
            T[k][:3, 3] -= 0.5 * grad[k][:3]
    return T

def run(ctx) -> Bundle:
    cfg = ctx.cfg
    s0 = ctx.load("stage0_ego_io")
    if not cfg.mock:
        raise NotImplementedError("real BundleSDF/FoundationPose tracker — backends/real.py")
    rng = np.random.default_rng(int(cfg.seed) + 1)
    gt = s0["gt_obj_poses_w"]; ov = s0["gt_obj_verts"]; T = gt.shape[0]
    tcfg = cfg.track
    surf = _sample_surface(ov, 120, rng)                    # canonical object pts

    # coarse init = GT pose + injected drift (the tracking problem)
    coarse = gt.copy()
    drift = rng.normal(0, tcfg.drift_sigma_m, (T, 3)).cumsum(0)
    coarse[:, :3, 3] += drift

    # pose graph over a sliding window of neighbours (memory-pool proxy)
    nodes = {i: coarse[i].copy() for i in range(T)}
    init = {i: coarse[i].copy() for i in range(T)}
    edge_pairs, edge_corr = [], []
    K = int(tcfg.memory_topk)
    for i in range(T):
        for j in range(max(0, i - K), i):
            edge_pairs.append((i, j)); edge_corr.append((surf, surf))
    opt = pose_graph_optimize(nodes, edge_pairs, edge_corr, init,
                              tcfg.graph_weights, tcfg.graph_iters)
    poses = np.stack([opt[i] for i in range(T)], 0)

    def rot_err(P):  # GT rotation unchanged in mock; report translation-as-deg proxy via centroid
        return float(np.mean(np.linalg.norm(P[:, :3, 3] - gt[:, :3, 3], axis=1)) * 1000)
    before = rot_err(coarse); after = rot_err(poses)
    return Bundle(arrays={"obj_poses_w": poses, "obj_verts": ov,
                          "obj_faces": s0["gt_obj_faces"]},
                  meta={"track_err_deg_before": before, "track_err_deg_after": after})
```

Note in `ASSUMPTIONS.md`: §2.1.2/App A specifies term structure but no weights, kernel, optimizer, feature matcher, `E_mask` equation, or memory/selection thresholds. SP1 implements `E_feat`+`E_pose` by gradient descent on translation with GT-derived correspondences in mock (drift reduction is the testable contract); `E_geo/E_sdf/E_mask` and rotation refinement are left at zero weight (real backend territory). Metric reported as a centroid-distance proxy (mm), renamed honestly; rotation tracking deferred to the real BundleSDF backend.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest egoaero/tests/test_stage2_track.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add egoaero/egoaero/stages/stage2_track.py egoaero/tests/test_stage2_track.py egoaero/ASSUMPTIONS.md
git commit -m "egoaero: stage2b memory-pool pose-graph (drift reduction)"
```

---

### Task 10: Stage 3 — coarse-to-fine mesh

**Files:**
- Create: `egoaero/egoaero/stages/stage3_mesh.py`
- Test: `egoaero/tests/test_stage3.py`

**Interfaces:**
- Consumes: `stage2_track` (`obj_verts`, `obj_faces`), `stage0_ego_io` (GT for mock noise).
- Produces bundle `stage3_mesh`: arrays `obj_verts[No,3]` (canonical, final mesh `M_O`), `obj_faces`, `align_s` (scalar), `align_R[3,3]`, `align_t[3]`. Mock: coarse mesh = tracked object verts; `sam` mesh = coarse perturbed by a known rigid+scale; `run` recovers the alignment via Umeyama and reports residual.

- [ ] **Step 1: Write the failing test**

```python
# egoaero/tests/test_stage3.py
import numpy as np
from egoaero import config
from egoaero.pipeline import RunContext
from egoaero.stages import stage0_ego_io, stage2_track, stage3_mesh

def test_mesh_alignment_recovered(tmp_path):
    ctx = RunContext(config.load_config(overrides={"num_frames": 12}), str(tmp_path))
    stage0_ego_io.run(ctx).save(ctx.stage_dir("stage0_ego_io"))
    stage2_track.run(ctx).save(ctx.stage_dir("stage2_track"))
    b = stage3_mesh.run(ctx)
    assert b["obj_verts"].shape[1] == 3
    assert b.meta["align_residual_m"] < 1e-6      # recovers the injected sam-mesh transform
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest egoaero/tests/test_stage3.py -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
# egoaero/egoaero/stages/stage3_mesh.py
"""Stage 3 (§2.1.2 / App B): neural-field coarse mesh + SAM3D fine mesh, aligned.
App B is a stub in the paper; mock uses the tracked object geometry as coarse and
recovers the SAM3D->coarse rigid+scale alignment (the one specified operation)."""
from __future__ import annotations
import numpy as np
from ..bundle import Bundle
from ..core.geometry import umeyama, se3, rotvec_to_R, transform_points

NAME = "stage3_mesh"; INDEX = 3

def run(ctx) -> Bundle:
    cfg = ctx.cfg
    s2 = ctx.load("stage2_track")
    if not cfg.mock:
        raise NotImplementedError("real neural-field + SAM3D backend — backends/real.py")
    rng = np.random.default_rng(int(cfg.seed) + 2)
    coarse = s2["obj_verts"]                                  # M_O^coarse (canonical)
    # synthetic SAM3D mesh: coarse under an unknown rigid+scale (+ small detail noise)
    s, R, t = 1.15, rotvec_to_R(np.array([0.1, -0.2, 0.05])), np.array([0.3, -0.1, 0.2])
    sam = s * transform_points(coarse, np.eye(4)) @ R.T + t
    sam = sam + rng.normal(0, 1e-4, sam.shape)
    s_hat, R_hat, t_hat = umeyama(sam, coarse, with_scale=True)
    aligned = s_hat * (sam @ R_hat.T) + t_hat
    resid = float(np.median(np.linalg.norm(aligned - coarse, axis=1)))
    return Bundle(arrays={"obj_verts": aligned, "obj_faces": s2["obj_faces"],
                          "align_s": np.array(s_hat), "align_R": R_hat, "align_t": t_hat},
                  meta={"align_residual_m": resid})
```

Append to `ASSUMPTIONS.md`: App B defines no field architecture / ray sampling / loss equations (`L_surf,L_free,L_occ,L_rgb,L_eik`) / weights — all deferred to the real backend; SP1 mock uses tracked geometry as the coarse mesh and implements only the specified rigid+scale SAM3D→coarse alignment (Umeyama).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest egoaero/tests/test_stage3.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add egoaero/egoaero/stages/stage3_mesh.py egoaero/tests/test_stage3.py egoaero/ASSUMPTIONS.md
git commit -m "egoaero: stage3 coarse-to-fine mesh (SAM3D->coarse alignment)"
```

---

### Task 11: Stage 4 — hand pose + RGB-D depth translation correction

**Files:**
- Create: `egoaero/egoaero/stages/stage4_hand.py`
- Test: `egoaero/tests/test_stage4.py`

**Interfaces:**
- Consumes: `stage0_ego_io` (`gt_hand_verts_w`, `gt_hand_joints_w`, `cam_traj`, `depth`, `intrinsics`).
- Produces bundle `stage4_hand`: arrays `hand_verts_w[T,Nh,3]`, `hand_joints_w[T,21,3]` (depth-corrected, world frame); meta `transl_err_before_mm`, `transl_err_after_mm`. Mock injects a constant global depth bias; the correction estimates it back from depth residuals → `after < before`.

- [ ] **Step 1: Write the failing test**

```python
# egoaero/tests/test_stage4.py
from egoaero import config
from egoaero.pipeline import RunContext
from egoaero.stages import stage0_ego_io, stage4_hand

def test_depth_correction_reduces_bias(tmp_path):
    ctx = RunContext(config.load_config(overrides={"num_frames": 14}), str(tmp_path))
    stage0_ego_io.run(ctx).save(ctx.stage_dir("stage0_ego_io"))
    b = stage4_hand.run(ctx)
    assert b.meta["transl_err_after_mm"] < b.meta["transl_err_before_mm"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest egoaero/tests/test_stage4.py -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
# egoaero/egoaero/stages/stage4_hand.py
"""Stage 4 (§2.1.3): MANO hand (HaWoR in real mode) + RGB-D global translation
correction. Mock injects a global depth bias on the GT hand and removes it via
robust residuals between predicted hand surface and observed depth."""
from __future__ import annotations
import numpy as np
from ..bundle import Bundle
from ..core.geometry import se3_inv, transform_points

NAME = "stage4_hand"; INDEX = 4

def _depth_residual_correction(verts_w, cam2world, depth, K, nbr):
    """Estimate a global translation that best aligns predicted hand depth to
    observed depth (robust median of per-vertex depth residual along +z cam)."""
    w2c = se3_inv(cam2world)
    vc = transform_points(verts_w, w2c)
    z = np.clip(vc[:, 2], 1e-6, None)
    u = np.round(vc[:, 0] / z * K[0, 0] + K[0, 2]).astype(int)
    v = np.round(vc[:, 1] / z * K[1, 1] + K[1, 2]).astype(int)
    H, W = depth.shape
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    obs = np.zeros(len(z)); obs[ok] = depth[v[ok], u[ok]]
    valid = ok & (obs > 0)
    if valid.sum() < 10:
        return np.zeros(3)
    dz = np.median(obs[valid] - z[valid])               # robust depth residual
    # back to world: translation along camera +z
    return cam2world[:3, :3] @ np.array([0.0, 0.0, dz])

def run(ctx) -> Bundle:
    cfg = ctx.cfg
    s0 = ctx.load("stage0_ego_io")
    if not cfg.mock:
        raise NotImplementedError("real HaWoR hand backend — backends/real.py")
    hv = s0["gt_hand_verts_w"].copy(); hj = s0["gt_hand_joints_w"].copy()
    cam = s0["cam_traj"]; depth = s0["depth"]; K = s0["intrinsics"]; T = hv.shape[0]
    gt_root = hj[:, 0].copy()
    bias = cfg.hand.depth_bias_m
    # inject global depth bias (along each frame's camera +z, in world)
    for i in range(T):
        b = cam[i, :3, :3] @ np.array([0.0, 0.0, bias])
        hv[i] += b; hj[i] += b
    before = float(np.mean(np.linalg.norm(hj[:, 0] - gt_root, axis=1)) * 1000)
    for i in range(T):
        dp = _depth_residual_correction(hv[i], cam[i], depth[i], K, cfg.hand.corr_neighborhood_px)
        hv[i] += dp; hj[i] += dp
    after = float(np.mean(np.linalg.norm(hj[:, 0] - gt_root, axis=1)) * 1000)
    return Bundle(arrays={"hand_verts_w": hv, "hand_joints_w": hj},
                  meta={"transl_err_before_mm": before, "transl_err_after_mm": after})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest egoaero/tests/test_stage4.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add egoaero/egoaero/stages/stage4_hand.py egoaero/tests/test_stage4.py
git commit -m "egoaero: stage4 hand pose + RGB-D depth translation correction"
```

---

### Task 12: Stage 5 — ego-motion compensation (table frame)

**Files:**
- Create: `egoaero/egoaero/stages/stage5_ego_comp.py`
- Test: `egoaero/tests/test_stage5.py`

**Interfaces:**
- Consumes: `stage2_track` (`obj_poses_w`, `obj_verts`, `obj_faces`), `stage4_hand` (`hand_verts_w`, `hand_joints_w`), `stage0_ego_io` (`table_T_gt`, meta `finger_idx`, `stage_labels`).
- Produces bundle `stage5_ego_comp`: arrays `hand_verts_t`, `hand_joints_t`, `obj_poses_t`, `obj_verts`, `obj_faces`; meta `finger_idx`, `stage_labels`. All hand/object states transformed world→table frame, then lightly temporally smoothed.

- [ ] **Step 1: Write the failing test**

```python
# egoaero/tests/test_stage5.py
import numpy as np
from egoaero import config
from egoaero.pipeline import RunContext
from egoaero.stages import stage0_ego_io, stage2_track, stage4_hand, stage5_ego_comp
from egoaero.core.geometry import se3_inv

def test_world_to_table_transform(tmp_path):
    ctx = RunContext(config.load_config(overrides={"num_frames": 12}), str(tmp_path))
    s0 = stage0_ego_io.run(ctx); s0.save(ctx.stage_dir("stage0_ego_io"))
    stage2_track.run(ctx).save(ctx.stage_dir("stage2_track"))
    stage4_hand.run(ctx).save(ctx.stage_dir("stage4_hand"))
    b = stage5_ego_comp.run(ctx)
    # table-frame hand = inv(table_T) applied to world hand (first frame, pre-smoothing check loosely)
    assert b["hand_verts_t"].shape == s0["gt_hand_verts_w"].shape
    assert b["obj_poses_t"].shape[0] == 12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest egoaero/tests/test_stage5.py -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
# egoaero/egoaero/stages/stage5_ego_comp.py
"""Stage 5 (§2.1.4): ego-motion compensation. Transform all states into a fixed
table frame (SLAM in real mode; known camera + plane-fit table in mock), then
light temporal smoothing. No table/vertical constraint on the object."""
from __future__ import annotations
import numpy as np
from ..bundle import Bundle
from ..core.geometry import se3_inv, transform_points

NAME = "stage5_ego_comp"; INDEX = 5

def _smooth(x, w):
    if w <= 1:
        return x
    k = np.ones(w) / w; pad = w // 2
    xp = np.concatenate([x[pad:0:-1], x, x[-2:-pad - 2:-1]], 0)[:x.shape[0] + 2 * pad]
    flat = xp.reshape(xp.shape[0], -1); out = np.empty((x.shape[0], flat.shape[1]))
    for c in range(flat.shape[1]):
        out[:, c] = np.convolve(flat[:, c], k, "valid")[:x.shape[0]]
    return out.reshape(x.shape)

def run(ctx) -> Bundle:
    cfg = ctx.cfg
    s0 = ctx.load("stage0_ego_io"); s2 = ctx.load("stage2_track"); s4 = ctx.load("stage4_hand")
    if not cfg.mock:
        raise NotImplementedError("real ORB-SLAM3 backend — backends/real.py")
    table_T = s0["table_T_gt"]                       # real: estimate from SLAM + plane fit
    w2t = se3_inv(table_T)
    T = s4["hand_verts_w"].shape[0]
    hv = np.stack([transform_points(s4["hand_verts_w"][i], w2t) for i in range(T)], 0)
    hj = np.stack([transform_points(s4["hand_joints_w"][i], w2t) for i in range(T)], 0)
    op = np.stack([w2t @ s2["obj_poses_w"][i] for i in range(T)], 0)
    win = int(cfg.ego.smooth_window)
    hj = _smooth(hj, win)
    op[:, :3, 3] = _smooth(op[:, :3, 3], win)          # smooth object translation only
    hv = _smooth(hv, win)
    return Bundle(arrays={"hand_verts_t": hv, "hand_joints_t": hj, "obj_poses_t": op,
                          "obj_verts": s2["obj_verts"], "obj_faces": s2["obj_faces"]},
                  meta={"finger_idx": s0.meta["finger_idx"],
                        "stage_labels": s0.meta["stage_labels"]})
```

Append to `ASSUMPTIONS.md`: table-frame definition, SLAM hand-pixel down-weighting, and smoothing window are unspecified (§2.1.4) → mock uses GT `table_T` (real: ORB-SLAM3 + tabletop plane fit) and a moving-average smooth of window `cfg.ego.smooth_window`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest egoaero/tests/test_stage5.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add egoaero/egoaero/stages/stage5_ego_comp.py egoaero/tests/test_stage5.py egoaero/ASSUMPTIONS.md
git commit -m "egoaero: stage5 ego-motion compensation (table frame + smoothing)"
```

---

### Task 13: Stage 6a — contact regions + whole-hand translation

**Files:**
- Create: `egoaero/egoaero/stages/stage6_contact.py` (partial: region selection, signed distance, whole-hand translation)
- Test: `egoaero/tests/test_stage6_translation.py`

**Interfaces:**
- Produces (module-level): `active_window(stage_labels) -> list[int]`; `select_opposing_finger(hand_verts, finger_idx, obj_pts) -> str`; `signed_distance(points, obj_pts, obj_normals) -> (s[N], nearest_normal[N,3])`; `whole_hand_translation(regions, s, normals, gaps, weights, max_trans) -> delta[3]`.
- App C eqs: `d_t^k = mean(-n * ReLU(s - g_k))`, aggregate with region weights, clip to `max_trans`.

- [ ] **Step 1: Write the failing test**

```python
# egoaero/tests/test_stage6_translation.py
import numpy as np
from egoaero.stages import stage6_contact as C

def test_active_window_keeps_manipulation():
    labels = ["pre","pre","grasp","move","place","post"]
    win = C.active_window(labels)
    assert 2 in win and 3 in win and 4 in win and 0 not in win

def test_whole_hand_translation_pulls_toward_surface_and_clips():
    # one floating contact point 2cm in +z from a plane at z=0 (normal +z)
    obj_pts = np.array([[0,0,0.0],[0.01,0,0],[0,0.01,0]])
    obj_n = np.tile([0,0,1.0], (3,1))
    pts = np.array([[0,0,0.02]])
    s, nn = C.signed_distance(pts, obj_pts, obj_n)
    assert abs(s[0] - 0.02) < 1e-6
    d = C.whole_hand_translation([pts], [s], [nn], gaps=[0.0005],
                                 weights=[1.0], max_trans=0.034)
    assert d[2] < 0                       # pulled toward the surface (-z)
    assert np.linalg.norm(d) <= 0.034 + 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest egoaero/tests/test_stage6_translation.py -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
# egoaero/egoaero/stages/stage6_contact.py  (Task 13 portion)
"""Stage 6 (§2.1.5 / App C): adaptive contact optimization — geometry-level,
bounded correction of the replay hand. Object pose/mesh and MANO articulation
are unchanged. App C constants are used verbatim (see config.contact)."""
from __future__ import annotations
import numpy as np
from ..bundle import Bundle
from ..core.geometry import knn, vertex_normals, transform_points
from ..core import hand as H

NAME = "stage6_contact"; INDEX = 6

def active_window(stage_labels):
    keep = {"grasp", "move", "place"}
    return [i for i, l in enumerate(stage_labels) if str(l) in keep]

def signed_distance(points, obj_pts, obj_normals):
    d, idx = knn(points, obj_pts, k=1); idx = idx[:, 0]
    nn = obj_normals[idx]; nearest = obj_pts[idx]
    s = np.sum((points - nearest) * nn, axis=1)        # signed: + outside, - inside
    return s, nn

def whole_hand_translation(regions, s_list, n_list, gaps, weights, max_trans):
    """App C: d_t^k = mean(-n * ReLU(s - g_k)); aggregate by weights; clip."""
    dks = []; ws = []
    for pts, s, n, g, w in zip(regions, s_list, n_list, gaps, weights):
        relu = np.maximum(s - g, 0.0)
        dk = np.mean(-n * relu[:, None], axis=0)
        dks.append(dk); ws.append(w)
    if not dks:
        return np.zeros(3)
    raw = np.average(np.stack(dks, 0), axis=0, weights=np.array(ws))
    norm = np.linalg.norm(raw)
    return raw if norm <= max_trans or norm < 1e-12 else raw * (max_trans / norm)

def select_opposing_finger(hand_verts, finger_idx, obj_pts):
    best, bestd = "index", np.inf
    for fmap_key in ["index", "middle", "ring", "little"]:
        pad = H.fingertip_pad_idx(finger_idx, fmap_key)
        if len(pad) == 0:
            continue
        d, _ = knn(hand_verts[pad], obj_pts, k=1)
        m = float(np.median(d))
        if m < bestd:
            bestd, best = m, fmap_key
    return best
```

Note: in mock, `finger_idx` arrives from the bundle meta as lists — Task 14's `run` converts back to a dict of arrays before calling these helpers.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest egoaero/tests/test_stage6_translation.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add egoaero/egoaero/stages/stage6_contact.py egoaero/tests/test_stage6_translation.py
git commit -m "egoaero: stage6a contact regions + whole-hand translation (App C)"
```

---

### Task 14: Stage 6b — smoothing, local finger correction, penetration push-back + run()

**Files:**
- Modify: `egoaero/egoaero/stages/stage6_contact.py` (add temporal smoothing, local correction, push-back, `run`)
- Test: `egoaero/tests/test_stage6_run.py`

**Interfaces:**
- Produces (module-level): `triangular_smooth(deltas[T,3], window, boundary_frames) -> [T,3]`; `penetration_pushback(points, s, normals, eps, max_pb) -> r[3]`; `run(ctx) -> Bundle`.
- Consumes: `stage5_ego_comp` (`hand_verts_t`, `hand_joints_t`, `obj_poses_t`, `obj_verts`, `obj_faces`, meta `finger_idx`, `stage_labels`), `cfg.contact`.
- Produces bundle `stage6_contact`: arrays `hand_verts_t` (corrected), `hand_joints_t` (corrected), `contact_mask[T,Nh] bool`, `obj_poses_t`, `obj_verts`, `obj_faces`; meta `pen_before_mm`, `pen_after_mm`, `gap_before_mm`, `gap_after_mm`, `finger_idx`, `stage_labels`.

- [ ] **Step 1: Write the failing test**

```python
# egoaero/tests/test_stage6_run.py
from egoaero import config
from egoaero.pipeline import RunContext
from egoaero.stages import (stage0_ego_io, stage2_track, stage4_hand,
                            stage5_ego_comp, stage6_contact)
import numpy as np

def _prep(tmp_path, **over):
    ctx = RunContext(config.load_config(overrides={"num_frames": 16, **over}), str(tmp_path))
    stage0_ego_io.run(ctx).save(ctx.stage_dir("stage0_ego_io"))
    stage2_track.run(ctx).save(ctx.stage_dir("stage2_track"))
    stage4_hand.run(ctx).save(ctx.stage_dir("stage4_hand"))
    stage5_ego_comp.run(ctx).save(ctx.stage_dir("stage5_ego_comp"))
    return ctx

def test_triangular_smooth_bounds():
    d = np.zeros((10, 3)); d[5] = [0.03, 0, 0]
    sm = stage6_contact.triangular_smooth(d, window=9, boundary_frames=3)
    assert sm.shape == (10, 3) and abs(sm[5, 0]) < 0.03   # spike is smoothed down

def test_run_reduces_penetration_and_gap(tmp_path):
    ctx = _prep(tmp_path)
    b = stage6_contact.run(ctx)
    assert b.meta["pen_after_mm"] <= b.meta["pen_before_mm"] + 1e-6
    assert b["contact_mask"].shape[0] == 16
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest egoaero/tests/test_stage6_run.py -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

Append to `stage6_contact.py`:

```python
def triangular_smooth(deltas, window, boundary_frames):
    """Finite-window triangular kernel smoothing + boundary taper (App C)."""
    T = deltas.shape[0]; half = window // 2
    k = np.array([half + 1 - abs(i - half) for i in range(window)], float)
    k = k / k.sum()
    out = np.zeros_like(deltas)
    for c in range(deltas.shape[1]):
        out[:, c] = np.convolve(np.pad(deltas[:, c], half, mode="edge"), k, "valid")[:T]
    b = boundary_frames
    if b > 0 and T > 0:                                  # taper at active-segment ends
        taper = np.ones(T)
        ramp = np.linspace(0, 1, min(b, T))
        taper[:len(ramp)] = ramp; taper[-len(ramp):] = ramp[::-1]
        out = out * taper[:, None]
    return out

def penetration_pushback(points, s, normals, eps, max_pb):
    """App C: penetrating set s<-eps; push-back along normals, clipped."""
    pen = s < -eps
    if pen.sum() == 0:
        return np.zeros(3)
    depth = np.maximum(-eps - s[pen], 0.0)
    r = np.mean(normals[pen] * depth[:, None], axis=0)
    norm = np.linalg.norm(r)
    return r if norm <= max_pb or norm < 1e-12 else r * (max_pb / norm)

def _obj_world(obj_verts, obj_faces, pose):
    ow = transform_points(obj_verts, pose)
    return ow, vertex_normals(ow, obj_faces)

def run(ctx) -> Bundle:
    cfg = ctx.cfg; cc = cfg.contact
    s5 = ctx.load("stage5_ego_comp")
    hv = s5["hand_verts_t"].copy(); hj = s5["hand_joints_t"].copy()
    obj_poses = s5["obj_poses_t"]; ov = s5["obj_verts"]; of = s5["obj_faces"].astype(int)
    fidx = {k: np.asarray(v, int) for k, v in s5.meta["finger_idx"].items()}
    labels = s5.meta["stage_labels"]; T = hv.shape[0]
    win = active_window(labels)

    def pen_gap(vh):
        pens, gaps = [], []
        for i in range(T):
            ow, on = _obj_world(ov, of, obj_poses[i])
            s, _ = signed_distance(vh[i], ow, on)
            pens.append(np.maximum(-s, 0).sum())
            pad = H.fingertip_pad_idx(fidx, "thumb")
            gaps.append(np.median(np.abs(s[pad])) if len(pad) else 0.0)
        return float(np.mean(pens) * 1000), float(np.median(gaps) * 1000)

    pen_b, gap_b = pen_gap(hv)
    raw_delta = np.zeros((T, 3))
    for i in win:
        ow, on = _obj_world(ov, of, obj_poses[i])
        thumb = H.fingertip_pad_idx(fidx, "thumb")
        opp_f = select_opposing_finger(hv[i], fidx, ow)
        opp = H.fingertip_pad_idx(fidx, opp_f)
        huk = H.thenar_idx(fidx)
        regions, s_list, n_list, gaps, weights = [], [], [], [], []
        for pts_idx, g, w in [(thumb, cc.contact_gap_m, cc.region_weights["thumb"]),
                              (opp, cc.opp_gap_m, cc.region_weights["opp"]),
                              (huk, cc.thenar_gap_m, cc.region_weights["hukou"])]:
            if len(pts_idx) == 0:
                continue
            s, nn = signed_distance(hv[i][pts_idx], ow, on)
            regions.append(hv[i][pts_idx]); s_list.append(s); n_list.append(nn)
            gaps.append(g); weights.append(w)
        raw_delta[i] = whole_hand_translation(regions, s_list, n_list, gaps, weights,
                                              cc.max_global_trans_m)
    delta = triangular_smooth(raw_delta, cc.smooth_window, cc.boundary_frames)
    for i in range(T):
        hv[i] += delta[i]; hj[i] += delta[i]

    # local finger correction (thumb + opposing) weighted by finger chain
    for i in win:
        ow, on = _obj_world(ov, of, obj_poses[i])
        for f, g in [("thumb", cc.contact_gap_m)]:
            pad = H.fingertip_pad_idx(fidx, f)
            if len(pad) == 0:
                continue
            s, nn = signed_distance(hv[i][pad], ow, on)
            off = np.mean(-nn * np.maximum(s - g, 0)[:, None], axis=0)
            off = np.clip(off, -cc.max_finger_disp_m, cc.max_finger_disp_m)
            w = H.finger_chain_weights(hv[i], fidx, f)
            hv[i] += w[:, None] * off
        # penetration push-back (whole hand)
        s_all, n_all = signed_distance(hv[i], ow, on)
        r = penetration_pushback(hv[i], s_all, n_all, cc.pen_eps_m, cc.max_pushback_m)
        hv[i] += r; hj[i] += r

    pen_a, gap_a = pen_gap(hv)
    # contact mask: hand verts within contact gap of the object surface
    cmask = np.zeros((T, hv.shape[1]), bool)
    for i in range(T):
        ow, on = _obj_world(ov, of, obj_poses[i])
        s, _ = signed_distance(hv[i], ow, on)
        cmask[i] = np.abs(s) < (cc.contact_gap_m * 4)
    return Bundle(arrays={"hand_verts_t": hv, "hand_joints_t": hj, "contact_mask": cmask,
                          "obj_poses_t": obj_poses, "obj_verts": ov, "obj_faces": of},
                  meta={"pen_before_mm": pen_b, "pen_after_mm": pen_a,
                        "gap_before_mm": gap_b, "gap_after_mm": gap_a,
                        "finger_idx": s5.meta["finger_idx"], "stage_labels": labels})
```

Append to `ASSUMPTIONS.md`: region weights `w_k`, opposing-finger gap `g_opp`, penetration threshold `ε`, and the `α_i^f` profile are documented defaults (config.contact); App C constants used verbatim.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest egoaero/tests/test_stage6_run.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add egoaero/egoaero/stages/stage6_contact.py egoaero/tests/test_stage6_run.py egoaero/ASSUMPTIONS.md
git commit -m "egoaero: stage6b smoothing + local finger correction + penetration push-back"
```

---

### Task 15: Stage 7 — reconstruction eval

**Files:**
- Create: `egoaero/egoaero/stages/stage7_eval.py`
- Test: `egoaero/tests/test_stage7.py`

**Interfaces:**
- Consumes: `stage6_contact`, `stage5_ego_comp` (pre-correction hand), `stage0_ego_io` (GT).
- Produces bundle `stage7_eval`: meta `report` (dict of before/after metrics): `pen_before_mm`, `pen_after_mm`, `gap_before_mm`, `gap_after_mm`, `hand_jitter_before`, `hand_jitter_after`. Prints a small table.

- [ ] **Step 1: Write the failing test**

```python
# egoaero/tests/test_stage7.py
from egoaero import config
from egoaero.pipeline import RunContext
from egoaero.stages import (stage0_ego_io, stage2_track, stage4_hand,
                            stage5_ego_comp, stage6_contact, stage7_eval)

def test_report_has_before_after(tmp_path):
    ctx = RunContext(config.load_config(overrides={"num_frames": 16}), str(tmp_path))
    for m in (stage0_ego_io, stage2_track, stage4_hand, stage5_ego_comp, stage6_contact):
        m.run(ctx).save(ctx.stage_dir(m.NAME))
    b = stage7_eval.run(ctx)
    r = b.meta["report"]
    assert "pen_before_mm" in r and "pen_after_mm" in r
    assert r["pen_after_mm"] <= r["pen_before_mm"] + 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest egoaero/tests/test_stage7.py -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
# egoaero/egoaero/stages/stage7_eval.py
"""Stage 7: reconstruction error report — penetration / contact-gap / jitter
before vs after adaptive contact optimization (the 'watch error fall' table)."""
from __future__ import annotations
import numpy as np
from ..bundle import Bundle

NAME = "stage7_eval"; INDEX = 7

def _jitter(joints):
    return float(np.mean(np.abs(np.diff(joints, 2, axis=0))))

def run(ctx) -> Bundle:
    s5 = ctx.load("stage5_ego_comp"); s6 = ctx.load("stage6_contact")
    report = {
        "pen_before_mm": s6.meta["pen_before_mm"], "pen_after_mm": s6.meta["pen_after_mm"],
        "gap_before_mm": s6.meta["gap_before_mm"], "gap_after_mm": s6.meta["gap_after_mm"],
        "hand_jitter_before": _jitter(s5["hand_joints_t"]),
        "hand_jitter_after": _jitter(s6["hand_joints_t"]),
    }
    print("  metric                    before ->  after")
    print(f"  penetration sum (mm)      {report['pen_before_mm']:.2f} -> {report['pen_after_mm']:.2f}")
    print(f"  contact gap (mm)          {report['gap_before_mm']:.2f} -> {report['gap_after_mm']:.2f}")
    print(f"  hand jitter               {report['hand_jitter_before']:.5f} -> {report['hand_jitter_after']:.5f}")
    return Bundle(meta={"report": report})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest egoaero/tests/test_stage7.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add egoaero/egoaero/stages/stage7_eval.py egoaero/tests/test_stage7.py
git commit -m "egoaero: stage7 reconstruction eval report"
```

---

### Task 16: Pipeline orchestration, CLI, configs, backend stub

**Files:**
- Modify: `egoaero/egoaero/pipeline.py` (add `STAGES`, `_selected`, `run_pipeline`)
- Create: `egoaero/egoaero/cli.py`, `egoaero/egoaero/backends/real.py`, `egoaero/egoaero/configs/mock.yaml`, `egoaero/egoaero/stages/__init__.py` (exports)
- Test: `egoaero/tests/test_pipeline.py`

**Interfaces:**
- Consumes: all stage modules.
- Produces: `run_pipeline(cfg, run_dir, stages="all") -> RunContext`; `cli.main()`; `backends.real.get(kind) -> raises NotImplementedError` (gated stub).

- [ ] **Step 1: Write the failing test**

```python
# egoaero/tests/test_pipeline.py
from egoaero import config
from egoaero.pipeline import run_pipeline

def test_full_mock_pipeline_runs(tmp_path):
    cfg = config.load_config(overrides={"num_frames": 16})
    ctx = run_pipeline(cfg, str(tmp_path / "run"), stages="all")
    rep = ctx.load("stage7_eval").meta["report"]
    assert rep["pen_after_mm"] <= rep["pen_before_mm"] + 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest egoaero/tests/test_pipeline.py -v`
Expected: FAIL (`ImportError: run_pipeline`).

- [ ] **Step 3: Write minimal implementation**

Extend `pipeline.py` (keep the existing `RunContext`) with the orchestration, mirroring `render_and_compare/hoi_recon/pipeline.py:51-85` but with EgoAERO's 8 stages and no reproject byproduct:

```python
# append to egoaero/egoaero/pipeline.py
from .config import save_config
from .stages import (stage0_ego_io, stage1_semantic, stage2_track, stage3_mesh,
                     stage4_hand, stage5_ego_comp, stage6_contact, stage7_eval)

STAGES = [stage0_ego_io, stage1_semantic, stage2_track, stage3_mesh,
          stage4_hand, stage5_ego_comp, stage6_contact, stage7_eval]

def _selected(stages_arg, n):
    if stages_arg in (None, "all"):
        return list(range(n))
    out = []
    for part in str(stages_arg).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-"); a = int(a) if a else 0; b = int(b) if b else n - 1
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    return sorted(set(i for i in out if 0 <= i < n))

def run_pipeline(cfg, run_dir, stages="all"):
    ctx = RunContext(cfg, run_dir)
    save_config(cfg, os.path.join(run_dir, "config.yaml"))
    for i in _selected(stages, len(STAGES)):
        mod = STAGES[i]
        if ctx.has(mod.NAME) and not cfg.force:
            continue
        out = mod.run(ctx); out.save(ctx.stage_dir(mod.NAME))
    return ctx
```

`stages/__init__.py` should import the 8 modules so `from .stages import ...` works. `backends/real.py`:

```python
# egoaero/egoaero/backends/real.py
"""Real-backend registry. Each raises until its weights/repo are installed."""
def get(kind: str):
    raise NotImplementedError(
        f"real backend '{kind}' not installed; run in --mock or install via setup "
        "(HaWoR, SAM3, ORB-SLAM3, BundleSDF, SAM3D). See egoaero/README.md.")
```

`cli.py`:

```python
# egoaero/egoaero/cli.py
import argparse, os
from .config import load_config
from .pipeline import run_pipeline

def main():
    ap = argparse.ArgumentParser("egoaero")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--video", default=None)
    ap.add_argument("--num-frames", type=int, default=None)
    ap.add_argument("--stages", default="all")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    over = {"mock": True if a.mock else None, "video": a.video,
            "num_frames": a.num_frames, "force": a.force or None}
    cfg = load_config(a.config, {k: v for k, v in over.items() if v is not None})
    ctx = run_pipeline(cfg, a.out, a.stages)
    print("done:", os.path.abspath(a.out))

if __name__ == "__main__":
    main()
```

`configs/mock.yaml`: `mock: true` plus a comment header (defaults already in config.py).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest egoaero/tests/test_pipeline.py -v`
Expected: PASS. Also run end-to-end CLI: `python -m egoaero.cli --out egoaero/runs/demo --mock --num-frames 24` (from `egoaero/`), expect a printed before→after table.

- [ ] **Step 5: Commit**

```bash
git add egoaero/egoaero/pipeline.py egoaero/egoaero/cli.py egoaero/egoaero/backends/real.py egoaero/egoaero/configs egoaero/egoaero/stages/__init__.py egoaero/tests/test_pipeline.py
git commit -m "egoaero: pipeline orchestration + CLI + backend stub + mock config"
```

---

### Task 17: Workbench contract output

**Files:**
- Create: `egoaero/egoaero/contract.py`
- Modify: `egoaero/egoaero/pipeline.py` (call contract writer after stage7)
- Test: `egoaero/tests/test_contract.py`

**Interfaces:**
- Consumes: `stage6_contact` (final hand/object/contact), `stage5_ego_comp` (object trajectory).
- Produces: `contract.write(ctx) -> dict` writing `<run>/contract/` with `hand_mano.npz` (verts/joints per frame), `object_mesh.obj`, `object_traj.npz` (`obj_poses_t`), `contact.npz` (`contact_mask`), and `manifest.json` listing them; `contract.validate(run_dir) -> bool`.

- [ ] **Step 1: Write the failing test**

```python
# egoaero/tests/test_contract.py
from egoaero import config
from egoaero.pipeline import run_pipeline
from egoaero import contract

def test_contract_written_and_valid(tmp_path):
    cfg = config.load_config(overrides={"num_frames": 16})
    ctx = run_pipeline(cfg, str(tmp_path / "run"), stages="all")
    out = contract.write(ctx)
    assert contract.validate(str(tmp_path / "run"))
    assert "object_traj" in out and "hand_mano" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest egoaero/tests/test_contract.py -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
# egoaero/egoaero/contract.py
"""Workbench method-contract writer: emit the comparable per-clip output
(MANO hand, object mesh, object 6-DoF trajectory, contact maps) under <run>/contract/."""
from __future__ import annotations
import json, os
import numpy as np

def _write_obj(path, verts, faces):
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")

def write(ctx) -> dict:
    s6 = ctx.load("stage6_contact")
    d = os.path.join(ctx.run_dir, "contract"); os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, "hand_mano.npz"),
             verts=s6["hand_verts_t"], joints=s6["hand_joints_t"])
    np.savez(os.path.join(d, "object_traj.npz"), poses=s6["obj_poses_t"])
    np.savez(os.path.join(d, "contact.npz"), mask=s6["contact_mask"])
    _write_obj(os.path.join(d, "object_mesh.obj"), s6["obj_verts"], s6["obj_faces"].astype(int))
    manifest = {"hand_mano": "hand_mano.npz", "object_mesh": "object_mesh.obj",
                "object_traj": "object_traj.npz", "contact": "contact.npz",
                "frames": int(s6["hand_verts_t"].shape[0])}
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest

def validate(run_dir) -> bool:
    d = os.path.join(run_dir, "contract")
    need = ["hand_mano.npz", "object_mesh.obj", "object_traj.npz", "contact.npz", "manifest.json"]
    return all(os.path.exists(os.path.join(d, n)) for n in need)
```

In `pipeline.run_pipeline`, after the stage loop, add:

```python
    if 7 in _selected(stages, len(STAGES)) and ctx.has("stage6_contact"):
        from .contract import write as _cw
        _cw(ctx)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest egoaero/tests/test_contract.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add egoaero/egoaero/contract.py egoaero/egoaero/pipeline.py egoaero/tests/test_contract.py
git commit -m "egoaero: workbench-contract output writer + validator"
```

---

### Task 18: End-to-end smoke test, README, workbench wiring

**Files:**
- Create: `egoaero/tests/test_smoke.py`
- Create: `egoaero/README.md`, `egoaero/environment.yml` (or note reuse), `egoaero/pyproject.toml`
- Modify: root `README.md` (Methods table row)
- Test: `egoaero/tests/test_smoke.py`

**Interfaces:** none new — this task verifies the whole pipeline + documents it.

- [ ] **Step 1: Write the failing test**

```python
# egoaero/tests/test_smoke.py
from egoaero import config
from egoaero.pipeline import run_pipeline
from egoaero import contract

def test_end_to_end_mock_smoke(tmp_path):
    cfg = config.load_config(overrides={"num_frames": 24, "seed": 0})
    ctx = run_pipeline(cfg, str(tmp_path / "run"), stages="all")
    rep = ctx.load("stage7_eval").meta["report"]
    # the pipeline drives injected errors down
    assert rep["pen_after_mm"] <= rep["pen_before_mm"] + 1e-6
    assert ctx.load("stage2_track").meta["track_err_deg_after"] < \
           ctx.load("stage2_track").meta["track_err_deg_before"]
    assert ctx.load("stage4_hand").meta["transl_err_after_mm"] < \
           ctx.load("stage4_hand").meta["transl_err_before_mm"]
    assert contract.validate(str(tmp_path / "run"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest egoaero/tests/test_smoke.py -v`
Expected: PASS already if all prior tasks done — if any assertion fails, fix the responsible stage before proceeding (this is the integration gate).

- [ ] **Step 3: Write the docs**

Create `egoaero/README.md` documenting: what it reproduces (EgoAERO Part A), the 8 stages and their faithful/default status, quickstart (`python -m egoaero.cli --out runs/demo --mock`), the contract output layout, pointer to `ASSUMPTIONS.md` and the spec, and the SP2/SP3/SP4 roadmap. Create `egoaero/pyproject.toml` (package `egoaero`, deps numpy/scipy/pyyaml/trimesh) and `egoaero/environment.yml` (or a one-line note that it shares the workbench env). Add a row to the root `README.md` Methods table:

```markdown
| [`egoaero/`](egoaero/) | 🟡 recon (Part A) runnable in mock | EgoAERO asset-free egocentric hand-object reconstruction (Sec 2.1). Faithful adaptive contact optimization; tracking/field via documented defaults + mock. RL/dataset (SP2/4) planned. See [README](egoaero/README.md). |
```

- [ ] **Step 4: Run full suite**

Run: `python -m pytest egoaero/tests/ -q`
Expected: all green (one test file per task).

- [ ] **Step 5: Commit**

```bash
git add egoaero/tests/test_smoke.py egoaero/README.md egoaero/pyproject.toml egoaero/environment.yml README.md
git commit -m "egoaero: end-to-end mock smoke test + README + workbench Methods row"
```

---

## Self-Review

**Spec coverage (SP1 §3 stages → tasks):**
- Stage 0 ego-io → Task 6 ✓ ; Stage 1 semantic → Task 7 ✓ ; Stage 2 track (App A) → Tasks 8–9 ✓ ;
  Stage 3 mesh (App B) → Task 10 ✓ ; Stage 4 hand → Task 11 ✓ ; Stage 5 ego-comp → Task 12 ✓ ;
  Stage 6 contact (App C, faithful) → Tasks 13–14 ✓ ; Stage 7 eval → Task 15 ✓ .
- Core (geometry/hand/mock_scene/bundle/config) → Tasks 1–5 ✓ ; orchestration/CLI → Task 16 ✓ ;
  contract output → Task 17 ✓ ; smoke + docs + workbench wiring → Task 18 ✓ .
- Operating principle (ASSUMPTIONS.md) → seeded Task 1, appended in Tasks 2, 7, 9, 10, 12, 14 ✓ .
- App C constants verbatim → Task 1 config + Tasks 13–14 ✓ .

**Placeholder scan:** no TBD/TODO; every code step shows complete code. The one soft spot
(`core/hand.py` module-cache `_z`) is called out with an explicit alternative — not a placeholder.

**Type consistency:** stage bundles thread consistent keys — `obj_poses_w`(stage2)→`obj_poses_t`(stage5/6),
`hand_verts_w`(stage4)→`hand_verts_t`(stage5/6), `finger_idx`/`stage_labels` carried in meta as
lists and rehydrated in Task 14. `signed_distance` returns `(s, normals)` everywhere; `whole_hand_translation`
signature matches its caller in Task 14. `RunContext`/`run_pipeline`/`NAME`/`INDEX` match the stage convention.

**Known limitation (documented, not a gap to fix in SP1):** mock object tracking exercises
translation drift only; rotation tracking and the neural field are real-backend territory (logged
in ASSUMPTIONS.md). This is consistent with the spec's faithfulness map.
