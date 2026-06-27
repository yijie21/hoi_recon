# CHOIR Stage 3 Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the pure-function foundation library for CHOIR's contact-aware joint
optimization (Stage 3): the energy-term preset registry (with a locked faithful preset),
phase segmentation, barycentric contact correspondence, the anatomical constraint, and
the fine-stage proxy metrics — all unit-tested, no GPU/subprocess/dataset needed.

**Architecture:** A new `hoi_recon/choir_fine/` package of small, single-responsibility,
env-agnostic modules (numpy + a little torch), each independently unit-tested with pytest.
These are the building blocks the Stage 3 subprocess optimizer (a follow-on plan) consumes:
weights come from `presets`, contact terms from `contact` + `phases`, the hand prior from
`anatomical`, and `metrics` feeds the evaluation harness. Design source of truth:
`docs/superpowers/specs/2026-06-12-choir-fine-stage-design.md`.

**Tech Stack:** Python 3.10, numpy 2.2, scipy (cKDTree), trimesh 4.12, torch 2.11, pytest.

---

## File Structure

- Create: `hoi_recon/choir_fine/__init__.py` — package marker.
- Create: `hoi_recon/choir_fine/presets.py` — named energy-term weight presets + accessor.
- Create: `hoi_recon/choir_fine/phases.py` — clip phase segmentation.
- Create: `hoi_recon/choir_fine/contact.py` — barycentric contact correspondence builder.
- Create: `hoi_recon/choir_fine/anatomical.py` — MANO twist-splay-bend constraint (torch).
- Create: `hoi_recon/choir_fine/metrics.py` — contact-gap + penetration proxy metrics.
- Create: `tests/test_choir_fine_presets.py`, `tests/test_choir_fine_phases.py`,
  `tests/test_choir_fine_contact.py`, `tests/test_choir_fine_anatomical.py`,
  `tests/test_choir_fine_metrics.py`.

All test commands assume the `hoi_recon` conda env:
`conda run -n hoi_recon python -m pytest ...` (or activate it first).

---

## Task 0: Prerequisite — pytest + package skeleton

**Files:**
- Create: `hoi_recon/choir_fine/__init__.py`

- [ ] **Step 1: Install pytest into the env**

Run: `conda run -n hoi_recon pip install pytest`
Expected: pytest installs (numpy/trimesh/torch already present).

- [ ] **Step 2: Create the package marker**

```python
# hoi_recon/choir_fine/__init__.py
"""CHOIR fine-stage (Stage 3) foundation components: energy-term presets, phase
segmentation, barycentric contact correspondence, anatomical constraint, proxy metrics.
See docs/superpowers/specs/2026-06-12-choir-fine-stage-design.md."""
```

- [ ] **Step 3: Verify it imports**

Run: `conda run -n hoi_recon python -c "import hoi_recon.choir_fine; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 4: Commit**

```bash
git add hoi_recon/choir_fine/__init__.py
git commit -m "choir_fine: package skeleton"
```

---

## Task 1: Energy-term preset registry + faithful-preset lock test

**Files:**
- Create: `hoi_recon/choir_fine/presets.py`
- Test: `tests/test_choir_fine_presets.py`

- [ ] **Step 1: Write the failing test (lock + accessor)**

```python
# tests/test_choir_fine_presets.py
import pytest
from hoi_recon.choir_fine import presets


def test_choir_faithful_weights_locked():
    """CHOIR arXiv:2605.20992 §7.3 weights — verbatim. This test is the anti-drift
    guard; do NOT change these numbers without changing the paper reference."""
    w = presets.CHOIR_FAITHFUL
    assert w["contact"] == 1000.0
    assert w["pen"] == 500.0
    assert w["sil"] == 500.0
    assert w["anc_2d"] == 0.5
    assert w["anc_anat"] == 30.0
    assert w["anc_pose_h"] == 100.0
    assert w["anc_pose_o"] == 100.0
    assert w["temp_pose_vel"] == 500.0
    assert w["temp_obj_vel"] == 500.0
    assert w["temp_wrist_anchor"] == 200.0
    assert w["temp_hand_tr_vel"] == 200.0
    assert w["temp_root_R_vel"] == 500.0
    assert w["temp_pose_acc"] == 1000.0
    assert w["temp_hand_tr_acc"] == 5000.0
    assert w["temp_root_R_acc"] == 3000.0
    assert w["hand_sil"] == 0.0          # CHOIR has no hand-silhouette term
    assert presets.LRS_FAITHFUL == {"object": 3e-4, "finger": 5e-4, "wrist": 5e-5}
    assert presets.ITERS_FAITHFUL == 800


def test_every_term_has_a_weight():
    """Every registered term name must have a weight in every preset (no silent gaps)."""
    for name in ("choir_faithful", "combined_v2"):
        w = presets.get_preset(name)
        assert set(w) >= set(presets.TERMS), set(presets.TERMS) - set(w)


def test_combined_v2_enables_our_toggles():
    """combined_v2 = faithful + our improvements on (hand silhouette here)."""
    w = presets.get_preset("combined_v2")
    assert w["hand_sil"] > 0.0           # T3 hand image registration on
    # faithful weights preserved
    assert w["contact"] == 1000.0


def test_get_preset_unknown_raises():
    with pytest.raises(KeyError):
        presets.get_preset("nope")
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n hoi_recon python -m pytest tests/test_choir_fine_presets.py -v`
Expected: FAIL — `ModuleNotFoundError: hoi_recon.choir_fine.presets`.

- [ ] **Step 3: Write minimal implementation**

```python
# hoi_recon/choir_fine/presets.py
"""Named energy-term weight presets for the CHOIR Stage-3 optimizer.

choir_faithful = CHOIR arXiv:2605.20992 §7.3 weights, VERBATIM and LOCKED (guarded by
tests/test_choir_fine_presets.py). combined_v2 = faithful + our improvement toggles.
Each term name is a key the optimizer's term registry looks up."""
from __future__ import annotations

import copy

# every energy term the Stage-3 optimizer can apply (weight 0 => inactive)
TERMS = [
    "contact", "template", "bridge", "gap", "patch",      # contact family
    "pen", "sil",                                          # penetration, object silhouette
    "anc_2d", "anc_anat", "anc_pose_h", "anc_pose_o",      # anchors
    "temp_pose_vel", "temp_obj_vel", "temp_wrist_anchor",  # temporal (velocity)
    "temp_hand_tr_vel", "temp_root_R_vel",
    "temp_pose_acc", "temp_hand_tr_acc", "temp_root_R_acc",  # temporal (acceleration)
    "hand_sil",                                            # our improvement (T3)
]

# CHOIR §7.3 verbatim. The contact-family stabilizers (template/bridge/gap/patch) are
# "annealed" in CHOIR; start them at 0 and let the optimizer schedule them.
CHOIR_FAITHFUL = {
    "contact": 1000.0, "template": 0.0, "bridge": 0.0, "gap": 0.0, "patch": 0.0,
    "pen": 500.0, "sil": 500.0,
    "anc_2d": 0.5, "anc_anat": 30.0, "anc_pose_h": 100.0, "anc_pose_o": 100.0,
    "temp_pose_vel": 500.0, "temp_obj_vel": 500.0, "temp_wrist_anchor": 200.0,
    "temp_hand_tr_vel": 200.0, "temp_root_R_vel": 500.0,
    "temp_pose_acc": 1000.0, "temp_hand_tr_acc": 5000.0, "temp_root_R_acc": 3000.0,
    "hand_sil": 0.0,
}
LRS_FAITHFUL = {"object": 3e-4, "finger": 5e-4, "wrist": 5e-5}
ITERS_FAITHFUL = 800

# contact-cache rebuild + anchor-build constants (CHOIR §7.3 / §7.2)
CONTACT_CACHE = {"dist_m": 0.05, "normal_deg": 60.0, "topk": 8, "softmax_sigma": 0.01}
ANCHOR_BUILD = {"n_surface": 10000, "knn": 50, "dist_m": 0.02, "normal_deg": 60.0}

# combined_v2 = faithful + our improvement toggles (each independently overridable)
COMBINED_V2 = {**copy.deepcopy(CHOIR_FAITHFUL), "hand_sil": 1.0}

_PRESETS = {"choir_faithful": CHOIR_FAITHFUL, "combined_v2": COMBINED_V2}


def get_preset(name: str) -> dict:
    """Return a deep copy of the named preset's weight dict. Raises KeyError if unknown."""
    if name not in _PRESETS:
        raise KeyError(f"unknown preset '{name}'; have {sorted(_PRESETS)}")
    return copy.deepcopy(_PRESETS[name])
```

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n hoi_recon python -m pytest tests/test_choir_fine_presets.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add hoi_recon/choir_fine/presets.py tests/test_choir_fine_presets.py
git commit -m "choir_fine: energy-term presets + faithful lock test"
```

---

## Task 2: Phase segmentation

**Files:**
- Create: `hoi_recon/choir_fine/phases.py`
- Test: `tests/test_choir_fine_phases.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_choir_fine_phases.py
import numpy as np
from hoi_recon.choir_fine import phases


def test_all_contact_is_manipulation():
    labels = phases.segment_phases(np.ones(10, bool))
    assert (labels == phases.PHASES.index("manipulation")).all()


def test_no_contact_without_motion_is_approach():
    labels = phases.segment_phases(np.zeros(8, bool))
    assert (labels == phases.PHASES.index("approach")).all()


def test_mid_contact_splits_approach_manip_release():
    cp = np.zeros(10, bool); cp[3:7] = True            # contact on frames 3..6
    labels = phases.segment_phases(cp)
    assert (labels[:3] == phases.PHASES.index("approach")).all()
    assert (labels[3:7] == phases.PHASES.index("manipulation")).all()
    assert (labels[7:] == phases.PHASES.index("release")).all()


def test_static_ends_with_motion_signal():
    cp = np.zeros(10, bool); cp[4:6] = True
    motion = np.full(10, 1.0); motion[:2] = 0.0; motion[8:] = 0.0   # static head + tail
    labels = phases.segment_phases(cp, motion=motion, static_thresh=0.1)
    assert (labels[:2] == phases.PHASES.index("pre_static")).all()
    assert (labels[8:] == phases.PHASES.index("post_static")).all()
    assert labels[2] == phases.PHASES.index("approach")            # moving, pre-contact
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n hoi_recon python -m pytest tests/test_choir_fine_phases.py -v`
Expected: FAIL — `ModuleNotFoundError: hoi_recon.choir_fine.phases`.

- [ ] **Step 3: Write minimal implementation**

```python
# hoi_recon/choir_fine/phases.py
"""Segment a clip into CHOIR's five interaction phases from per-frame contact presence
(and optional per-frame motion magnitude). Contact terms are applied only on
manipulation frames downstream; the static phases let the optimizer skip moving-only
terms where the hand is at rest. CHOIR §7.3."""
from __future__ import annotations

import numpy as np

PHASES = ["pre_static", "approach", "manipulation", "release", "post_static"]


def segment_phases(contact_present, motion=None, static_thresh=1e-3):
    """contact_present: (T,) bool. motion: optional (T,) float per-frame motion magnitude.
    Returns labels (T,) int indexing PHASES."""
    contact_present = np.asarray(contact_present, bool)
    T = len(contact_present)
    labels = np.full(T, PHASES.index("approach"), int)        # default: approach
    idx = np.where(contact_present)[0]

    if len(idx) == 0:
        if motion is not None:
            labels[np.asarray(motion) < static_thresh] = PHASES.index("pre_static")
        return labels

    f0, f1 = int(idx[0]), int(idx[-1])
    labels[f0:f1 + 1] = PHASES.index("manipulation")
    labels[:f0] = PHASES.index("approach")
    labels[f1 + 1:] = PHASES.index("release")

    if motion is not None:
        motion = np.asarray(motion, float)
        for t in range(f0):                                   # leading static run
            if motion[t] < static_thresh:
                labels[t] = PHASES.index("pre_static")
            else:
                break
        for t in range(T - 1, f1, -1):                        # trailing static run
            if motion[t] < static_thresh:
                labels[t] = PHASES.index("post_static")
            else:
                break
    return labels
```

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n hoi_recon python -m pytest tests/test_choir_fine_phases.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add hoi_recon/choir_fine/phases.py tests/test_choir_fine_phases.py
git commit -m "choir_fine: phase segmentation"
```

---

## Task 3: Barycentric contact correspondence

**Files:**
- Create: `hoi_recon/choir_fine/contact.py`
- Test: `tests/test_choir_fine_contact.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_choir_fine_contact.py
import numpy as np
import trimesh
from hoi_recon.choir_fine import contact


def _quad_mesh():
    """A flat 2x2 quad in the z=0 plane (two triangles), normals +z."""
    v = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], float)
    f = np.array([[0, 1, 2], [0, 2, 3]])
    return trimesh.Trimesh(v, f, process=False)


def test_near_point_gets_valid_correspondence():
    m = _quad_mesh()
    hand = np.array([[0.0, 0.0, 0.01]])              # 1cm above the surface
    out = contact.build_correspondences(hand, m, dist_thresh=0.02, topk=8, seed=0)
    assert out["valid"][0]
    assert out["weight"][0].sum() == \
        __import__("pytest").approx(1.0, abs=1e-5)   # softmax weights normalized
    # anchors lie on the surface (z ~ 0)
    anchors = out["anchor"][0][out["weight"][0] > 0]
    assert np.abs(anchors[:, 2]).max() < 1e-6


def test_far_point_is_invalid():
    m = _quad_mesh()
    hand = np.array([[0.0, 0.0, 0.10]])              # 10cm away > 2cm gate
    out = contact.build_correspondences(hand, m, dist_thresh=0.02, topk=8, seed=0)
    assert not out["valid"][0]


def test_wrong_side_normal_gate_rejects():
    """A point approaching from BEHIND the surface (−z) fails the normal cone."""
    m = _quad_mesh()
    hand = np.array([[0.0, 0.0, -0.01]])             # below the +z surface
    out = contact.build_correspondences(hand, m, dist_thresh=0.02,
                                        normal_deg=60.0, topk=8, seed=0)
    assert not out["valid"][0]


def test_bary_reconstructs_anchor():
    """face_id + barycentric must reconstruct the stored anchor point."""
    m = _quad_mesh()
    hand = np.array([[0.3, -0.2, 0.005]])
    out = contact.build_correspondences(hand, m, dist_thresh=0.05, topk=4, seed=0)
    assert out["valid"][0]
    k = int(np.argmax(out["weight"][0]))
    fid = out["face_id"][0, k]
    bary = out["bary"][0, k]
    recon = (m.triangles[fid] * bary[:, None]).sum(0)
    assert np.allclose(recon, out["anchor"][0, k], atol=1e-5)
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n hoi_recon python -m pytest tests/test_choir_fine_contact.py -v`
Expected: FAIL — `ModuleNotFoundError: hoi_recon.choir_fine.contact`.

- [ ] **Step 3: Write minimal implementation**

```python
# hoi_recon/choir_fine/contact.py
"""Barycentric contact correspondence (CHOIR §7.2). For each hand contact point, find up
to top-K object-surface anchors that pass a distance gate (<dist_thresh) and a
surface-normal compatibility gate (hand->anchor direction within normal_deg of the object
normal). Anchors are sampled surface points stored as (face_id, barycentric, softmax
weight) so they move rigidly with the object pose during optimization."""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def build_correspondences(hand_pts, mesh, *, n_surface=10000, knn=50, topk=8,
                          dist_thresh=0.02, normal_deg=60.0, softmax_sigma=0.01, seed=0):
    """hand_pts: (Nh,3). mesh: trimesh.Trimesh (object, current frame, world coords).
    Returns dict of arrays indexed [hand_vertex, k]:
      face_id (Nh,topk) int (-1 where invalid), bary (Nh,topk,3), anchor (Nh,topk,3),
      weight (Nh,topk) softmax over kept anchors (0 where invalid), valid (Nh,) bool."""
    hand_pts = np.asarray(hand_pts, float)
    Nh = len(hand_pts)
    pts, face_idx = trimesh.sample.sample_surface(mesh, n_surface, seed=seed)
    fnormals = np.asarray(mesh.face_normals)[face_idx]            # (n_surface,3)
    tree = cKDTree(pts)
    cos_thresh = np.cos(np.deg2rad(normal_deg))

    face_id = np.full((Nh, topk), -1, np.int64)
    bary = np.zeros((Nh, topk, 3), float)
    anchor = np.zeros((Nh, topk, 3), float)
    weight = np.zeros((Nh, topk), float)
    valid = np.zeros(Nh, bool)

    k_query = min(knn, n_surface)
    dists, nn = tree.query(hand_pts, k=k_query)
    if k_query == 1:
        dists, nn = dists[:, None], nn[:, None]

    for i in range(Nh):
        cand = nn[i]
        d = dists[i]
        a = pts[cand]                                            # candidate anchors
        # direction hand->anchor must oppose the outward normal (hand outside, anchor
        # on the contact-facing side): (anchor-hand) . normal < 0  AND aligned within cone
        dirv = a - hand_pts[i]
        dn = np.linalg.norm(dirv, axis=1) + 1e-12
        cosang = -(dirv / dn[:, None] * fnormals[cand]).sum(1)   # +1 when facing the hand
        ok = (d < dist_thresh) & (cosang > cos_thresh)
        if not ok.any():
            continue
        sel = np.where(ok)[0][:topk]                             # nearest-first, top-K
        gi = cand[sel]
        face_id[i, :len(sel)] = face_idx[gi]
        anchor[i, :len(sel)] = pts[gi]
        bary[i, :len(sel)] = trimesh.triangles.points_to_barycentric(
            mesh.triangles[face_idx[gi]], pts[gi])
        w = np.exp(-(d[sel] ** 2) / softmax_sigma)
        weight[i, :len(sel)] = w / w.sum()
        valid[i] = True

    return {"face_id": face_id, "bary": bary, "anchor": anchor,
            "weight": weight, "valid": valid}
```

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n hoi_recon python -m pytest tests/test_choir_fine_contact.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add hoi_recon/choir_fine/contact.py tests/test_choir_fine_contact.py
git commit -m "choir_fine: barycentric contact correspondence"
```

---

## Task 4: Anatomical (twist-splay-bend) constraint

**Files:**
- Create: `hoi_recon/choir_fine/anatomical.py`
- Test: `tests/test_choir_fine_anatomical.py`

> Note: fingers flex (bend) primarily about one joint axis. This penalizes the off-flexion
> components (twist about the bone axis + splay/abduction) of each of MANO's 15 joints. The
> axis-to-component mapping uses MANO's axis-angle convention (component 2 = bend); the
> exact per-joint axis is a documented approximation of CHOIR's "twist-splay-bend" term.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_choir_fine_anatomical.py
import torch
from hoi_recon.choir_fine import anatomical


def test_pure_bend_is_low():
    pose = torch.zeros(1, 15, 3)
    pose[..., 2] = 0.8                          # bend axis only
    assert float(anatomical.anatomical_loss(pose)) < 1e-6


def test_twist_is_penalized():
    pose = torch.zeros(1, 15, 3)
    pose[..., 0] = 0.8                          # twist about bone axis
    assert float(anatomical.anatomical_loss(pose)) > 0.1


def test_splay_is_penalized():
    pose = torch.zeros(1, 15, 3)
    pose[..., 1] = 0.8                          # splay / abduction
    assert float(anatomical.anatomical_loss(pose)) > 0.1


def test_accepts_flat_45_shape():
    pose = torch.zeros(45)                      # (15*3,) flat is accepted
    pose[2::3] = 0.5                            # bend components -> low
    assert float(anatomical.anatomical_loss(pose)) < 1e-6


def test_loss_is_scalar_and_differentiable():
    pose = torch.zeros(1, 15, 3, requires_grad=True)
    loss = anatomical.anatomical_loss(pose + torch.tensor([0.5, 0.0, 0.0]))  # twist offset
    assert loss.dim() == 0                     # scalar
    loss.backward()
    assert pose.grad is not None and torch.isfinite(pose.grad).all()
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n hoi_recon python -m pytest tests/test_choir_fine_anatomical.py -v`
Expected: FAIL — `ModuleNotFoundError: hoi_recon.choir_fine.anatomical`.

- [ ] **Step 3: Write minimal implementation**

```python
# hoi_recon/choir_fine/anatomical.py
"""MANO anatomical constraint (CHOIR L^h_anat). Fingers flex about a single joint axis;
penalize the off-flexion rotation components — twist (about the bone axis) and splay
(abduction) — leaving bend (the flexion axis) free. Operates on MANO hand_pose in
axis-angle. The flexion axis is taken as component index 2 (MANO convention); penalizing
the other two components is a documented approximation of CHOIR's twist-splay-bend term."""
from __future__ import annotations

import torch

BEND_AXIS = 2          # axis-angle component that finger flexion lives on (MANO)


def anatomical_loss(hand_pose):
    """hand_pose: (...,15,3) or (...,45) axis-angle. Returns a scalar tensor."""
    aa = hand_pose.reshape(*hand_pose.shape[:-1], 15, 3) if hand_pose.shape[-1] == 45 \
        else hand_pose.reshape(-1, 15, 3) if hand_pose.dim() == 1 else hand_pose
    off = [i for i in range(3) if i != BEND_AXIS]      # twist + splay axes
    return (aa[..., off] ** 2).mean()
```

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n hoi_recon python -m pytest tests/test_choir_fine_anatomical.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add hoi_recon/choir_fine/anatomical.py tests/test_choir_fine_anatomical.py
git commit -m "choir_fine: anatomical twist-splay-bend constraint"
```

---

## Task 5: Fine-stage proxy metrics (contact gap + penetration)

**Files:**
- Create: `hoi_recon/choir_fine/metrics.py`
- Test: `tests/test_choir_fine_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_choir_fine_metrics.py
import numpy as np
from hoi_recon.choir_fine import metrics


def test_contact_gap_zero_when_touching():
    # hand point coincident with a surface point on the single contact frame
    hand_c = np.zeros((1, 1, 3))                       # (T=1, Nh=1, 3)
    surf = np.zeros((1, 1, 3))                         # (T=1, No=1, 3)
    cp = np.array([True])
    assert metrics.contact_gap(hand_c, surf, cp) == 0.0


def test_contact_gap_is_distance():
    hand_c = np.zeros((1, 1, 3))
    surf = np.full((1, 1, 3), 0.0); surf[0, 0, 2] = 0.05    # 5cm away
    assert metrics.contact_gap(hand_c, surf, np.array([True])) == \
        __import__("pytest").approx(0.05)


def test_contact_gap_nan_when_no_contact_frames():
    hand_c = np.zeros((2, 1, 3)); surf = np.zeros((2, 1, 3))
    assert np.isnan(metrics.contact_gap(hand_c, surf, np.array([False, False])))


def test_penetration_positive_inside():
    # one hand vertex 1cm inside a surface whose outward normal is +z, at z=0
    hand = np.array([[[0.0, 0.0, -0.01]]])             # (T=1, Nh=1, 3), below surface
    surf = np.array([[[0.0, 0.0, 0.0]]])               # nearest surface point
    nrm = np.array([[[0.0, 0.0, 1.0]]])                # outward normal +z
    pen = metrics.penetration_depth(hand, surf, nrm)
    assert pen == __import__("pytest").approx(0.01, abs=1e-6)


def test_penetration_zero_outside():
    hand = np.array([[[0.0, 0.0, 0.02]]])              # above surface (outside)
    surf = np.array([[[0.0, 0.0, 0.0]]])
    nrm = np.array([[[0.0, 0.0, 1.0]]])
    assert metrics.penetration_depth(hand, surf, nrm) == 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n hoi_recon python -m pytest tests/test_choir_fine_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: hoi_recon.choir_fine.metrics`.

- [ ] **Step 3: Write minimal implementation**

```python
# hoi_recon/choir_fine/metrics.py
"""Fine-stage proxy metrics for the evaluation harness (no GT needed):
  contact_gap        — median nearest hand->object surface distance on contact frames
  penetration_depth  — summed one-sided penetration (hand inside object) over the clip
These feed scripts/object_confidence.py for per-preset A/B + ablation."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def contact_gap(hand_contact, obj_surface, contact_present):
    """hand_contact: (T,Nh,3) hand contact verts. obj_surface: (T,No,3) object surface
    points (world). contact_present: (T,) bool. Returns the median over contact frames
    of the mean nearest-surface distance, or nan if no contact frames."""
    hand_contact = np.asarray(hand_contact, float)
    obj_surface = np.asarray(obj_surface, float)
    contact_present = np.asarray(contact_present, bool)
    gaps = []
    for t in np.where(contact_present)[0]:
        d, _ = cKDTree(obj_surface[t]).query(hand_contact[t], k=1)
        gaps.append(float(np.mean(d)))
    return float(np.median(gaps)) if gaps else float("nan")


def penetration_depth(hand_verts, nearest_surface, surface_normal):
    """hand_verts: (T,Nh,3). nearest_surface/surface_normal: (T,Nh,3) the nearest object
    surface point and its outward normal for each hand vertex. Returns summed one-sided
    penetration: sum of max(0, (surface - hand) . normal) over all vertices/frames
    (positive when the hand vertex is inside the object)."""
    hand_verts = np.asarray(hand_verts, float)
    nearest_surface = np.asarray(nearest_surface, float)
    surface_normal = np.asarray(surface_normal, float)
    signed = ((nearest_surface - hand_verts) * surface_normal).sum(-1)   # >0 => inside
    return float(np.clip(signed, 0.0, None).sum())
```

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n hoi_recon python -m pytest tests/test_choir_fine_metrics.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add hoi_recon/choir_fine/metrics.py tests/test_choir_fine_metrics.py
git commit -m "choir_fine: contact-gap + penetration proxy metrics"
```

---

## Task 6: Full-suite green + smoke import

- [ ] **Step 1: Run the whole new suite**

Run: `conda run -n hoi_recon python -m pytest tests/test_choir_fine_*.py -q`
Expected: all pass (presets 4, phases 4, contact 4, anatomical 5, metrics 5).

- [ ] **Step 2: Confirm existing smoke tests still pass**

Run: `conda run -n hoi_recon python -m pytest tests/test_smoke.py -q`
Expected: existing tests still pass (no regression from the new package).

- [ ] **Step 3: Commit any fixups**

```bash
git add -A && git commit -m "choir_fine: foundation suite green" --allow-empty
```

---

## Self-Review notes (addressed)

- **Spec coverage:** This plan covers the spec's §3.2 shared components (contact
  correspondence, phase segmentation, anatomical), §3.1 preset registry + §6 faithful-
  preset lock test, and the §5 proxy-metric functions. NOT covered here (follow-on plans):
  the subprocess optimizer integration that consumes these (§3.1 term registry wiring),
  the config presets/run + ablation harness script (§5/§6 `ablate_fine.py`), the regression
  test vs `runs/grab_combined`, and all of Phase 2 (§4) + the HO3D GT adapter (§5).
- **Type consistency:** `get_preset` returns a dict keyed by `TERMS`; `build_correspondences`
  returns `face_id/bary/anchor/weight/valid`; `segment_phases` returns indices into
  `PHASES`; metrics take `(T,Nh,3)` arrays. These names are reused consistently across tasks.
- **No placeholders:** every code step is complete and runnable.

## Follow-on plans (not in this plan)

1. **Stage 3 optimizer integration** — refactor `joint_opt.py` into a term-registry
   optimizer (`choir_fine_opt.py`, sam3d env) consuming presets + correspondences + phases;
   add `configs/choir_faithful.yaml` + `configs/combined_v2.yaml`; wire into stage 7.
2. **Evaluation + ablation** — extend `scripts/object_confidence.py` with the metrics;
   `scripts/ablate_fine.py`; regression test vs `runs/grab_combined`.
3. **Phase 2** — flow-matching ray-depth rectifier module + GraspPair pipeline (gated on
   DexGraspNet) + synthetic-stub test.
4. **HO3D GT benchmark** adapter.
