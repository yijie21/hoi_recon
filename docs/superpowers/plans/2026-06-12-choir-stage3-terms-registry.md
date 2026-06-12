# CHOIR Stage 3 Energy Terms + Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the differentiable energy-term functions and the weighted term registry that
the CHOIR Stage-3 optimizer minimizes, plus the `choir_faithful` / `combined_v2` config
presets — all pure, CPU-unit-tested torch, so the optimizer subprocess (follow-on plan #1b)
just assembles tested pieces.

**Architecture:** New `hoi_recon/choir_fine/terms_torch.py` (each CHOIR energy term as a pure
torch function, testable on tiny CPU tensors) + `hoi_recon/choir_fine/registry.py` (sum the
weighted active terms from a preset) + two config files naming the presets. These compose
with the already-merged `hoi_recon/choir_fine/` foundation (presets, contact, phases,
anatomical, metrics). Torch is available in BOTH the `hoi_recon` env (tests run here) and the
`sam3d-objects` env (the optimizer subprocess imports these modules via PYTHONPATH in #1b), so
the same tested code runs in both. Spec: `docs/superpowers/specs/2026-06-12-choir-fine-stage-design.md` §3.1.

**Tech Stack:** Python 3.10, torch 2.11 (CPU for tests), numpy 2.2, pyyaml, pytest.

---

## File Structure

- Create: `hoi_recon/choir_fine/terms_torch.py` — differentiable CHOIR energy terms.
- Create: `hoi_recon/choir_fine/registry.py` — weighted term assembler.
- Create: `configs/choir_faithful.yaml`, `configs/combined_v2.yaml` — Stage-3 presets.
- Create: `tests/test_choir_fine_terms.py`, `tests/test_choir_fine_registry.py`,
  `tests/test_choir_fine_configs.py`.

All test commands assume the `hoi_recon` conda env:
`conda run -n hoi_recon python -m pytest ...`. pytest is already installed there.

---

## Task 1: Contact + penetration energy terms

**Files:**
- Create: `hoi_recon/choir_fine/terms_torch.py`
- Test: `tests/test_choir_fine_terms.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_choir_fine_terms.py
import torch
import pytest
from hoi_recon.choir_fine import terms_torch as T


def test_contact_loss_zero_when_anchors_coincide():
    hand_c = torch.zeros(1, 2, 3)                     # (T=1, Nc=2, 3)
    anchors = torch.zeros(1, 2, 4, 3)                 # K=4 anchors all at the hand vert
    weights = torch.full((1, 2, 4), 0.25)
    conf = torch.ones(1, 2)
    assert float(T.contact_loss(hand_c, anchors, weights, conf)) == 0.0


def test_contact_loss_is_weighted_distance():
    hand_c = torch.zeros(1, 1, 3)
    anchors = torch.zeros(1, 1, 2, 3)
    anchors[0, 0, 0, 2] = 0.1                         # anchor 0 is 0.1 away in z
    weights = torch.tensor([[[1.0, 0.0]]])            # all weight on anchor 0
    conf = torch.ones(1, 1)
    # loss = conf * (w0 * 0.1^2) / conf = 0.01
    assert float(T.contact_loss(hand_c, anchors, weights, conf)) == pytest.approx(0.01)


def test_contact_loss_confidence_normalizes():
    # two verts, one with confidence 0 -> ignored
    hand_c = torch.zeros(1, 2, 3)
    anchors = torch.zeros(1, 2, 1, 3)
    anchors[0, 0, 0, 0] = 0.2                         # vert 0 anchor 0.2 away
    anchors[0, 1, 0, 0] = 1.0                         # vert 1 far, but conf 0
    weights = torch.ones(1, 2, 1)
    conf = torch.tensor([[1.0, 0.0]])
    assert float(T.contact_loss(hand_c, anchors, weights, conf)) == pytest.approx(0.04)


def test_penetration_zero_outside():
    hand = torch.tensor([[[0.0, 0.0, 0.02]]])         # 2cm outside (above) the surface
    surf = torch.zeros(1, 1, 3)
    nrm = torch.tensor([[[0.0, 0.0, 1.0]]])
    assert float(T.penetration_loss(hand, surf, nrm)) == 0.0


def test_penetration_clamps_with_tolerance():
    hand = torch.tensor([[[0.0, 0.0, -0.01]]])        # 1cm inside
    surf = torch.zeros(1, 1, 3)
    nrm = torch.tensor([[[0.0, 0.0, 1.0]]])
    # signed = (surf-hand).n = 0.01 ; (0.01 - eps0.005).clamp(0,0.04) = 0.005
    assert float(T.penetration_loss(hand, surf, nrm)) == pytest.approx(0.005)


def test_penetration_normalizes_normal():
    hand = torch.tensor([[[0.0, 0.0, -0.01]]])
    surf = torch.zeros(1, 1, 3)
    nrm = torch.tensor([[[0.0, 0.0, 2.0]]])           # non-unit normal must not scale depth
    assert float(T.penetration_loss(hand, surf, nrm)) == pytest.approx(0.005)


def test_terms_are_differentiable():
    hand_c = torch.zeros(1, 1, 3, requires_grad=True)
    anchors = torch.ones(1, 1, 1, 3)
    loss = T.contact_loss(hand_c, anchors, torch.ones(1, 1, 1), torch.ones(1, 1))
    loss.backward()
    assert hand_c.grad is not None and torch.isfinite(hand_c.grad).all()
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n hoi_recon python -m pytest tests/test_choir_fine_terms.py -v`
Expected: FAIL — `ModuleNotFoundError: hoi_recon.choir_fine.terms_torch`.

- [ ] **Step 3: Write minimal implementation**

```python
# hoi_recon/choir_fine/terms_torch.py
"""Differentiable CHOIR Stage-3 energy terms (arXiv:2605.20992 §4.3). Each is a pure torch
function over a per-frame optimization state, testable on tiny CPU tensors. The optimizer
(follow-on plan) computes each term and sums them through registry.assemble_energy."""
from __future__ import annotations

import torch


def contact_loss(hand_c, anchors, weights, confidence):
    """CHOIR Eq 6 soft barycentric contact loss.
      hand_c:     (T,Nc,3) hand contact vertices
      anchors:    (T,Nc,K,3) object-surface anchor points (top-K per vertex)
      weights:    (T,Nc,K) softmax weights over anchors (sum to 1 per vertex)
      confidence: (T,Nc) per-vertex contact confidence/gate
    Returns scalar = sum_{t,i} c * sum_k w * ||v-a||^2  /  sum_{t,i} c."""
    diff = hand_c.unsqueeze(-2) - anchors              # (T,Nc,K,3)
    sq = (diff ** 2).sum(-1)                            # (T,Nc,K)
    per_vert = (weights * sq).sum(-1)                   # (T,Nc)
    return (confidence * per_vert).sum() / confidence.sum().clamp(min=1e-8)


def penetration_loss(hand_verts, nearest_surface, surface_normal, eps=0.005, clip=0.04):
    """CHOIR Eq 23 one-sided non-penetration. Penalizes hand vertices inside the object
    beyond a tolerance eps, per-vertex residual clamped to `clip`. Normals are normalized
    internally so non-unit normals do not scale the depth.
      hand_verts/nearest_surface/surface_normal: (T,Nh,3)."""
    n = surface_normal / surface_normal.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    signed = ((nearest_surface - hand_verts) * n).sum(-1)   # >0 when hand is inside
    return (signed - eps).clamp(min=0.0, max=clip).mean()
```

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n hoi_recon python -m pytest tests/test_choir_fine_terms.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add hoi_recon/choir_fine/terms_torch.py tests/test_choir_fine_terms.py
git commit -m "choir_fine: contact + penetration energy terms"
```

---

## Task 2: Temporal + keypoint-reprojection terms

**Files:**
- Modify: `hoi_recon/choir_fine/terms_torch.py`
- Test: `tests/test_choir_fine_terms.py` (add tests)

- [ ] **Step 1: Write the failing test (append to the file)**

```python
# append to tests/test_choir_fine_terms.py

def test_velocity_zero_for_constant():
    x = torch.ones(5, 3)
    assert float(T.velocity_loss(x)) == 0.0


def test_velocity_positive_for_ramp():
    x = torch.arange(5.0).reshape(5, 1).repeat(1, 3)   # linear ramp, step 1 each frame
    assert float(T.velocity_loss(x)) == pytest.approx(1.0)   # all first-diffs are 1; mean=1


def test_acceleration_zero_for_linear_ramp():
    x = torch.arange(5.0).reshape(5, 1).repeat(1, 3)   # constant velocity -> zero accel
    assert float(T.acceleration_loss(x)) == pytest.approx(0.0, abs=1e-6)


def test_acceleration_positive_for_curve():
    x = (torch.arange(5.0) ** 2).reshape(5, 1)         # quadratic -> constant 2nd diff
    assert float(T.acceleration_loss(x)) > 0.0


def test_keypoint_reproj_zero_when_aligned():
    K = torch.tensor([[100.0, 0, 50.0], [0, 100.0, 50.0], [0, 0, 1.0]])
    joints = torch.tensor([[[0.0, 0.0, 1.0]]])         # projects to (cx,cy)=(50,50)
    kp2d = torch.tensor([[[50.0, 50.0]]])
    valid = torch.ones(1)
    assert float(T.keypoint_reproj_loss(joints, kp2d, K, valid)) == pytest.approx(0.0)


def test_keypoint_reproj_bounded_and_positive_when_off():
    K = torch.tensor([[100.0, 0, 50.0], [0, 100.0, 50.0], [0, 0, 1.0]])
    joints = torch.tensor([[[0.0, 0.0, 1.0]]])
    kp2d = torch.tensor([[[80.0, 50.0]]])              # 30px off in u
    valid = torch.ones(1)
    v = float(T.keypoint_reproj_loss(joints, kp2d, K, valid))
    assert 0.0 < v < 1.0                                # Geman-McClure is bounded in [0,1)
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n hoi_recon python -m pytest tests/test_choir_fine_terms.py -k "velocity or acceleration or keypoint" -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'velocity_loss'`.

- [ ] **Step 3: Write minimal implementation (append to terms_torch.py)**

```python
# append to hoi_recon/choir_fine/terms_torch.py

def velocity_loss(x):
    """Mean squared first temporal difference. x: (T, ...). Smoothness on any per-frame
    quantity (MANO pose, object rotation/translation). CHOIR L_temp velocity terms."""
    return ((x[1:] - x[:-1]) ** 2).mean()


def acceleration_loss(x):
    """Mean squared second temporal difference. x: (T, ...). Kills residual jitter that
    velocity terms leave. CHOIR L_temp acceleration terms."""
    return ((x[2:] - 2 * x[1:-1] + x[:-2]) ** 2).mean()


def keypoint_reproj_loss(joints_cam, kp2d, K, valid, sigma_px=60.0):
    """Geman-McClure-robust 2D keypoint reprojection (CHOIR L^h_2D). Projects camera-frame
    joints with K and compares to kp2d; the robust kernel r2/(r2+sigma^2) is bounded in
    [0,1) so outlier joints can't dominate.
      joints_cam: (T,J,3); kp2d: (T,J,2) full-image px; K: (3,3); valid: (T,) frame mask."""
    z = joints_cam[..., 2].clamp(min=1e-4)
    u = K[0, 0] * joints_cam[..., 0] / z + K[0, 2]
    v = K[1, 1] * joints_cam[..., 1] / z + K[1, 2]
    r2 = (u - kp2d[..., 0]) ** 2 + (v - kp2d[..., 1]) ** 2
    s2 = float(sigma_px) ** 2
    return ((r2 / (r2 + s2)) * valid[:, None]).mean()
```

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n hoi_recon python -m pytest tests/test_choir_fine_terms.py -v`
Expected: 13 passed (7 from Task 1 + 6 new).

- [ ] **Step 5: Commit**

```bash
git add hoi_recon/choir_fine/terms_torch.py tests/test_choir_fine_terms.py
git commit -m "choir_fine: temporal + keypoint-reprojection energy terms"
```

---

## Task 3: Weighted term registry

**Files:**
- Create: `hoi_recon/choir_fine/registry.py`
- Test: `tests/test_choir_fine_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_choir_fine_registry.py
import torch
import pytest
from hoi_recon.choir_fine import registry


def test_assemble_sums_weighted_active_terms():
    weights = {"contact": 10.0, "pen": 500.0, "sil": 0.0}
    values = {"contact": torch.tensor(2.0), "pen": torch.tensor(0.1),
              "sil": torch.tensor(9.9)}
    total = registry.assemble_energy(weights, values)
    # 10*2 + 500*0.1 + (sil weight 0 -> skipped) = 20 + 50 = 70
    assert float(total) == pytest.approx(70.0)


def test_assemble_skips_zero_weight_terms_entirely():
    """A zero-weight term must not contribute even if its value is huge/NaN-prone."""
    weights = {"contact": 1.0, "sil": 0.0}
    values = {"contact": torch.tensor(1.0), "sil": torch.tensor(float("inf"))}
    total = registry.assemble_energy(weights, values)
    assert float(total) == pytest.approx(1.0)          # inf*0 skipped, not nan


def test_assemble_raises_on_value_without_weight():
    with pytest.raises(KeyError):
        registry.assemble_energy({"contact": 1.0}, {"contact": torch.tensor(1.0),
                                                    "mystery": torch.tensor(1.0)})


def test_assemble_is_differentiable():
    x = torch.tensor(3.0, requires_grad=True)
    total = registry.assemble_energy({"a": 2.0}, {"a": x * x})
    total.backward()
    assert float(x.grad) == pytest.approx(12.0)        # d(2*x^2)/dx = 4x = 12
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n hoi_recon python -m pytest tests/test_choir_fine_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: hoi_recon.choir_fine.registry`.

- [ ] **Step 3: Write minimal implementation**

```python
# hoi_recon/choir_fine/registry.py
"""Weighted energy-term registry: sum the active (non-zero-weight) terms of an optimization
step. The optimizer computes each term's scalar value into a dict, and this assembles the
total loss using a preset's weight dict (hoi_recon.choir_fine.presets). Zero-weight terms
are skipped entirely (their value never enters the graph), so an inactive term cannot inject
NaN/inf or waste a backward pass."""
from __future__ import annotations


def assemble_energy(weights, values):
    """weights: {term_name: float}. values: {term_name: scalar tensor}. Returns the summed
    weighted total (a scalar tensor, or 0.0 if no active terms). Raises KeyError if a value
    has no corresponding weight (guards against typos / unregistered terms)."""
    missing = [k for k in values if k not in weights]
    if missing:
        raise KeyError(f"term value(s) without a weight: {missing}")
    total = 0.0
    for name, val in values.items():
        w = weights[name]
        if w != 0:
            total = total + w * val
    return total
```

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n hoi_recon python -m pytest tests/test_choir_fine_registry.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add hoi_recon/choir_fine/registry.py tests/test_choir_fine_registry.py
git commit -m "choir_fine: weighted term registry (assemble_energy)"
```

---

## Task 4: Stage-3 config presets

**Files:**
- Create: `configs/choir_faithful.yaml`, `configs/combined_v2.yaml`
- Test: `tests/test_choir_fine_configs.py`

> Context: the repo loads YAML via `hoi_recon.config.load_config(path)` which deep-merges over
> built-in defaults and returns an attribute-accessible `Config`. These two configs set
> `fine_preset` (the name the Stage-3 optimizer will resolve via `presets.get_preset`) on top
> of the existing combined pipeline. `choir_faithful` is the locked CHOIR baseline (our
> object/hand improvements OFF where they differ from CHOIR); `combined_v2` keeps the combined
> pipeline + turns our improvement toggles ON.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_choir_fine_configs.py
from hoi_recon.config import load_config
from hoi_recon.choir_fine import presets


def test_choir_faithful_config_names_locked_preset():
    cfg = load_config("configs/choir_faithful.yaml")
    assert cfg.get("fine_preset") == "choir_faithful"
    w = presets.get_preset(cfg["fine_preset"])
    assert w["contact"] == 1000.0 and w["hand_sil"] == 0.0   # faithful: no hand-sil


def test_combined_v2_config_enables_toggles():
    cfg = load_config("configs/combined_v2.yaml")
    assert cfg.get("fine_preset") == "combined_v2"
    w = presets.get_preset(cfg["fine_preset"])
    assert w["hand_sil"] > 0.0                                # our improvement on


def test_both_configs_are_real_mode_with_a_resolvable_preset():
    for path in ("configs/choir_faithful.yaml", "configs/combined_v2.yaml"):
        cfg = load_config(path)
        assert cfg.mock is False
        presets.get_preset(cfg["fine_preset"])               # must not raise
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n hoi_recon python -m pytest tests/test_choir_fine_configs.py -v`
Expected: FAIL — `FileNotFoundError` for `configs/choir_faithful.yaml`.

- [ ] **Step 3: Write the two config files**

```yaml
# configs/choir_faithful.yaml
# Faithful CHOIR fine-stage baseline: combined coarse + CHOIR Stage-3 weights (locked).
# fine_preset resolves to hoi_recon.choir_fine.presets.CHOIR_FAITHFUL.
mock: false
coarse: choir
fine_preset: choir_faithful
backend:
  hand: hamer
  object: sam3d
  depth: moge
  object_pose: choir_tracker
  sam3d_env: sam3d-objects
optim:
  differentiable: true
contact:
  dist_thresh_m: 0.02
  normal_thresh_deg: 60.0
smoothing:
  window: 5
```

```yaml
# configs/combined_v2.yaml
# Our best pipeline + CHOIR Stage-3 structure with our improvement toggles ON.
# fine_preset resolves to hoi_recon.choir_fine.presets.COMBINED_V2 (faithful + hand_sil etc.).
mock: false
coarse: choir
fine_preset: combined_v2
backend:
  hand: hamer
  object: sam3d
  depth: moge
  object_pose: render_compare      # our stronger object tracker
  sam3d_env: sam3d-objects
optim:
  differentiable: true
contact:
  dist_thresh_m: 0.02
  normal_thresh_deg: 60.0
smoothing:
  window: 5
```

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n hoi_recon python -m pytest tests/test_choir_fine_configs.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add configs/choir_faithful.yaml configs/combined_v2.yaml tests/test_choir_fine_configs.py
git commit -m "choir_fine: choir_faithful + combined_v2 Stage-3 config presets"
```

---

## Task 5: Full-suite green

- [ ] **Step 1: Run the whole choir_fine suite + smoke**

Run: `conda run -n hoi_recon python -m pytest tests/test_choir_fine_*.py tests/test_smoke.py -q`
Expected: all pass (foundation 23 + terms 13 + registry 4 + configs 3 + smoke 5 = 48).

- [ ] **Step 2: Commit (empty-ok checkpoint)**

```bash
git add -A && git commit -m "choir_fine: terms + registry + configs suite green" --allow-empty
```

---

## Self-Review notes (addressed)

- **Spec coverage:** Covers spec §3.1 energy-term registry (the differentiable terms +
  `assemble_energy`) and the `choir_faithful` / `combined_v2` presets at the config level.
  NOT covered here (follow-on #1b): wiring these into the subprocess optimizer
  (`choir_fine_opt.py` refactor of `joint_opt.py`), the contact-cache rebuild loop in the
  optimizer, per-group LR Adam + 800 iters, the smoke/integration run, the ablation harness
  (`scripts/ablate_fine.py`), and the `combined` regression vs `runs/grab_combined`.
- **Type consistency:** term functions take `(T,...)` tensors and return scalar tensors;
  `assemble_energy(weights: dict, values: dict)`; configs expose `fine_preset` consumed by
  `presets.get_preset`. Names reused consistently.
- **No placeholders:** every code/test step is complete and runnable.

## Follow-on plans (not in this plan)

1. **#1b — Stage 3 optimizer integration:** refactor `scripts/subprocess_entries/sam-3d-objects/joint_opt.py`
   into `choir_fine_opt.py` that imports `hoi_recon.choir_fine.{terms_torch,registry,contact,
   phases,anatomical}` (via PYTHONPATH), builds correspondences + phases, runs per-group Adam
   (object 3e-4 / finger 5e-4 / wrist 5e-5, 800 iters) with periodic contact-cache rebuild,
   summing terms via `assemble_energy(get_preset(cfg.fine_preset), values)`. Wire into stage 7;
   verify with a low-iter smoke run on cached `runs/grab` stages.
2. **#2 — Evaluation + ablation:** extend `scripts/object_confidence.py` with `choir_fine.metrics`;
   `scripts/ablate_fine.py`; `combined` regression vs `runs/grab_combined`.
3. **#3 — Phase 2 generative rectifier** + GraspPair pipeline (gated on DexGraspNet).
4. **#4 — HO3D GT benchmark adapter.**
