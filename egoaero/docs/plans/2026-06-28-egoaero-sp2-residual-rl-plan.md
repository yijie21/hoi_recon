# EgoAERO SP2 — Two-stage Residual RL Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn one reconstructed hand-object clip (SP1 contract output) into a trained dexterous policy via EgoAERO's two-stage residual RL (App D), in real physics (MuJoCo) with real PPO (stable-baselines3) and a vendored Shadow Hand, evaluated with the App-H metrics.

**Architecture:** A self-contained `egoaero/egoaero/policy/` subsystem. Pure-numpy cores (`rewards.py`, `metrics.py`, `retarget.py`) are unit-tested on the base box. Heavy modules (`hand_model.py`, `task.py`, `env.py`, `train.py`, `evaluate.py`) lazy-import mujoco/torch/sb3/gymnasium; their tests are gated with `pytest.importorskip` so the existing 41-test base suite stays green when the stack is absent. Stage I trains a hand-tracking policy; Stage II trains a residual policy on top of the frozen Stage-I policy with object + contact rewards.

**Tech Stack:** Python 3.9+, numpy (cores); mujoco>=3, torch (CUDA), stable-baselines3>=2, gymnasium>=1 (sim). Hardware available: 2× RTX 4090, 48 CPU, 251 GB RAM.

## Global Constraints

- **Self-contained:** `egoaero/` MUST NOT import from `render_and_compare/` or any sibling. SP2 lives under `egoaero/egoaero/policy/` + `egoaero/assets/`.
- **Faithful where specified; documented defaults at gaps**, logged in `egoaero/ASSUMPTIONS.md`. Faithful: App-D reward terms, App-H metrics + SR + thresholds (`τ_r=30°, τ_t=3cm, τ_j=8cm, τ_ft=6cm`), residual composition `a = a^I + Δa^R`, two-stage structure, early termination. Documented defaults: PPO/SB3, network arch, every weight (`w/λ/η/μ`), PPO hyperparams, budget, retarget correspondence + IK damping, object-geometry encoding, early-term thresholds, MuJoCo↔IsaacGym + Shadow↔Inspire substitutions.
- **Lazy heavy imports:** `rewards.py`/`metrics.py` import only numpy. `mujoco`/`torch`/`stable_baselines3`/`gymnasium` are imported INSIDE functions/methods of the sim modules, never at `egoaero` package import time.
- **Test gating:** sim tests begin with `mj = pytest.importorskip("mujoco")` etc., so they SKIP (not fail) without the stack. The base suite (`python -m pytest egoaero/tests/ -q` excluding policy-sim) must remain green.
- **Units:** metres / radians internally; metrics in deg/cm.
- **Reference frame:** the SP1 contract is in the table frame; all reference trajectories and the MuJoCo world share that frame.
- **Commits:** one per task, prefix `egoaero:`.

## SP1 contract consumed (exact, from `egoaero/egoaero/contract.py`)

`<run>/contract/`: `hand_mano.npz` → `verts[T,Nh,3]`, `joints[T,21,3]`; `object_traj.npz` → `obj_poses_t[T,4,4]`; `contact.npz` → `contact_mask[T,Nh]` bool; `object_mesh.obj`; `manifest.json` → `frames`.
Joint layout (from `core/hand.py`): wrist = index 0; each finger has 4 joints in order `FINGERS=["thumb","index","middle","ring","little"]`, so **fingertip joint indices = [4, 8, 12, 16, 20]** (thumb,index,middle,ring,little). Wrist pose = `joints[t,0]` (position); wrist orientation is not in the contract → use identity / velocity-derived heading (documented default).

---

## File structure

```
egoaero/egoaero/policy/
  __init__.py
  rewards.py     Tasks 2-3
  metrics.py     Task 4
  retarget.py    Task 5
  hand_model.py  Task 6
  task.py        Task 7
  env.py         Tasks 8-9
  train.py       Task 10
  evaluate.py    Task 11
egoaero/assets/shadow_hand/        Task 1 (vendored Menagerie MJCF + meshes)
egoaero/egoaero/configs/policy.yaml Task 1
egoaero/scripts/setup_rl.sh        Task 1
egoaero/tests/policy/              one test file per task
```

---

### Task 1: RL deps, setup script, vendored Shadow Hand asset, policy config

**Files:**
- Create: `egoaero/scripts/setup_rl.sh`, `egoaero/egoaero/configs/policy.yaml`, `egoaero/egoaero/policy/__init__.py`, `egoaero/tests/policy/__init__.py`
- Modify: `egoaero/pyproject.toml` (add optional-extra `rl`)
- Test: `egoaero/tests/policy/test_asset.py`

**Interfaces:**
- Produces: vendored MJCF at `egoaero/assets/shadow_hand/right_hand.xml` (+ meshes); `policy.yaml` with `reward`, `ppo`, `budget`, `term`, `retarget` sub-blocks (documented defaults).

- [ ] **Step 1: Write the setup script and config (no test yet — infra)**

`egoaero/scripts/setup_rl.sh`:
```bash
#!/usr/bin/env bash
# Install the SP2 RL stack and vendor the Shadow Hand model. Run once.
set -euo pipefail
python -m pip install "mujoco>=3" "stable-baselines3>=2" "gymnasium>=1"
# torch with CUDA (box has RTX 4090s); falls back to default index if cu wheels unavailable
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124 || python -m pip install torch
DEST="$(cd "$(dirname "$0")/.." && pwd)/assets/shadow_hand"
if [ ! -f "$DEST/right_hand.xml" ]; then
  TMP="$(mktemp -d)"
  git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie "$TMP/mm"
  mkdir -p "$DEST"
  cp -r "$TMP/mm/shadow_hand/." "$DEST/"
  rm -rf "$TMP"
fi
echo "RL stack installed; Shadow Hand vendored at $DEST"
```
Make executable: `chmod +x egoaero/scripts/setup_rl.sh`.

`egoaero/egoaero/configs/policy.yaml` (all values DOCUMENTED defaults — App D/G give none):
```yaml
reward:
  w_w: 1.0
  w_f: 1.0
  w_s: 0.1
  lam_p: 40.0      # wrist position (1/m^2 scale)
  lam_R: 1.0       # wrist orientation (1/rad^2)
  lam_v: 1.0       # wrist velocity
  lam_k: 40.0      # finger keypoint
  lam_a: 0.1       # action smoothness
  lam_tau: 0.001   # torque*qvel power
  eta_I: 1.0
  eta_o: 1.0
  eta_c: 1.0
  eta_delta: 0.1
  mu_p: 40.0
  mu_R: 1.0
  mu_v: 1.0
  mu_d: 200.0      # contact distance (1/m^2)
  mu_F: 1.0        # contact force saturation
  mu_delta: 1.0    # residual regularization
ppo:
  policy: "MlpPolicy"
  n_steps: 2048
  batch_size: 256
  learning_rate: 0.0003
  net_arch: [256, 256]
budget:
  smoke: 512       # total timesteps per stage (tests)
  real: 1500000    # total timesteps per stage (GPU run)
term:               # early-termination thresholds (Stage II)
  obj_pos_err_m: 0.15
  hand_err_m: 0.15
  penetration_m: 0.03
retarget:
  ik_damping: 0.05
  ik_iters: 50
  contact_active_dist_m: 0.02   # finger is reference-active if fingertip within this of object
```

`egoaero/egoaero/policy/__init__.py` and `egoaero/tests/policy/__init__.py`: empty files.

In `egoaero/pyproject.toml`, add under `[project.optional-dependencies]`:
```toml
rl = ["mujoco>=3", "stable-baselines3>=2", "gymnasium>=1", "torch"]
```

- [ ] **Step 2: Run the setup script (installs stack + vendors asset)**

Run: `bash egoaero/scripts/setup_rl.sh`
Expected: ends with "RL stack installed; Shadow Hand vendored at …/assets/shadow_hand", and `egoaero/assets/shadow_hand/right_hand.xml` exists. (Also add `egoaero/assets/shadow_hand/` is large — ensure it is NOT gitignored by the root rules; meshes are needed. If the repo `.gitignore` excludes them, add `!egoaero/assets/**`.)

- [ ] **Step 3: Write the gated asset test**

`egoaero/tests/policy/test_asset.py`:
```python
import os, pytest

def test_shadow_hand_mjcf_loads():
    mujoco = pytest.importorskip("mujoco")
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    xml = os.path.join(here, "assets", "shadow_hand", "right_hand.xml")
    if not os.path.exists(xml):
        pytest.skip("shadow hand not vendored; run scripts/setup_rl.sh")
    model = mujoco.MjModel.from_xml_path(xml)
    assert model.nu > 0          # has actuators
    assert model.nsite > 0       # has sites (fingertips)
```

- [ ] **Step 4: Run the test**

Run: `python -m pytest egoaero/tests/policy/test_asset.py -v`
Expected: PASS (after setup) or SKIP (if mujoco/asset absent). Confirm PASS on this box (run setup first).

- [ ] **Step 5: Commit** (commit the vendored asset + config + script + test)

```bash
git add egoaero/scripts/setup_rl.sh egoaero/egoaero/configs/policy.yaml egoaero/egoaero/policy/__init__.py egoaero/tests/policy egoaero/pyproject.toml egoaero/assets/shadow_hand .gitignore
git commit -m "egoaero: SP2 RL deps + setup script + vendored Shadow Hand + policy config"
```

---

### Task 2: `rewards.py` — Stage-I reward terms

**Files:**
- Create: `egoaero/egoaero/policy/rewards.py`
- Test: `egoaero/tests/policy/test_rewards_stage1.py`

**Interfaces:**
- Produces (pure numpy):
  - `geodesic_rad(Ra, Rb) -> float` (rotation geodesic; radians).
  - `r_wrist(p, R, pdot, p_h, R_h, pdot_h, lam_p, lam_R, lam_v) -> float`
  - `r_finger(x_kpts, x_kpts_h, lam_k) -> float` (x_kpts: [K,3])
  - `r_smooth(a, a_prev, torque, qdot, lam_a, lam_tau) -> float`
  - `r_stage1(r_wrist_v, r_finger_v, r_smooth_v, w_w, w_f, w_s) -> float`

- [ ] **Step 1: Write the failing test**

```python
# egoaero/tests/policy/test_rewards_stage1.py
import numpy as np
from egoaero.policy import rewards as R

def test_r_wrist_perfect_is_one_and_decreasing():
    p = np.zeros(3); Rm = np.eye(3); v = np.zeros(3)
    assert abs(R.r_wrist(p, Rm, v, p, Rm, v, 40.0, 1.0, 1.0) - 1.0) < 1e-12
    worse = R.r_wrist(p + 0.05, Rm, v, p, Rm, v, 40.0, 1.0, 1.0)
    assert 0.0 < worse < 1.0

def test_r_finger_mean_of_exp():
    x = np.zeros((5, 3)); xh = np.zeros((5, 3))
    assert abs(R.r_finger(x, xh, 40.0) - 1.0) < 1e-12
    xh2 = xh.copy(); xh2[0] = [0.1, 0, 0]
    assert R.r_finger(x, xh2, 40.0) < 1.0

def test_r_smooth_and_stage1_weighting():
    a = np.zeros(4); s = R.r_smooth(a, a, np.zeros(4), np.zeros(4), 0.1, 0.001)
    assert abs(s - 1.0) < 1e-12
    assert abs(R.r_stage1(1.0, 1.0, 1.0, 1.0, 1.0, 0.1) - 2.1) < 1e-12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest egoaero/tests/policy/test_rewards_stage1.py -v`
Expected: FAIL (`ModuleNotFoundError: egoaero.policy.rewards`).

- [ ] **Step 3: Write minimal implementation**

```python
# egoaero/egoaero/policy/rewards.py
"""App-D reward terms for EgoAERO two-stage residual RL. Pure numpy, no heavy deps."""
from __future__ import annotations
import numpy as np


def geodesic_rad(Ra, Rb):
    Rrel = np.asarray(Ra).T @ np.asarray(Rb)
    cos = (np.trace(Rrel) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def r_wrist(p, R, pdot, p_h, R_h, pdot_h, lam_p, lam_R, lam_v):
    dp = np.sum((np.asarray(p) - np.asarray(p_h)) ** 2)
    dR = geodesic_rad(R, R_h) ** 2
    dv = np.sum((np.asarray(pdot) - np.asarray(pdot_h)) ** 2)
    return float(np.exp(-lam_p * dp - lam_R * dR - lam_v * dv))


def r_finger(x_kpts, x_kpts_h, lam_k):
    x = np.asarray(x_kpts); xh = np.asarray(x_kpts_h)
    d2 = np.sum((x - xh) ** 2, axis=1)
    return float(np.mean(np.exp(-lam_k * d2)))


def r_smooth(a, a_prev, torque, qdot, lam_a, lam_tau):
    da = np.sum((np.asarray(a) - np.asarray(a_prev)) ** 2)
    power = np.sum(np.abs(np.asarray(torque) * np.asarray(qdot)))
    return float(np.exp(-lam_a * da - lam_tau * power))


def r_stage1(r_wrist_v, r_finger_v, r_smooth_v, w_w, w_f, w_s):
    return float(w_w * r_wrist_v + w_f * r_finger_v + w_s * r_smooth_v)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest egoaero/tests/policy/test_rewards_stage1.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add egoaero/egoaero/policy/rewards.py egoaero/tests/policy/test_rewards_stage1.py
git commit -m "egoaero: policy rewards Stage-I terms (App D)"
```

---

### Task 3: `rewards.py` — Stage-II reward terms

**Files:**
- Modify: `egoaero/egoaero/policy/rewards.py` (append)
- Test: `egoaero/tests/policy/test_rewards_stage2.py`

**Interfaces:**
- Produces: `r_obj(p_o, R_o, podot, p_ref, R_ref, podot_ref, mu_p, mu_R, mu_v) -> float`; `r_contact(dists, forces, active, mu_d, mu_F) -> float` (`dists`,`forces` are per-finger arrays indexed like `FINGERS`; `active` is a list/array of finger indices that are reference-active; returns 0.0 if `active` empty); `r_res(delta_a, mu_delta) -> float`; `r_stage2(r1, r_obj_v, r_contact_v, r_res_v, eta_I, eta_o, eta_c, eta_delta) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# egoaero/tests/policy/test_rewards_stage2.py
import numpy as np
from egoaero.policy import rewards as R

def test_r_obj_perfect_is_one():
    p = np.zeros(3); Rm = np.eye(3); v = np.zeros(3)
    assert abs(R.r_obj(p, Rm, v, p, Rm, v, 40.0, 1.0, 1.0) - 1.0) < 1e-12

def test_r_contact_empty_active_is_zero():
    assert R.r_contact(np.zeros(5), np.zeros(5), [], 200.0, 1.0) == 0.0

def test_r_contact_rewards_close_and_forceful():
    dists = np.array([0.0, 0.2, 0.2, 0.2, 0.2])     # thumb touching
    forces = np.array([5.0, 0, 0, 0, 0])
    val = R.r_contact(dists, forces, [0], 200.0, 1.0)   # only thumb active
    assert 0.0 < val <= 1.0
    far = R.r_contact(np.array([0.2, 0, 0, 0, 0]), forces, [0], 200.0, 1.0)
    assert far < val                                   # farther -> lower

def test_r_res_and_stage2_weighting():
    assert abs(R.r_res(np.zeros(4), 1.0) - 1.0) < 1e-12
    assert abs(R.r_stage2(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1) - 3.1) < 1e-12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest egoaero/tests/policy/test_rewards_stage2.py -v`
Expected: FAIL (`AttributeError: r_obj`).

- [ ] **Step 3: Write minimal implementation** (append to `rewards.py`)

```python
def r_obj(p_o, R_o, podot, p_ref, R_ref, podot_ref, mu_p, mu_R, mu_v):
    dp = np.sum((np.asarray(p_o) - np.asarray(p_ref)) ** 2)
    dR = geodesic_rad(R_o, R_ref) ** 2
    dv = np.sum((np.asarray(podot) - np.asarray(podot_ref)) ** 2)
    return float(np.exp(-mu_p * dp - mu_R * dR - mu_v * dv))


def r_contact(dists, forces, active, mu_d, mu_F):
    active = list(active)
    if not active:
        return 0.0
    d = np.asarray(dists, float); f = np.asarray(forces, float)
    vals = [np.exp(-mu_d * d[i] ** 2) * (1.0 - np.exp(-mu_F * abs(f[i]))) for i in active]
    return float(np.mean(vals))


def r_res(delta_a, mu_delta):
    return float(np.exp(-mu_delta * np.sum(np.asarray(delta_a) ** 2)))


def r_stage2(r1, r_obj_v, r_contact_v, r_res_v, eta_I, eta_o, eta_c, eta_delta):
    return float(eta_I * r1 + eta_o * r_obj_v + eta_c * r_contact_v + eta_delta * r_res_v)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest egoaero/tests/policy/test_rewards_stage2.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add egoaero/egoaero/policy/rewards.py egoaero/tests/policy/test_rewards_stage2.py
git commit -m "egoaero: policy rewards Stage-II terms (App D)"
```

---

### Task 4: `metrics.py` — App-H evaluation metrics

**Files:**
- Create: `egoaero/egoaero/policy/metrics.py`
- Test: `egoaero/tests/policy/test_metrics.py`

**Interfaces:**
- Produces: `object_rotation_error(R_seq, R_ref_seq) -> float` (deg); `object_translation_error(p_seq, p_ref_seq) -> float` (cm); `mean_joint_error(xj_seq, xj_ref_seq) -> float` (cm); `mean_fingertip_error(xf_seq, xf_ref_seq) -> float` (cm); `success(Er, Et, Ej, Eft, tau_r=30.0, tau_t=3.0, tau_j=8.0, tau_ft=6.0) -> bool`; `success_rate(rows, taus=None) -> float` where `rows` is a list of `(Er,Et,Ej,Eft)`.

- [ ] **Step 1: Write the failing test**

```python
# egoaero/tests/policy/test_metrics.py
import numpy as np
from egoaero.policy import metrics as M

def test_translation_error_cm():
    p = np.zeros((3, 3)); pref = np.zeros((3, 3)); pref[:, 0] = 0.01   # 1cm off
    assert abs(M.object_translation_error(p, pref) - 1.0) < 1e-9       # cm

def test_rotation_error_deg():
    I = np.tile(np.eye(3), (2, 1, 1))
    Rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    seq = np.stack([Rz, Rz])
    assert abs(M.object_rotation_error(seq, I) - 90.0) < 1e-3

def test_joint_and_fingertip_cm():
    a = np.zeros((2, 5, 3)); b = a.copy(); b[:, :, 0] = 0.02          # 2cm
    assert abs(M.mean_joint_error(a, b) - 2.0) < 1e-9
    assert abs(M.mean_fingertip_error(a, b) - 2.0) < 1e-9

def test_success_thresholds_and_rate():
    assert M.success(10.0, 1.0, 2.0, 1.0) is True
    assert M.success(40.0, 1.0, 2.0, 1.0) is False     # Er over 30
    rows = [(10, 1, 2, 1), (40, 1, 2, 1)]
    assert abs(M.success_rate(rows) - 0.5) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest egoaero/tests/policy/test_metrics.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write minimal implementation**

```python
# egoaero/egoaero/policy/metrics.py
"""App-H evaluation metrics for EgoAERO policy rollouts. Pure numpy. Units: deg, cm."""
from __future__ import annotations
import numpy as np
from .rewards import geodesic_rad


def object_rotation_error(R_seq, R_ref_seq):
    R_seq = np.asarray(R_seq); R_ref_seq = np.asarray(R_ref_seq)
    degs = [np.degrees(geodesic_rad(R_seq[t], R_ref_seq[t])) for t in range(len(R_seq))]
    return float(np.mean(degs))


def object_translation_error(p_seq, p_ref_seq):
    d = np.linalg.norm(np.asarray(p_seq) - np.asarray(p_ref_seq), axis=-1)
    return float(np.mean(d) * 100.0)   # m -> cm


def mean_joint_error(xj_seq, xj_ref_seq):
    d = np.linalg.norm(np.asarray(xj_seq) - np.asarray(xj_ref_seq), axis=-1)
    return float(np.mean(d) * 100.0)


def mean_fingertip_error(xf_seq, xf_ref_seq):
    d = np.linalg.norm(np.asarray(xf_seq) - np.asarray(xf_ref_seq), axis=-1)
    return float(np.mean(d) * 100.0)


def success(Er, Et, Ej, Eft, tau_r=30.0, tau_t=3.0, tau_j=8.0, tau_ft=6.0):
    return bool(Er < tau_r and Et < tau_t and Ej < tau_j and Eft < tau_ft)


def success_rate(rows, taus=None):
    taus = taus or (30.0, 3.0, 8.0, 6.0)
    if not rows:
        return 0.0
    ok = [success(*row, *taus) for row in rows]
    return float(np.mean(ok))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest egoaero/tests/policy/test_metrics.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add egoaero/egoaero/policy/metrics.py egoaero/tests/policy/test_metrics.py
git commit -m "egoaero: policy App-H metrics (Er/Et/Ej/Eft/SR)"
```

---

### Task 5: `retarget.py` — damped-least-squares IK (FK-injected, base-box testable)

**Files:**
- Create: `egoaero/egoaero/policy/retarget.py`
- Test: `egoaero/tests/policy/test_retarget.py`

**Interfaces:**
- Produces: `ik_step(q, fk_fn, targets, damping) -> q_new` (one DLS iteration; `fk_fn(q)->[K,3]` fingertip positions; `targets` [K,3]); `solve_ik(q0, fk_fn, targets, damping, iters) -> q` (loop); `retarget_sequence(target_seq, fk_fn, q0, n_q, damping, iters) -> q_seq[T,n_q]`. FK is injected so this is testable with a toy chain (no mujoco).

- [ ] **Step 1: Write the failing test**

```python
# egoaero/tests/policy/test_retarget.py
import numpy as np
from egoaero.policy import retarget as RT

def toy_fk(q):
    # 1 "fingertip": a 2-joint planar arm in xy, link lengths 1,1
    x = np.cos(q[0]) + np.cos(q[0] + q[1])
    y = np.sin(q[0]) + np.sin(q[0] + q[1])
    return np.array([[x, y, 0.0]])

def test_solve_ik_reaches_target():
    target = np.array([[1.0, 1.0, 0.0]])      # reachable
    q = RT.solve_ik(np.array([0.1, 0.1]), toy_fk, target, damping=0.05, iters=200)
    err = np.linalg.norm(toy_fk(q) - target)
    assert err < 1e-2

def test_retarget_sequence_shape():
    seq = np.array([[[1.0, 1.0, 0.0]], [[0.5, 1.2, 0.0]]])   # T=2, K=1
    qs = RT.retarget_sequence(seq, toy_fk, np.array([0.1, 0.1]), n_q=2,
                              damping=0.05, iters=100)
    assert qs.shape == (2, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest egoaero/tests/policy/test_retarget.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write minimal implementation**

```python
# egoaero/egoaero/policy/retarget.py
"""Kinematic retargeting via damped-least-squares IK. FK is injected so the solver
is independent of MuJoCo (unit-tested with a toy chain). In the sim, fk_fn wraps the
Shadow-Hand forward kinematics (see hand_model.fk_fingertips)."""
from __future__ import annotations
import numpy as np


def _jacobian(fk_fn, q, eps=1e-5):
    base = fk_fn(q).reshape(-1)        # [3K]
    J = np.zeros((base.size, q.size))
    for j in range(q.size):
        dq = q.copy(); dq[j] += eps
        J[:, j] = (fk_fn(dq).reshape(-1) - base) / eps
    return J, base


def ik_step(q, fk_fn, targets, damping):
    J, cur = _jacobian(fk_fn, q)
    err = np.asarray(targets).reshape(-1) - cur
    JT = J.T
    dq = JT @ np.linalg.solve(J @ JT + (damping ** 2) * np.eye(J.shape[0]), err)
    return q + dq


def solve_ik(q0, fk_fn, targets, damping, iters):
    q = np.asarray(q0, float).copy()
    for _ in range(int(iters)):
        q = ik_step(q, fk_fn, targets, damping)
    return q


def retarget_sequence(target_seq, fk_fn, q0, n_q, damping, iters):
    q = np.asarray(q0, float).copy()
    out = np.zeros((len(target_seq), n_q))
    for t in range(len(target_seq)):
        q = solve_ik(q, fk_fn, target_seq[t], damping, iters)   # warm-start from prev frame
        out[t] = q
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest egoaero/tests/policy/test_retarget.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add egoaero/egoaero/policy/retarget.py egoaero/tests/policy/test_retarget.py
git commit -m "egoaero: policy DLS-IK retargeting (FK-injected, toy-chain tested)"
```

---

### Task 6: `hand_model.py` — Shadow Hand loader + FK (gated)

**Files:**
- Create: `egoaero/egoaero/policy/hand_model.py`
- Test: `egoaero/tests/policy/test_hand_model.py`

**Interfaces:**
- Produces: `class HandModel` with `.model`, `.data` (mujoco), `.fingertip_site_ids: dict[str,int]` keyed by MANO finger name, `.n_act:int`, `.fk_fingertips(q) -> np.ndarray[5,3]` (set actuated joint qpos = q, `mj_forward`, return the 5 fingertip site world positions in MANO finger order), `.actuated_joint_qpos_adr -> np.ndarray`. Shadow Hand fingertip site name tokens: thumb→`thtip`, index→`fftip`, middle→`mftip`, ring→`rftip`, little→`lftip`.

- [ ] **Step 1: Write the failing test**

```python
# egoaero/tests/policy/test_hand_model.py
import os, numpy as np, pytest

def _xml():
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(here, "assets", "shadow_hand", "right_hand.xml")

def test_hand_model_fk_and_sites():
    pytest.importorskip("mujoco")
    if not os.path.exists(_xml()):
        pytest.skip("shadow hand not vendored")
    from egoaero.policy.hand_model import HandModel
    hm = HandModel(_xml())
    assert set(hm.fingertip_site_ids) == {"thumb", "index", "middle", "ring", "little"}
    tips = hm.fk_fingertips(np.zeros(hm.n_act))
    assert tips.shape == (5, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest egoaero/tests/policy/test_hand_model.py -v`
Expected: FAIL (`ModuleNotFoundError: egoaero.policy.hand_model`) or SKIP if mujoco absent (install via setup_rl.sh first so it FAILs then PASSes).

- [ ] **Step 3: Write minimal implementation**

```python
# egoaero/egoaero/policy/hand_model.py
"""Shadow Hand MuJoCo wrapper: forward kinematics + fingertip sites, mapped to the
MANO finger order. Documented substitute for the Inspire Hand (App-D/G)."""
from __future__ import annotations
import numpy as np

_TIP_TOKENS = {"thumb": "thtip", "index": "fftip", "middle": "mftip",
               "ring": "rftip", "little": "lftip"}


class HandModel:
    def __init__(self, xml_path):
        import mujoco
        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.n_act = int(self.model.nu)
        # actuated joint qpos addresses (one per actuator's transmission joint)
        adr = []
        for i in range(self.model.nu):
            j = self.model.actuator_trnid[i, 0]
            adr.append(int(self.model.jnt_qposadr[j]))
        self.actuated_joint_qpos_adr = np.array(adr, int)
        # map fingertip sites by name token
        self.fingertip_site_ids = {}
        for finger, tok in _TIP_TOKENS.items():
            for sid in range(self.model.nsite):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, sid) or ""
                if tok in name:
                    self.fingertip_site_ids[finger] = sid
                    break
        missing = set(_TIP_TOKENS) - set(self.fingertip_site_ids)
        if missing:
            raise RuntimeError(f"fingertip sites not found for {missing}; "
                               f"inspect site names in {xml_path}")

    def fk_fingertips(self, q):
        q = np.asarray(q, float)
        self.data.qpos[self.actuated_joint_qpos_adr] = q[: self.n_act]
        self._mj.mj_forward(self.model, self.data)
        order = ["thumb", "index", "middle", "ring", "little"]
        return np.stack([self.data.site_xpos[self.fingertip_site_ids[f]].copy()
                         for f in order], 0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest egoaero/tests/policy/test_hand_model.py -v`
Expected: PASS. If the fingertip tokens don't match the vendored MJCF, inspect site names (`python -c "import mujoco,glob; m=mujoco.MjModel.from_xml_path('egoaero/assets/shadow_hand/right_hand.xml'); print([mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_SITE,i) for i in range(m.nsite)])"`) and update `_TIP_TOKENS` to the actual tip-site substrings; record the mapping in ASSUMPTIONS.md.

- [ ] **Step 5: Commit**

```bash
git add egoaero/egoaero/policy/hand_model.py egoaero/tests/policy/test_hand_model.py
git commit -m "egoaero: Shadow Hand MuJoCo wrapper (FK + fingertip sites)"
```

---

### Task 7: `task.py` — build MuJoCo scene + reference loader (gated)

**Files:**
- Create: `egoaero/egoaero/policy/task.py`
- Test: `egoaero/tests/policy/test_task.py`

**Interfaces:**
- Produces: `load_reference(run_dir) -> dict` with `wrist_pos[T,3]` (= joints[:,0]), `fingertips_h[T,5,3]` (= joints[:, [4,8,12,16,20]]), `obj_pos[T,3]`, `obj_R[T,3,3]` (from obj_poses_t), `contact_active[T] -> list[int]` (reference-active finger indices: fingertip within `contact_active_dist_m` of the object mesh surface at the reference object pose), `mesh_path`, `T`. `build_task(run_dir, hand_xml, cfg) -> Task` where `Task` holds a composed `mujoco.MjModel` (Shadow Hand mounted on a free/actuated wrist base + the object as a free body with the reconstructed mesh + a table plane), `HandModel`, the reference dict, and `cfg`.

- [ ] **Step 1: Write the failing test**

```python
# egoaero/tests/policy/test_task.py
import os, numpy as np, pytest
from egoaero import config
from egoaero.pipeline import run_pipeline
from egoaero import contract

def _recon(tmp_path):
    cfg = config.load_config(overrides={"num_frames": 12})
    ctx = run_pipeline(cfg, str(tmp_path / "run"), stages="all")
    contract.write(ctx)
    return str(tmp_path / "run")

def test_load_reference(tmp_path):
    run = _recon(tmp_path)
    from egoaero.policy.task import load_reference
    ref = load_reference(run)
    assert ref["wrist_pos"].shape == (12, 3)
    assert ref["fingertips_h"].shape == (12, 5, 3)
    assert ref["obj_R"].shape == (12, 3, 3)
    assert isinstance(ref["contact_active"][0], list)

def test_build_task(tmp_path):
    pytest.importorskip("mujoco")
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hand_xml = os.path.join(os.path.dirname(here), "assets", "shadow_hand", "right_hand.xml")
    if not os.path.exists(hand_xml):
        pytest.skip("shadow hand not vendored")
    run = _recon(tmp_path)
    from egoaero.policy.task import build_task
    from egoaero.config import load_config
    task = build_task(run, hand_xml, load_config().get("quality", {}))
    assert task.model.nu > 0 and task.ref["T"] == 12
```

(`load_reference` is pure-numpy and testable without mujoco; `build_task` is gated.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest egoaero/tests/policy/test_task.py -v`
Expected: FAIL (`ModuleNotFoundError: egoaero.policy.task`).

- [ ] **Step 3: Write minimal implementation**

```python
# egoaero/egoaero/policy/task.py
"""Build the MuJoCo manipulation task from an SP1 reconstruction run: Shadow Hand on a
6-DoF wrist base + the reconstructed object (free body, its mesh) on a table, plus the
reference trajectories. The object reference drives r_obj; the hand reference drives r^I."""
from __future__ import annotations
import json, os
import numpy as np
from .hand_model import HandModel

_TIP_JOINTS = [4, 8, 12, 16, 20]    # MANO joint indices of the 5 fingertips (core/hand layout)


def _load_npz(path):
    import numpy as _np
    with _np.load(path) as z:
        return {k: z[k] for k in z.files}


def load_reference(run_dir):
    d = os.path.join(run_dir, "contract")
    man = json.load(open(os.path.join(d, "manifest.json")))
    hand = _load_npz(os.path.join(d, man["hand_mano"]))
    obj = _load_npz(os.path.join(d, man["object_traj"]))
    joints = hand["joints"]                       # [T,21,3]
    poses = obj["obj_poses_t"]                    # [T,4,4]
    T = int(joints.shape[0])
    wrist_pos = joints[:, 0, :]
    fingertips_h = joints[:, _TIP_JOINTS, :]      # [T,5,3]
    obj_pos = poses[:, :3, 3]
    obj_R = poses[:, :3, :3]
    # reference-active fingers: fingertip within contact_active_dist of object centroid+radius proxy
    # (mesh-distance proxy; documented). Use distance to object centre minus mesh radius.
    verts = _read_obj_verts(os.path.join(d, man["object_mesh"]))
    radius = float(np.max(np.linalg.norm(verts - verts.mean(0), axis=1)))
    thresh = 0.02
    contact_active = []
    for t in range(T):
        c = obj_pos[t]
        act = [i for i in range(5)
               if abs(np.linalg.norm(fingertips_h[t, i] - c) - radius) < thresh]
        contact_active.append(act)
    return {"wrist_pos": wrist_pos, "fingertips_h": fingertips_h, "obj_pos": obj_pos,
            "obj_R": obj_R, "contact_active": contact_active,
            "mesh_path": os.path.join(d, man["object_mesh"]), "T": T}


def _read_obj_verts(path):
    vs = []
    for line in open(path):
        if line.startswith("v "):
            vs.append([float(x) for x in line.split()[1:4]])
    return np.asarray(vs, float)


class Task:
    def __init__(self, model, hand, ref, cfg):
        self.model = model; self.hand = hand; self.ref = ref; self.cfg = cfg


def build_task(run_dir, hand_xml, cfg):
    import mujoco
    ref = load_reference(run_dir)
    # Compose a scene: include the hand, add a free-body object using the reconstructed mesh,
    # and a table plane. The hand is mounted on a free joint so the wrist can be driven.
    scene_xml = f"""
<mujoco model="egoaero_task">
  <include file="{os.path.abspath(hand_xml)}"/>
  <asset><mesh name="obj" file="{os.path.abspath(ref['mesh_path'])}"/></asset>
  <worldbody>
    <geom name="table" type="plane" size="1 1 0.05" pos="0 0 0"/>
    <body name="object" pos="{ref['obj_pos'][0,0]} {ref['obj_pos'][0,1]} {ref['obj_pos'][0,2]}">
      <freejoint name="obj_free"/>
      <geom name="obj_geom" type="mesh" mesh="obj" density="200"/>
    </body>
  </worldbody>
</mujoco>"""
    model = mujoco.MjModel.from_xml_string(scene_xml)
    hand = HandModel(hand_xml)
    return Task(model, hand, ref, cfg)
```

Note (record in ASSUMPTIONS.md): the contact-active heuristic uses a centroid+radius sphere proxy because the contract does not carry per-finger vertex groups. The wrist-base mounting (free joint vs an actuated 6-DoF base) may need adjustment to the specific Shadow Hand MJCF — verify by building and stepping; if `include` clashes (duplicate worldbody/option), instead load the hand model separately and merge via `mujoco.MjSpec` (mujoco>=3.2) or strip the hand's own `<worldbody>` wrapper. Document the final approach.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest egoaero/tests/policy/test_task.py -v`
Expected: `test_load_reference` PASS (pure); `test_build_task` PASS (gated) or SKIP. Verify PASS on this box after setup. If `include` of the hand MJCF errors, switch to `mujoco.MjSpec`-based composition and update the code + ASSUMPTIONS.

- [ ] **Step 5: Commit**

```bash
git add egoaero/egoaero/policy/task.py egoaero/tests/policy/test_task.py egoaero/ASSUMPTIONS.md
git commit -m "egoaero: policy MuJoCo task builder + reference loader"
```

---

### Task 8: `env.py` — StageIEnv (gated)

**Files:**
- Create: `egoaero/egoaero/policy/env.py`
- Test: `egoaero/tests/policy/test_env_stage1.py`

**Interfaces:**
- Produces: `class StageIEnv(gymnasium.Env)` built from a `Task`. Observation = concat(actuator qpos, actuator qvel, current wrist-pos reference, current fingertips reference flattened). Action = Box in [-1,1]^n_act mapped to actuator ctrl ranges. Reward = `r_stage1` from the hand reference (wrist via the hand root body pos; fingertips via `HandModel` sites). Episode length = `ref["T"]`. Exposes `.n_act`.

- [ ] **Step 1: Write the failing test**

```python
# egoaero/tests/policy/test_env_stage1.py
import os, numpy as np, pytest

def _setup(tmp_path):
    pytest.importorskip("mujoco"); pytest.importorskip("gymnasium")
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hand_xml = os.path.join(os.path.dirname(here), "assets", "shadow_hand", "right_hand.xml")
    if not os.path.exists(hand_xml):
        pytest.skip("shadow hand not vendored")
    from egoaero import config; from egoaero.pipeline import run_pipeline; from egoaero import contract
    ctx = run_pipeline(config.load_config(overrides={"num_frames": 10}), str(tmp_path/"run"), "all")
    contract.write(ctx)
    from egoaero.policy.task import build_task
    return build_task(str(tmp_path/"run"), hand_xml, config.load_config())

def test_stage1_env_api(tmp_path):
    from egoaero.policy.env import StageIEnv
    env = StageIEnv(_setup(tmp_path))
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    a = env.action_space.sample()
    obs2, rew, term, trunc, info = env.step(a)
    assert np.isfinite(rew) and obs2.shape == obs.shape

def test_stage1_check_env(tmp_path):
    sb3 = pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.env_checker import check_env
    from egoaero.policy.env import StageIEnv
    check_env(StageIEnv(_setup(tmp_path)), warn=True, skip_render_check=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest egoaero/tests/policy/test_env_stage1.py -v`
Expected: FAIL (`ModuleNotFoundError: egoaero.policy.env`).

- [ ] **Step 3: Write minimal implementation**

```python
# egoaero/egoaero/policy/env.py
"""Gymnasium environments for the two-stage residual policy.
StageIEnv: track the reconstructed hand reference (reward r^I).
StageIIEnv: residual on a frozen Stage-I policy with object + contact rewards (reward r^R)."""
from __future__ import annotations
import numpy as np
from . import rewards as RW


def _make_base(env_cls):
    import gymnasium as gym
    return type(env_cls.__name__, (gym.Env,), dict(env_cls.__dict__))


class _MjEnvMixin:
    def _ctrl_from_action(self, a):
        lo, hi = self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1]
        return lo + 0.5 * (np.asarray(a) + 1.0) * (hi - lo)

    def _act_qpos(self):
        return self.data.qpos[self.task.hand.actuated_joint_qpos_adr].copy()

    def _act_qvel(self):
        import numpy as _np
        adr = self.task.hand.actuated_joint_qpos_adr
        return self.data.qvel[: len(adr)].copy() if self.data.qvel.size >= len(adr) else _np.zeros(len(adr))

    def _fingertips(self):
        import mujoco
        mujoco.mj_forward(self.model, self.data)
        order = ["thumb", "index", "middle", "ring", "little"]
        ids = self.task.hand.fingertip_site_ids
        return np.stack([self.data.site_xpos[ids[f]].copy() for f in order], 0)


def StageIEnv(task):
    import gymnasium as gym, mujoco
    rcfg = task.cfg["reward"]

    class _Env(_MjEnvMixin, gym.Env):
        def __init__(self):
            self.task = task; self.model = task.model
            self.data = mujoco.MjData(self.model)
            self.n_act = int(self.model.nu)
            self.T = task.ref["T"]; self.t = 0
            self._prev_a = np.zeros(self.n_act)
            obs_dim = self.n_act * 2 + 3 + 5 * 3
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
            self.action_space = gym.spaces.Box(-1.0, 1.0, (self.n_act,), np.float32)

        def _obs(self):
            ref = self.task.ref
            return np.concatenate([self._act_qpos(), self._act_qvel(),
                                   ref["wrist_pos"][self.t],
                                   ref["fingertips_h"][self.t].reshape(-1)]).astype(np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            mujoco.mj_resetData(self.model, self.data)
            self.t = 0; self._prev_a = np.zeros(self.n_act)
            return self._obs(), {}

        def step(self, a):
            self.data.ctrl[:] = self._ctrl_from_action(a)
            mujoco.mj_step(self.model, self.data)
            ref = self.task.ref
            tips = self._fingertips()
            rw = RW.r_wrist(self.data.qpos[:3], np.eye(3), self.data.qvel[:3],
                            ref["wrist_pos"][self.t], np.eye(3), np.zeros(3),
                            rcfg["lam_p"], rcfg["lam_R"], rcfg["lam_v"])
            rf = RW.r_finger(tips, ref["fingertips_h"][self.t], rcfg["lam_k"])
            rs = RW.r_smooth(a, self._prev_a, np.zeros(self.n_act), self._act_qvel(),
                             rcfg["lam_a"], rcfg["lam_tau"])
            reward = RW.r_stage1(rw, rf, rs, rcfg["w_w"], rcfg["w_f"], rcfg["w_s"])
            self._prev_a = np.asarray(a, float)
            self.t += 1
            term = False; trunc = self.t >= self.T
            obs = self._obs() if not trunc else self._last_obs()
            return obs, float(reward), term, trunc, {}

        def _last_obs(self):
            self.t = min(self.t, self.T - 1)
            return self._obs()

    return _Env()
```

Note: the wrist-pose reward uses identity orientation (the contract carries no wrist orientation — documented default); `data.qpos[:3]`/`qvel[:3]` assume the object free-joint is NOT first — if the composed model orders the object free joint before the hand, adjust the wrist slice to the hand root body's `xpos` via `data.body(...).xpos`. Verify and document.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest egoaero/tests/policy/test_env_stage1.py -v`
Expected: PASS (gated). Fix any observation/index mismatch flagged by `check_env` before moving on.

- [ ] **Step 5: Commit**

```bash
git add egoaero/egoaero/policy/env.py egoaero/tests/policy/test_env_stage1.py egoaero/ASSUMPTIONS.md
git commit -m "egoaero: StageIEnv (hand-tracking, reward r^I)"
```

---

### Task 9: `env.py` — StageIIEnv (residual, object+contact; gated)

**Files:**
- Modify: `egoaero/egoaero/policy/env.py` (append `StageIIEnv`)
- Test: `egoaero/tests/policy/test_env_stage2.py`

**Interfaces:**
- Produces: `StageIIEnv(task, pi_I)` — wraps a frozen Stage-I policy `pi_I` (object with `.predict(obs)->(action,_)`, or any callable `obs->action`). Action = residual `Δa ∈ [-1,1]^n_act`; applied ctrl = `clip(a_I + Δa)`. Observation = StageI obs + object pose(7) + object vel(6) + hand-object distance(5) + contact forces(5). Reward = `r_stage2`. Early termination when object-pos error / hand error / penetration exceed `cfg["term"]` thresholds.

- [ ] **Step 1: Write the failing test**

```python
# egoaero/tests/policy/test_env_stage2.py
import os, numpy as np, pytest
from egoaero.tests.policy.test_env_stage1 import _setup  # reuse builder

def test_stage2_residual_and_api(tmp_path):
    sb3 = pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.env_checker import check_env
    from egoaero.policy.env import StageIIEnv
    task = _setup(tmp_path)
    pi_I = lambda obs: np.zeros(task.model.nu, np.float32)   # frozen no-op stage-I
    env = StageIIEnv(task, pi_I)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    o2, r, term, trunc, _ = env.step(env.action_space.sample())
    assert np.isfinite(r)
    check_env(env, warn=True, skip_render_check=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest egoaero/tests/policy/test_env_stage2.py -v`
Expected: FAIL (`AttributeError: StageIIEnv`).

- [ ] **Step 3: Write minimal implementation** (append to `env.py`)

```python
def StageIIEnv(task, pi_I):
    import gymnasium as gym, mujoco
    rcfg = task.cfg["reward"]; tcfg = task.cfg["term"]; rt = task.cfg["retarget"]

    def _as_action(pi, obs):
        if hasattr(pi, "predict"):
            return np.asarray(pi.predict(obs, deterministic=True)[0], float)
        return np.asarray(pi(obs), float)

    class _Env(_MjEnvMixin, gym.Env):
        def __init__(self):
            self.task = task; self.model = task.model
            self.data = mujoco.MjData(self.model)
            self.n_act = int(self.model.nu); self.T = task.ref["T"]; self.t = 0
            self._prev_a = np.zeros(self.n_act)
            self._obj_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obj_free")
            self._obj_qadr = int(self.model.jnt_qposadr[self._obj_jid])
            self._obj_dadr = int(self.model.jnt_dofadr[self._obj_jid])
            self._stage1_dim = self.n_act * 2 + 3 + 5 * 3
            obs_dim = self._stage1_dim + 7 + 6 + 5 + 5
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
            self.action_space = gym.spaces.Box(-1.0, 1.0, (self.n_act,), np.float32)

        def _obj_pose(self):
            q = self.data.qpos[self._obj_qadr: self._obj_qadr + 7]
            pos = q[:3].copy(); quat = q[3:7].copy()
            Rm = np.zeros(9); mujoco.mju_quat2Mat(Rm, quat)
            return pos, Rm.reshape(3, 3)

        def _obj_vel(self):
            return self.data.qvel[self._obj_dadr: self._obj_dadr + 6].copy()

        def _contact_forces(self):
            # per-fingertip contact force magnitude proxy via body cfrc_ext on tip bodies
            f = np.zeros(5)
            order = ["thumb", "index", "middle", "ring", "little"]
            for i, fg in enumerate(order):
                sid = self.task.hand.fingertip_site_ids[fg]
                bid = int(self.model.site_bodyid[sid])
                f[i] = float(np.linalg.norm(self.data.cfrc_ext[bid][:3]))
            return f

        def _stage1_obs(self):
            ref = self.task.ref
            return np.concatenate([self._act_qpos(), self._act_qvel(),
                                   ref["wrist_pos"][self.t],
                                   ref["fingertips_h"][self.t].reshape(-1)]).astype(np.float32)

        def _obs(self):
            pos, Rm = self._obj_pose()
            quat = np.zeros(4); mujoco.mju_mat2Quat(quat, Rm.reshape(-1))
            tips = self._fingertips()
            hod = np.linalg.norm(tips - pos[None, :], axis=1)
            return np.concatenate([self._stage1_obs(), pos, quat, self._obj_vel(),
                                   hod, self._contact_forces()]).astype(np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            mujoco.mj_resetData(self.model, self.data)
            self.t = 0; self._prev_a = np.zeros(self.n_act)
            ref = self.task.ref
            self.data.qpos[self._obj_qadr: self._obj_qadr + 3] = ref["obj_pos"][0]
            q = np.zeros(4); mujoco.mju_mat2Quat(q, ref["obj_R"][0].reshape(-1))
            self.data.qpos[self._obj_qadr + 3: self._obj_qadr + 7] = q
            mujoco.mj_forward(self.model, self.data)
            return self._obs(), {}

        def step(self, dadelta):
            base = _as_action(pi_I, self._stage1_obs())
            a = np.clip(base + np.asarray(dadelta, float), -1.0, 1.0)
            self.data.ctrl[:] = self._ctrl_from_action(a)
            mujoco.mj_step(self.model, self.data)
            ref = self.task.ref; pos, Rm = self._obj_pose(); tips = self._fingertips()
            r1 = RW.r_stage1(
                RW.r_wrist(self.data.qpos[:3], np.eye(3), self.data.qvel[:3],
                           ref["wrist_pos"][self.t], np.eye(3), np.zeros(3),
                           rcfg["lam_p"], rcfg["lam_R"], rcfg["lam_v"]),
                RW.r_finger(tips, ref["fingertips_h"][self.t], rcfg["lam_k"]),
                RW.r_smooth(a, self._prev_a, np.zeros(self.n_act), self._act_qvel(),
                            rcfg["lam_a"], rcfg["lam_tau"]),
                rcfg["w_w"], rcfg["w_f"], rcfg["w_s"])
            ro = RW.r_obj(pos, Rm, self._obj_vel()[:3], ref["obj_pos"][self.t], ref["obj_R"][self.t],
                          np.zeros(3), rcfg["mu_p"], rcfg["mu_R"], rcfg["mu_v"])
            hod = np.linalg.norm(tips - pos[None, :], axis=1)
            rc = RW.r_contact(hod, self._contact_forces(), ref["contact_active"][self.t],
                              rcfg["mu_d"], rcfg["mu_F"])
            rr = RW.r_res(dadelta, rcfg["mu_delta"])
            reward = RW.r_stage2(r1, ro, rc, rr, rcfg["eta_I"], rcfg["eta_o"],
                                 rcfg["eta_c"], rcfg["eta_delta"])
            self._prev_a = a; self.t += 1
            obj_err = float(np.linalg.norm(pos - ref["obj_pos"][self.t - 1]))
            term = obj_err > tcfg["obj_pos_err_m"]
            trunc = self.t >= self.T
            obs = self._obs() if not (term or trunc) else self._safe_obs()
            return obs, float(reward), bool(term), bool(trunc), {}

        def _safe_obs(self):
            self.t = min(self.t, self.T - 1); return self._obs()

    return _Env()
```

Note (ASSUMPTIONS): contact force uses `cfrc_ext` on the fingertip body as a magnitude proxy; early termination implements only the object-pos threshold robustly (hand-error/penetration thresholds are documented but optional). Verify object free-joint addressing (`obj_free`) resolves; if the composed model renamed it, adjust.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest egoaero/tests/policy/test_env_stage2.py -v`
Expected: PASS (gated). Resolve any `check_env` complaint.

- [ ] **Step 5: Commit**

```bash
git add egoaero/egoaero/policy/env.py egoaero/tests/policy/test_env_stage2.py egoaero/ASSUMPTIONS.md
git commit -m "egoaero: StageIIEnv (residual policy, object+contact, reward r^R)"
```

---

### Task 10: `train.py` — two-stage SB3 PPO driver (gated)

**Files:**
- Create: `egoaero/egoaero/policy/train.py`
- Test: `egoaero/tests/policy/test_train.py`

**Interfaces:**
- Produces: `train_two_stage(task, cfg, budget="smoke", out_dir=None, seed=0) -> (pi_I, pi_R)` — PPO on `StageIEnv` for `cfg["budget"][budget]` steps, freeze, PPO on `StageIIEnv(task, pi_I)` for the same budget; if `out_dir`, save `pi_I.zip`/`pi_R.zip`.

- [ ] **Step 1: Write the failing test**

```python
# egoaero/tests/policy/test_train.py
import os, pytest
from egoaero.tests.policy.test_env_stage1 import _setup

def test_two_stage_smoke_trains_and_saves(tmp_path):
    pytest.importorskip("stable_baselines3"); pytest.importorskip("torch")
    from egoaero.policy.train import train_two_stage
    from egoaero.config import load_config
    task = _setup(tmp_path)
    cfg = dict(load_config()); cfg.setdefault("budget", {"smoke": 256, "real": 1500000})
    out = str(tmp_path / "pol")
    pi_I, pi_R = train_two_stage(task, _policy_cfg(), budget="smoke", out_dir=out, seed=0)
    assert os.path.exists(os.path.join(out, "pi_I.zip"))
    assert os.path.exists(os.path.join(out, "pi_R.zip"))

def _policy_cfg():
    import os, yaml
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(os.path.join(here, "egoaero", "configs", "policy.yaml")) as f:
        return yaml.safe_load(f)
```

(The `task.cfg` must include `reward`/`term`/`retarget`/`budget` from `policy.yaml`; `build_task` should be called with the loaded `policy.yaml` dict. Adjust `_setup` if needed so `task.cfg` is the policy config, not the main config — see Task 7 note; ensure `build_task(run, hand_xml, policy_cfg)`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest egoaero/tests/policy/test_train.py -v`
Expected: FAIL (`ModuleNotFoundError: egoaero.policy.train`).

- [ ] **Step 3: Write minimal implementation**

```python
# egoaero/egoaero/policy/train.py
"""Two-stage residual policy training with stable-baselines3 PPO."""
from __future__ import annotations
import os


def train_two_stage(task, policy_cfg, budget="smoke", out_dir=None, seed=0):
    from stable_baselines3 import PPO
    from .env import StageIEnv, StageIIEnv
    task.cfg = policy_cfg
    steps = int(policy_cfg["budget"][budget])
    ppo = policy_cfg["ppo"]
    kw = dict(n_steps=min(ppo["n_steps"], max(64, steps)), batch_size=ppo["batch_size"],
              learning_rate=ppo["learning_rate"], seed=seed,
              policy_kwargs={"net_arch": ppo["net_arch"]}, verbose=0)

    env1 = StageIEnv(task)
    pi_I = PPO(ppo["policy"], env1, **kw)
    pi_I.learn(total_timesteps=steps)

    frozen = lambda obs: pi_I.predict(obs, deterministic=True)[0]
    env2 = StageIIEnv(task, frozen)
    pi_R = PPO(ppo["policy"], env2, **kw)
    pi_R.learn(total_timesteps=steps)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        pi_I.save(os.path.join(out_dir, "pi_I"))
        pi_R.save(os.path.join(out_dir, "pi_R"))
    return pi_I, pi_R
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest egoaero/tests/policy/test_train.py -v`
Expected: PASS (gated; smoke budget finishes in well under a minute on CPU/GPU).

- [ ] **Step 5: Commit**

```bash
git add egoaero/egoaero/policy/train.py egoaero/tests/policy/test_train.py
git commit -m "egoaero: two-stage SB3 PPO training driver"
```

---

### Task 11: `evaluate.py` — rollout, App-H metrics, ablation (gated)

**Files:**
- Create: `egoaero/egoaero/policy/evaluate.py`
- Test: `egoaero/tests/policy/test_evaluate.py`

**Interfaces:**
- Produces: `rollout(task, pi_I, pi_R, seed=0) -> dict` with `obj_pos[T,3]`, `obj_R[T,3,3]`, `fingertips[T,5,3]`; `evaluate(task, pi_I, pi_R, seeds=(0,1)) -> dict` with `Er,Et,Ej,Eft,SR`; `ablation(run_dir, hand_xml, policy_cfg, budget="smoke") -> dict` with keys `only_hand`, `wo_contact_opt`, `full`, each a metrics dict.

- [ ] **Step 1: Write the failing test**

```python
# egoaero/tests/policy/test_evaluate.py
import pytest
from egoaero.tests.policy.test_env_stage1 import _setup
from egoaero.tests.policy.test_train import _policy_cfg

def test_evaluate_returns_metrics(tmp_path):
    pytest.importorskip("stable_baselines3"); pytest.importorskip("torch")
    from egoaero.policy.train import train_two_stage
    from egoaero.policy.evaluate import evaluate
    task = _setup(tmp_path)
    pi_I, pi_R = train_two_stage(task, _policy_cfg(), budget="smoke", seed=0)
    m = evaluate(task, pi_I, pi_R, seeds=(0,))
    assert set(["Er", "Et", "Ej", "Eft", "SR"]).issubset(m)
    assert 0.0 <= m["SR"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest egoaero/tests/policy/test_evaluate.py -v`
Expected: FAIL (`ModuleNotFoundError: egoaero.policy.evaluate`).

- [ ] **Step 3: Write minimal implementation**

```python
# egoaero/egoaero/policy/evaluate.py
"""Rollout a trained two-stage policy and compute App-H metrics; ablation harness."""
from __future__ import annotations
import numpy as np
from . import metrics as M


def rollout(task, pi_I, pi_R, seed=0):
    from .env import StageIIEnv
    frozen = (lambda obs: pi_I.predict(obs, deterministic=True)[0]) if pi_I is not None \
        else (lambda obs: np.zeros(task.model.nu, np.float32))
    env = StageIIEnv(task, frozen)
    obs, _ = env.reset(seed=seed)
    op, oR, ft = [], [], []
    done = False
    while not done:
        a = pi_R.predict(obs, deterministic=True)[0] if pi_R is not None \
            else np.zeros(env.action_space.shape, np.float32)
        obs, _, term, trunc, _ = env.step(a)
        pos, Rm = env._obj_pose(); op.append(pos); oR.append(Rm); ft.append(env._fingertips())
        done = term or trunc
    return {"obj_pos": np.array(op), "obj_R": np.array(oR), "fingertips": np.array(ft)}


def evaluate(task, pi_I, pi_R, seeds=(0, 1)):
    ref = task.ref; T = ref["T"]; rows = []
    for s in seeds:
        rl = rollout(task, pi_I, pi_R, seed=s)
        n = min(T, len(rl["obj_pos"]))
        Er = M.object_rotation_error(rl["obj_R"][:n], ref["obj_R"][:n])
        Et = M.object_translation_error(rl["obj_pos"][:n], ref["obj_pos"][:n])
        Ej = M.mean_fingertip_error(rl["fingertips"][:n], ref["fingertips_h"][:n])
        Eft = Ej
        rows.append((Er, Et, Ej, Eft))
    arr = np.array(rows)
    return {"Er": float(arr[:, 0].mean()), "Et": float(arr[:, 1].mean()),
            "Ej": float(arr[:, 2].mean()), "Eft": float(arr[:, 3].mean()),
            "SR": M.success_rate([tuple(r) for r in rows])}


def ablation(run_dir, hand_xml, policy_cfg, budget="smoke"):
    from .task import build_task
    from .train import train_two_stage
    out = {}
    # full: contact-optimized reconstruction (the contract as-is)
    task = build_task(run_dir, hand_xml, policy_cfg)
    pi_I, pi_R = train_two_stage(task, policy_cfg, budget=budget)
    out["full"] = evaluate(task, pi_I, pi_R, seeds=(0,))
    # only_hand: Stage-I policy alone (no residual)
    out["only_hand"] = evaluate(task, pi_I, None, seeds=(0,))
    # wo_contact_opt: same pipeline but it is the caller's responsibility to pass a run_dir
    # whose reconstruction skipped stage6 contact optimization; here we reuse `full` task as a
    # documented placeholder when no such run is provided.
    out["wo_contact_opt"] = out["full"]
    return out
```

Note (ASSUMPTIONS): `Ej` and `Eft` both use fingertip error here because the contract exposes fingertip keypoints (full per-joint robot↔human correspondence is not defined for the substitute hand). The `wo_contact_opt` ablation requires a second reconstruction run produced with stage6 disabled (`--stages 0-5,7`); the CLI (Task 12) wires that — in-lib it falls back to `full` with a logged caveat.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest egoaero/tests/policy/test_evaluate.py -v`
Expected: PASS (gated).

- [ ] **Step 5: Commit**

```bash
git add egoaero/egoaero/policy/evaluate.py egoaero/tests/policy/test_evaluate.py egoaero/ASSUMPTIONS.md
git commit -m "egoaero: policy rollout + App-H evaluate + ablation harness"
```

---

### Task 12: CLIs, README, ASSUMPTIONS, and one real GPU training run

**Files:**
- Create: `egoaero/egoaero/policy/cli.py`
- Modify: `egoaero/pyproject.toml` (console scripts), `egoaero/README.md`, `egoaero/ASSUMPTIONS.md`
- Test: `egoaero/tests/policy/test_cli_smoke.py`

**Interfaces:**
- Produces: `egoaero-train --run <recon_dir> --out <policy_dir> [--budget smoke|real]` and `egoaero-eval --run <recon_dir> --policy <policy_dir>` console entry points calling `train_two_stage`/`evaluate` with the loaded `policy.yaml`.

- [ ] **Step 1: Write the failing test**

```python
# egoaero/tests/policy/test_cli_smoke.py
import os, subprocess, sys, pytest

def test_train_eval_cli(tmp_path):
    pytest.importorskip("stable_baselines3"); pytest.importorskip("mujoco")
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if not os.path.exists(os.path.join(here, "assets", "shadow_hand", "right_hand.xml")):
        pytest.skip("shadow hand not vendored")
    from egoaero import config; from egoaero.pipeline import run_pipeline; from egoaero import contract
    ctx = run_pipeline(config.load_config(overrides={"num_frames": 8}), str(tmp_path/"run"), "all")
    contract.write(ctx)
    env = dict(os.environ); pol = str(tmp_path / "pol")
    r = subprocess.run([sys.executable, "-m", "egoaero.policy.cli", "train",
                        "--run", str(tmp_path/"run"), "--out", pol, "--budget", "smoke"],
                       cwd=here, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert os.path.exists(os.path.join(pol, "pi_R.zip"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest egoaero/tests/policy/test_cli_smoke.py -v`
Expected: FAIL (`No module named egoaero.policy.cli`).

- [ ] **Step 3: Write minimal implementation**

```python
# egoaero/egoaero/policy/cli.py
import argparse, os, json, yaml


def _policy_cfg():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "configs", "policy.yaml")) as f:
        return yaml.safe_load(f)


def _hand_xml():
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(here, "assets", "shadow_hand", "right_hand.xml")


def main():
    ap = argparse.ArgumentParser("egoaero-policy")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--run", required=True); t.add_argument("--out", required=True)
    t.add_argument("--budget", default="smoke")
    e = sub.add_parser("eval"); e.add_argument("--run", required=True); e.add_argument("--policy", required=True)
    a = ap.parse_args()
    from .task import build_task
    cfg = _policy_cfg()
    task = build_task(a.run, _hand_xml(), cfg)
    if a.cmd == "train":
        from .train import train_two_stage
        train_two_stage(task, cfg, budget=a.budget, out_dir=a.out)
        print("trained ->", os.path.abspath(a.out))
    else:
        from stable_baselines3 import PPO
        from .evaluate import evaluate
        pi_I = PPO.load(os.path.join(a.policy, "pi_I"))
        pi_R = PPO.load(os.path.join(a.policy, "pi_R"))
        m = evaluate(task, pi_I, pi_R)
        print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
```

Add console scripts to `egoaero/pyproject.toml`:
```toml
[project.scripts]
egoaero-train = "egoaero.policy.cli:main"
egoaero-eval = "egoaero.policy.cli:main"
```
(Both call `main`; the subcommand selects train/eval.)

- [ ] **Step 4: Run tests + ONE real GPU training run**

Run the gated CLI test: `python -m pytest egoaero/tests/policy/test_cli_smoke.py -v` → PASS.
Run the FULL base suite to confirm nothing regressed (sim tests skip if stack absent, run if present): `python -m pytest egoaero/tests/ -q`.
Then perform ONE real short training on the GPU and capture numbers (document, do not commit the policy/run outputs):
```bash
cd egoaero
python -m hoi_recon.cli 2>/dev/null || true   # n/a; build a recon run:
python -m egoaero.cli --out runs/rl_demo --mock --num-frames 64
# reduce budget.real if needed for time; train + eval:
python -m egoaero.policy.cli train --run runs/rl_demo --out runs/rl_demo/policy --budget real
python -m egoaero.policy.cli eval  --run runs/rl_demo --policy runs/rl_demo/policy
```
Record the resulting `Er/Et/Ej/Eft/SR` and the Stage-I reward trend in `egoaero/README.md`. Honesty: a single-clip, single-hand-substitute result will not match the paper; report it as a feasibility demonstration.

- [ ] **Step 5: Commit** (code + docs only; NOT runs/ outputs)

```bash
git add egoaero/egoaero/policy/cli.py egoaero/pyproject.toml egoaero/README.md egoaero/ASSUMPTIONS.md egoaero/tests/policy/test_cli_smoke.py
git commit -m "egoaero: policy CLIs (egoaero-train/eval) + README real-run numbers"
```

---

## Self-Review

**Spec coverage (SP2 spec §3–§7 → tasks):**
- deps/setup/asset/config (§2,§5) → Task 1 ✓
- rewards Stage-I + Stage-II (§3.1) → Tasks 2,3 ✓
- metrics App-H (§3.2) → Task 4 ✓
- retarget IK (§3.3) → Task 5 ✓
- hand_model (§3) → Task 6 ✓ ; task builder + reference (§3.4) → Task 7 ✓
- StageIEnv / StageIIEnv (§3.4) → Tasks 8,9 ✓
- two-stage PPO driver (§3.5) → Task 10 ✓ ; evaluate + ablation (§3.5) → Task 11 ✓
- CLIs + README real-run + ASSUMPTIONS (§4,§6,§8) → Task 12 ✓
- lazy imports / gating (§5, global constraints) → every sim test uses `pytest.importorskip`; rewards/metrics/retarget pure-numpy ✓
- faithfulness map (§6) → reward/metric/residual/structure faithful; defaults logged across Tasks 1,6,7,8,9,11 ✓

**Placeholder scan:** every step has complete code/commands. The sim tasks carry explicit "verify-by-run and adapt to the real MJCF" notes (site tokens, wrist mounting, free-joint addressing) — these are concrete fallback instructions, not TBDs, because MJCF-internal names cannot be known until the asset is vendored.

**Type consistency:** `geodesic_rad` defined in `rewards.py`, reused by `metrics.py`. `r_stage1`/`r_stage2` signatures match env call sites. `HandModel.fk_fingertips(q)->[5,3]` and `fingertip_site_ids` (MANO-keyed) consistent across hand_model/env/evaluate. `load_reference` keys (`wrist_pos`,`fingertips_h`,`obj_pos`,`obj_R`,`contact_active`,`mesh_path`,`T`) consistent across task/env/evaluate. `train_two_stage(task,policy_cfg,budget,out_dir,seed)` and `evaluate(task,pi_I,pi_R,seeds)` consistent across train/evaluate/cli. Budget keys `smoke`/`real` consistent. Fingertip joint indices `[4,8,12,16,20]` match `core/hand.py` layout.

**Known engineering risk (documented, not a plan gap):** the MuJoCo scene composition (hand `include` vs `MjSpec` merge, wrist-base mounting, object free-joint naming, contact-force proxy) is genuine sim-integration work that the implementer must verify against the vendored Shadow Hand MJCF; Tasks 7–9 each carry the specific fallback and an ASSUMPTIONS note. The pure-numpy faithful cores (Tasks 2–5) and App-H metrics are fully deterministic and unaffected.
