# CHOIR Stage 3 Optimizer Integration (#1b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the tested CHOIR energy terms + registry into a runnable Stage-3 optimizer
selected by `fine_preset`, producing a final 4D HOI for `configs/choir_faithful.yaml` /
`configs/combined_v2.yaml` — additive (the existing `joint_opt.py` path for
`configs/combined.yaml` stays untouched, so no regression).

**Architecture:** A pure `hoi_recon/choir_fine/step.py::compute_geometric_terms(state)` maps an
optimization state to the CHOIR geometric term-value dict using the already-tested
`terms_torch`/`anatomical`/`registry` functions (CPU-unit-tested). A new subprocess optimizer
`scripts/subprocess_entries/sam-3d-objects/choir_fine_opt.py` (sam3d env) imports the
`hoi_recon.choir_fine` package via `PYTHONPATH`, builds barycentric contact correspondences
(`choir_fine.contact`) + phases (`choir_fine.phases`), and runs per-group Adam summing terms
via `registry.assemble_energy(weights, {**geometric, **render})`. A driver
`run_choir_fine_optimizer` (in `real_perception.py`) writes the preset weights as JSON and
sets `PYTHONPATH`; `stage7` dispatches to it when `cfg.fine_preset` is set. Spec:
`docs/superpowers/specs/2026-06-12-choir-fine-stage-design.md` §3.1.

**Tech Stack:** Python 3.10, torch 2.11, PyTorch3D + smplx (sam3d env), numpy, pytest.

---

## File Structure

- Create: `hoi_recon/choir_fine/step.py` — state → geometric term-value dict (testable).
- Create: `scripts/subprocess_entries/sam-3d-objects/choir_fine_opt.py` — the optimizer
  (sam3d env subprocess); adapted from the existing `joint_opt.py`.
- Modify: `hoi_recon/backends/real_perception.py` — add `run_choir_fine_optimizer` driver.
- Modify: `hoi_recon/stages/stage7_contact_optim.py` — dispatch on `cfg.fine_preset`.
- Modify: `scripts/setup_third_party.sh` — install the new subprocess script into the clone.
- Test: `tests/test_choir_fine_step.py`.

Tests run in the `hoi_recon` env: `conda run -n hoi_recon python -m pytest ...`.
The smoke run (Task 5) needs the `sam3d-objects` env + a cached `runs/grab` (stages 0–6).

---

## Task 1: `compute_geometric_terms` — state → term-value dict (TDD)

**Files:**
- Create: `hoi_recon/choir_fine/step.py`
- Test: `tests/test_choir_fine_step.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_choir_fine_step.py
import torch
import pytest
from hoi_recon.choir_fine import step, presets, registry, terms_torch as T


def _state(Tf=4):
    """Minimal synthetic optimization state (tiny tensors)."""
    return {
        "hand_c": torch.zeros(Tf, 3, 3), "anchors": torch.zeros(Tf, 3, 2, 3),
        "anc_w": torch.full((Tf, 3, 2), 0.5), "conf": torch.ones(Tf, 3),
        "pen_hand": torch.zeros(Tf, 5, 3), "pen_surf": torch.zeros(Tf, 5, 3),
        "pen_normal": torch.ones(Tf, 5, 3),
        "joints_cam": torch.tensor([[[0.0, 0.0, 1.0]]]).repeat(Tf, 21, 1),
        "kp2d": torch.zeros(Tf, 21, 2), "K": torch.eye(3), "kp_valid": torch.ones(Tf),
        "mano_pose": torch.zeros(Tf, 15, 3),
        "hand_verts": torch.zeros(Tf, 8, 3), "hand_verts_init": torch.zeros(Tf, 8, 3),
        "o_t": torch.zeros(Tf, 3), "o_t0": torch.zeros(Tf, 3),
        "o_r6": torch.zeros(Tf, 6), "o_r60": torch.zeros(Tf, 6),
        "p6": torch.zeros(Tf, 15, 6), "transl": torch.zeros(Tf, 3),
        "g6": torch.zeros(Tf, 6),
        "wrist": torch.zeros(Tf, 3), "wrist_init": torch.zeros(Tf, 3),
    }


def test_returns_all_geometric_term_keys():
    v = step.compute_geometric_terms(_state())
    expected = {"contact", "pen", "anc_2d", "anc_anat", "anc_pose_h", "anc_pose_o",
                "temp_pose_vel", "temp_obj_vel", "temp_wrist_anchor", "temp_hand_tr_vel",
                "temp_root_R_vel", "temp_pose_acc", "temp_hand_tr_acc", "temp_root_R_acc"}
    assert set(v) == expected
    for name, val in v.items():
        assert torch.is_tensor(val) and val.dim() == 0, name


def test_values_match_underlying_term_functions():
    s = _state()
    s["anchors"][:, :, 0, 2] = 0.1                       # anchor 0 is 0.1m away in z
    v = step.compute_geometric_terms(s)
    ref = T.contact_loss(s["hand_c"], s["anchors"], s["anc_w"], s["conf"])
    assert float(v["contact"]) == pytest.approx(float(ref))


def test_assembles_with_a_real_preset():
    """The geometric dict + zeroed render terms must assemble against a real preset
    (every value key has a weight) and be differentiable."""
    s = _state()
    s["o_t"] = s["o_t"].clone().requires_grad_(True)
    v = step.compute_geometric_terms(s)
    v_full = {**v, "sil": torch.zeros(()), "hand_sil": torch.zeros(()),
              "template": torch.zeros(()), "bridge": torch.zeros(()),
              "gap": torch.zeros(()), "patch": torch.zeros(())}
    total = registry.assemble_energy(presets.get_preset("choir_faithful"), v_full)
    assert torch.is_tensor(total)
    total.backward()                                     # must not raise
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n hoi_recon python -m pytest tests/test_choir_fine_step.py -v`
Expected: FAIL — `ModuleNotFoundError: hoi_recon.choir_fine.step`.

- [ ] **Step 3: Write minimal implementation**

```python
# hoi_recon/choir_fine/step.py
"""Map a Stage-3 optimization state to the CHOIR geometric energy-term values (the terms
that do NOT need rendering). The optimizer adds the render-dependent terms (sil, hand_sil)
and assembles everything via registry.assemble_energy. Keys match hoi_recon.choir_fine.presets
TERMS (minus the render + contact-family terms the optimizer handles separately)."""
from __future__ import annotations

from . import terms_torch as T
from .anatomical import anatomical_loss


def compute_geometric_terms(state) -> dict:
    """state: dict of per-frame tensors (see the optimizer for the exact contents).
    Returns {term_name: scalar tensor}."""
    v = {}
    v["contact"] = T.contact_loss(state["hand_c"], state["anchors"], state["anc_w"], state["conf"])
    v["pen"] = T.penetration_loss(state["pen_hand"], state["pen_surf"], state["pen_normal"])
    v["anc_2d"] = T.keypoint_reproj_loss(state["joints_cam"], state["kp2d"], state["K"],
                                         state["kp_valid"])
    v["anc_anat"] = anatomical_loss(state["mano_pose"])
    v["anc_pose_h"] = ((state["hand_verts"] - state["hand_verts_init"]) ** 2).mean()
    v["anc_pose_o"] = (((state["o_t"] - state["o_t0"]) ** 2).mean()
                       + ((state["o_r6"] - state["o_r60"]) ** 2).mean())
    v["temp_pose_vel"] = T.velocity_loss(state["p6"])
    v["temp_obj_vel"] = T.velocity_loss(state["o_t"]) + T.velocity_loss(state["o_r6"])
    v["temp_wrist_anchor"] = ((state["wrist"] - state["wrist_init"]) ** 2).mean()
    v["temp_hand_tr_vel"] = T.velocity_loss(state["transl"])
    v["temp_root_R_vel"] = T.velocity_loss(state["g6"])
    v["temp_pose_acc"] = T.acceleration_loss(state["p6"])
    v["temp_hand_tr_acc"] = T.acceleration_loss(state["transl"])
    v["temp_root_R_acc"] = T.acceleration_loss(state["g6"])
    return v
```

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n hoi_recon python -m pytest tests/test_choir_fine_step.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add hoi_recon/choir_fine/step.py tests/test_choir_fine_step.py
git commit -m "choir_fine: compute_geometric_terms (state -> term-value dict)"
```

---

## Task 2: `choir_fine_opt.py` subprocess optimizer (adapt joint_opt.py)

**Files:**
- Create: `scripts/subprocess_entries/sam-3d-objects/choir_fine_opt.py`

> This optimizer mirrors the existing `scripts/subprocess_entries/sam-3d-objects/joint_opt.py`
> scaffolding (MANO layer, PyTorch3D silhouette render, the chunked render loop) but replaces
> the hand-coded weighted loss with the term registry. Start from a copy of `joint_opt.py` and
> apply the enumerated changes below; the file is a GPU subprocess (no CPU unit test) —
> acceptance is the Task 5 smoke run.

- [ ] **Step 1: Copy the base optimizer**

```bash
cp scripts/subprocess_entries/sam-3d-objects/joint_opt.py \
   scripts/subprocess_entries/sam-3d-objects/choir_fine_opt.py
```

- [ ] **Step 2: Make the package importable + add args**

At the top of `choir_fine_opt.py`, after the existing imports, add (so the sam3d-env process
can import the tested library; the driver also sets PYTHONPATH, this is a belt-and-suspenders):

```python
import json
# hoi_recon.choir_fine is importable via PYTHONPATH (set by the driver). Import the
# tested energy-term library so this optimizer only *assembles* validated pieces.
from hoi_recon.choir_fine import step as choir_step
from hoi_recon.choir_fine import registry as choir_registry
from hoi_recon.choir_fine import contact as choir_contact
from hoi_recon.choir_fine import phases as choir_phases
```

In `main()`'s argparser, replace the individual `--w_*` weight args with:

```python
    ap.add_argument("--weights", required=True, help="json file: {term_name: weight} preset")
    ap.add_argument("--lr_object", type=float, default=3e-4)
    ap.add_argument("--lr_finger", type=float, default=5e-4)
    ap.add_argument("--lr_wrist", type=float, default=5e-5)
```
and set `ap.set_defaults` / change `--iters` default to `800`.

- [ ] **Step 3: Load weights + set CHOIR per-group LRs**

After parsing args, load the preset weights and build the Adam param groups with CHOIR LRs
(replace the existing `opt = torch.optim.Adam([...])` block):

```python
    weights = json.load(open(a.weights))
    opt = torch.optim.Adam([
        {"params": [o_r6, o_t], "lr": a.lr_object},
        {"params": [p6, betas], "lr": a.lr_finger},
        {"params": [g6, transl], "lr": a.lr_wrist},
    ])
```

- [ ] **Step 4: Replace the inline loss with the registry**

Inside the `for it in range(a.iters):` loop, replace the hand-coded `loss = loss + ...`
accumulation AND the chunked `cl = (...)` accumulation with: build the `state` dict, call
`choir_step.compute_geometric_terms(state)`, compute the render terms (object don't-care IoU
silhouette → `sil`, hand silhouette → `hand_sil`; reuse the existing `l_sil`/`l_hsil` render
code), then assemble:

```python
        # ... after computing hv (hand verts), jh (joints), object world verts/poses,
        #     barycentric anchors (anc_pts, anc_w, conf) for this iter ...
        state = {
            "hand_c": hv[:, cidx], "anchors": anc_pts, "anc_w": anc_w, "conf": conf,
            "pen_hand": hv, "pen_surf": pen_surf, "pen_normal": pen_normal,
            "joints_cam": jh, "kp2d": kp_t, "K": Kt, "kp_valid": kpvalid,
            "mano_pose": rot6d_to_aa(p6),            # axis-angle for anatomical
            "hand_verts": hv, "hand_verts_init": hamer_v,
            "o_t": o_t, "o_t0": o_t0, "o_r6": o_r6, "o_r60": o_r60,
            "p6": p6, "transl": transl, "g6": g6,
            "wrist": jh[:, 0], "wrist_init": hamer_j0,
        }
        values = choir_step.compute_geometric_terms(state)
        values["sil"] = l_sil_value          # object don't-care IoU (existing render code)
        values["hand_sil"] = l_hsil_value    # existing hand-silhouette render code
        for k in ("template", "bridge", "gap", "patch"):
            values[k] = torch.zeros((), device=dev)   # annealed contact family (start 0)
        loss = choir_registry.assemble_energy(weights, values)
        opt.zero_grad(); loss.backward(); opt.step()
```

Provide a small `rot6d_to_aa` helper (matrix→axis-angle via `pytorch3d.transforms`) and gather
`pen_surf`/`pen_normal` (nearest object surface point + normal per hand vertex) and the
barycentric `anc_pts/anc_w/conf` from `choir_contact.build_correspondences` on the current
object world mesh, rebuilt every 10 iters (mirror the existing `cache`/`recache` cadence).
Restrict contact terms to manipulation-phase frames via
`choir_phases.segment_phases(visible)`.

- [ ] **Step 5: Keep the output contract + install the script**

Keep the final `np.savez(a.out, hand_verts=..., hand_joints=..., obj_poses=..., visible=...)`.
Then sync into the clone and byte-check it parses:

```bash
cp scripts/subprocess_entries/sam-3d-objects/choir_fine_opt.py third_party/sam-3d-objects/
conda run -n hoi_recon python -m py_compile scripts/subprocess_entries/sam-3d-objects/choir_fine_opt.py
```
Expected: compiles with no error.

- [ ] **Step 6: Commit**

```bash
git add scripts/subprocess_entries/sam-3d-objects/choir_fine_opt.py third_party/sam-3d-objects/choir_fine_opt.py
git commit -m "choir_fine: registry-based Stage-3 subprocess optimizer (choir_fine_opt)"
```

---

## Task 3: `run_choir_fine_optimizer` driver

**Files:**
- Modify: `hoi_recon/backends/real_perception.py`

- [ ] **Step 1: Add the driver (after `run_joint_optimizer`)**

```python
def run_choir_fine_optimizer(cfg, run_dir, s2, s6, frame_paths, mask_paths, K):
    """CHOIR-faithful / combined_v2 Stage-3 optimizer (registry-based). Writes the named
    preset's weights as JSON, runs choir_fine_opt.py in the sam3d env with PYTHONPATH set to
    this repo (so it imports the tested hoi_recon.choir_fine library), and returns
    (hand_verts[T,778,3], hand_joints[T,21,3] or None, obj_poses[T,4,4]). Cached."""
    import subprocess, json
    from ..logging_utils import log
    from ..choir_fine import presets
    from ..config import _REPO_ROOT
    if s2.get("mano_pose") is None:
        raise BackendNotAvailable("CHOIR Stage-3 needs MANO params (re-run stage 2)")
    repo = require_repo(cfg.paths.third_party, "sam-3d-objects", "")
    mano_dir = _resolve_mano_dir(cfg.paths.checkpoints)
    jo_dir = os.path.join(run_dir, "choir_fine"); os.makedirs(jo_dir, exist_ok=True)
    out_npz = os.path.join(jo_dir, "out.npz")
    preset_name = cfg.get("fine_preset", "choir_faithful")
    if not os.path.exists(out_npz):
        hnpz = os.path.join(jo_dir, "hand.npz")
        hand_side = s2.get("hand_side")
        if hand_side is None:
            hand_side = np.ones(len(s2["verts"]), np.int64)
        np.savez(hnpz, mano_global=s2["mano_global"], mano_pose=s2["mano_pose"],
                 mano_betas=s2["mano_betas"], verts=s2["verts"], joints=s2["joints"],
                 contact_idx=s2["contact_idx"], hand_faces=s2["hand_faces"],
                 hand_side=hand_side, kp2d=s2.get("kp2d", np.zeros((len(s2["verts"]), 21, 2))))
        onpz = os.path.join(jo_dir, "obj.npz")
        np.savez(onpz, verts=np.asarray(s6["obj_verts"]), faces=s6["obj_faces"],
                 vertex_colors=s6["obj_colors"], poses=s6["obj_poses"])
        wjson = os.path.join(jo_dir, "weights.json")
        json.dump({k: float(v) for k, v in presets.get_preset(preset_name).items()},
                  open(wjson, "w"))
        Kp = os.path.join(jo_dir, "K.npy"); np.save(Kp, np.asarray(K))
        env_name = (cfg.backend.get("sam3d_env", "sam3d-objects")
                    if hasattr(cfg.backend, "get") else "sam3d-objects")
        conda = os.environ.get("CONDA_EXE", "conda")
        masks_dir = os.path.dirname(os.path.abspath(
            mask_paths[next(i for i, p in enumerate(mask_paths) if p)]))
        frames_dir = os.path.dirname(os.path.abspath(frame_paths[0]))
        occl = _hand_occluder_dir(run_dir)
        cmd = [conda, "run", "--no-capture-output", "-n", env_name, "python",
               os.path.join(repo, "choir_fine_opt.py"), "--hand", os.path.abspath(hnpz),
               "--obj", os.path.abspath(onpz), "--frames_dir", frames_dir,
               "--masks_dir", masks_dir, "--K", os.path.abspath(Kp),
               "--mano_dir", mano_dir, "--out", os.path.abspath(out_npz),
               "--weights", os.path.abspath(wjson), "--iters", "800"]
        if occl:
            cmd += ["--occluder_dir", occl]
        env = {**os.environ, "PYTHONPATH": _REPO_ROOT
               + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")}
        log(f"CHOIR Stage-3 optimizer (preset '{preset_name}', env '{env_name}')...")
        r = subprocess.run(cmd, cwd=repo, env=env)
        if r.returncode != 0 or not os.path.exists(out_npz):
            raise BackendNotAvailable(f"CHOIR Stage-3 optimizer failed (exit {r.returncode}).")
    else:
        log(f"CHOIR Stage-3 optimizer: reusing cached {out_npz}")
    d = np.load(out_npz)
    hj = d["hand_joints"] if "hand_joints" in d.files else None
    return d["hand_verts"], hj, d["obj_poses"]
```

- [ ] **Step 2: Verify it imports**

Run: `conda run -n hoi_recon python -c "from hoi_recon.backends.real_perception import run_choir_fine_optimizer; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add hoi_recon/backends/real_perception.py
git commit -m "choir_fine: run_choir_fine_optimizer driver (weights JSON + PYTHONPATH)"
```

---

## Task 4: stage7 dispatch on `fine_preset`

**Files:**
- Modify: `hoi_recon/stages/stage7_contact_optim.py`

- [ ] **Step 1: Add the dispatch branch**

In `stage7_contact_optim.py`, in the real+differentiable branch, BEFORE the existing
`run_joint_optimizer` call, add a `fine_preset` dispatch:

```python
    fine_preset = cfg.get("fine_preset") if hasattr(cfg, "get") else None
    if not cfg.mock and fine_preset and s6.get("obj_colors") is not None:
        import os as _os
        from ..backends.real_perception import run_choir_fine_optimizer, list_frames
        s0 = ctx.load("stage0_preprocess"); s1 = ctx.load("stage1_detect_track"); s2 = ctx.load("stage2_hand")
        frame_paths = list_frames(s0.assets["frames_dir"])
        mdir = s1.assets["masks_dir"]
        mask_paths = [(_os.path.join(mdir, f"{i:05d}.npy")
                       if _os.path.exists(_os.path.join(mdir, f"{i:05d}.npy")) else None)
                      for i in range(T)]
        hand_verts, hj, poses = run_choir_fine_optimizer(cfg, ctx.stage_dir(NAME), s2, s6,
                                                         frame_paths, mask_paths, s0["intrinsics"])
        hand_joints = hj if hj is not None else s6["hand_joints"]
        hand_c = hand_verts[:, contact_idx]
        d = poses[:, :3, 3] - poses0[:, :3, 3]
        log(f"CHOIR Stage-3 optimizer (preset '{fine_preset}') applied")
    elif not cfg.mock and (o.get("differentiable", False) if hasattr(o, "get") else False) \
            and s6.get("obj_colors") is not None:
        # ... existing run_joint_optimizer branch unchanged ...
```

(Convert the existing `if not cfg.mock and differentiable and ...:` into this `elif` so
`fine_preset` takes precedence; everything below — finalize/contact-map — is unchanged.)

- [ ] **Step 2: Verify mock still runs (no regression)**

Run: `conda run -n hoi_recon python -m hoi_recon.cli --out /tmp/choir_disp_mock --mock --num-frames 16 --stages 7 2>&1 | tail -2`
Expected: stage 7 runs in mock mode (the new branch is skipped because `cfg.mock`).

Run: `conda run -n hoi_recon python -m pytest tests/test_smoke.py -q`
Expected: still passing.

- [ ] **Step 3: Commit**

```bash
git add hoi_recon/stages/stage7_contact_optim.py
git commit -m "choir_fine: stage7 dispatch to the CHOIR Stage-3 optimizer on fine_preset"
```

---

## Task 5: End-to-end smoke run (acceptance)

**Files:** none (verification + a setup-script line)

- [ ] **Step 1: Install the subprocess entry in setup_third_party.sh**

Confirm `scripts/setup_third_party.sh`'s entry-install loop copies
`scripts/subprocess_entries/sam-3d-objects/*.py` into `third_party/sam-3d-objects/` (it copies
all `*.py` in the dir, so `choir_fine_opt.py` is already covered). If a file-list is hardcoded,
add `choir_fine_opt.py`. Run `bash scripts/setup_third_party.sh sam-3d-objects` and confirm the
script reports installing `choir_fine_opt.py`.

- [ ] **Step 2: Smoke run choir_faithful on a cached run dir**

Pre-req: a `runs/grab` with stages 0–6 cached (from `configs/combined.yaml`). Copy its
prefix so stages 0–6 are reused:

```bash
cd /mnt/yijie/code/hoi_recon
mkdir -p runs/grab_choirfine && cp -r runs/grab/stage0_preprocess runs/grab/stage1_detect_track \
  runs/grab/stage2_hand runs/grab/stage3_object runs/grab/stage4_align \
  runs/grab/stage5_coarse_fit runs/grab/stage6_rectify runs/grab/hand_masks runs/grab_choirfine/ 2>/dev/null
conda run --no-capture-output -n hoi_recon python -m hoi_recon.cli --video examples/grab.mp4 \
  --out runs/grab_choirfine --real --config configs/choir_faithful.yaml --stages 7-8 --force \
  2>&1 | grep -E "CHOIR Stage-3|jopt|iter|final:|Error|Traceback|exit" | tail -20
```
Expected: the CHOIR Stage-3 optimizer runs (logs iterations), stage 7 saves, and
`runs/grab_choirfine/stage7_contact_optim/arrays.npz` exists with `hand_verts (T,778,3)` and
`obj_poses (T,4,4)`.

- [ ] **Step 3: Verify the output is sane**

```bash
conda run -n hoi_recon python -c "
import numpy as np
d = np.load('runs/grab_choirfine/stage7_contact_optim/arrays.npz')
assert d['hand_verts'].shape[1:] == (778, 3), d['hand_verts'].shape
assert d['obj_poses'].shape[1:] == (4, 4), d['obj_poses'].shape
assert np.isfinite(d['hand_verts']).all() and np.isfinite(d['obj_poses']).all()
print('CHOIR Stage-3 output OK:', d['hand_verts'].shape, d['obj_poses'].shape)
"
```
Expected: prints `CHOIR Stage-3 output OK ...` (finite, correct shapes).

- [ ] **Step 4: Commit any setup_third_party.sh change + a note**

```bash
git add scripts/setup_third_party.sh 2>/dev/null; git commit -m "choir_fine: install choir_fine_opt entry script" --allow-empty
```

---

## Self-Review notes (addressed)

- **Spec coverage:** Implements spec §3.1 — the term-registry optimizer consuming presets +
  barycentric correspondences + phases, with CHOIR per-group LRs / 800 iters, selected by
  `fine_preset`, additive to the existing pipeline (no regression). The `combined` path
  (`joint_opt.py`) is untouched. Contact-family stabilizers (template/bridge/gap/patch) are
  wired as zero-weight placeholders (CHOIR anneals them; full annealing schedule deferred).
- **Type consistency:** `compute_geometric_terms(state: dict) -> dict` keys ⊂ `presets.TERMS`;
  `run_choir_fine_optimizer(cfg, run_dir, s2, s6, frame_paths, mask_paths, K)` mirrors
  `run_joint_optimizer`; returns `(hand_verts, hand_joints|None, obj_poses)`.
- **Integration caveat (explicit):** Tasks 2 & 5 are GPU-subprocess work verified by a smoke
  run, not unit tests — the testable core (`compute_geometric_terms`) is unit-tested in Task 1.
  Task 2 adapts the existing `joint_opt.py`; the implementer should read it and apply the
  enumerated changes (this is the realistic way to evolve a 300-line GPU optimizer).

## Follow-on plans (not in this plan)

1. **#2 — Evaluation + ablation:** `scripts/object_confidence.py` + `choir_fine.metrics`;
   `scripts/ablate_fine.py` (run choir_faithful vs combined_v2 vs combined, emit the table).
2. **Contact-family annealing schedule** (template/bridge/gap/patch) if the smoke run shows
   contact instability.
3. **#3 Phase 2** generative rectifier; **#4 HO3D GT** adapter.
