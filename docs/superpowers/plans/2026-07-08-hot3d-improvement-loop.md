# HOT3D HOI-Recovery Improvement Loop — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Systematically improve model-free HOI recovery (object pose trajectory from RGB + rendered GT depth) on the frozen 6-clip HOT3D benchmark, tier by tier, committing only gate-passing improvements.

**Architecture:** A leaderboard + acceptance-gate harness wraps the existing `run_batch.py` driver. Four experiment tiers modify stage 1 (SAM2 segmentation) and stage 4 (`_joint_refine` in `object_icp.py`), each behind a config flag so arms are reproducible. Every tier: literature pass → implement → screen on 1 clip (render-and-eyeball) → full 6-clip eval → gate → commit or record failure.

**Tech Stack:** Python (rc5090 conda env: torch 2.11+cu128, SAM2, HaMeR, open3d, trimesh), SAM-3D subprocess (sam3d5090 env), pytest for pure-logic tests.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-08-hot3d-improvement-loop-design.md`.
- Benchmark inputs `/workspace/datasets/hot3d/rc_input_*` are FROZEN — never regenerate them mid-campaign.
- Pipeline may consume RGB + rendered GT depth only; GT poses/models are evaluation-only.
- Acceptance gate (lexicographic): no clip regresses >20% on chamfer or rot_traj vs best committed arm, AND (worst-clip chamfer strictly improves OR (worst-clip chamfer ties within 1 mm AND mean rot_traj strictly improves)).
- A run whose log contains "falling back to depth-lift" is INVALID — rerun once on a free GPU; never score it.
- Render-and-eyeball: before any full 6-clip eval, screen the change on one clip and visually inspect the overlay video.
- Python for all commands: `/workspace/miniconda3/envs/rc5090/bin/python` (call it `$PY`).
- All new experiment behavior is behind config flags, default OFF; flipping a default requires a gate pass.
- Commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Benchmark manifest + arm-aware batch driver

**Files:**
- Create: `/workspace/datasets/hot3d/bench6.json` (data, not committed)
- Modify: `compare/hot3d/run_batch.py`
- Create: symlink `/workspace/datasets/hot3d/rc_input_002500_vase -> rc_input_002500`

**Interfaces:**
- Produces: `run_batch.py <selection.json> [--arm NAME] [--config PATH]` — runs land in `runs/hot3d_<cat>_<num>_<ARM>`, summary in `compare/hot3d/batch_summary_<ARM>.json`. Task 2's leaderboard reads those summaries and the per-run `gt_pose_hot3d_<run>.json` files.

- [ ] **Step 1: Create the frozen benchmark manifest + vase symlink**

```bash
ln -sfn /workspace/datasets/hot3d/rc_input_002500 /workspace/datasets/hot3d/rc_input_002500_vase
cat > /workspace/datasets/hot3d/bench6.json <<'EOF'
[
 {"clip": "clip-002500", "cat": "vase",          "uid": "16"},
 {"clip": "clip-002349", "cat": "potato_masher", "uid": "5"},
 {"clip": "clip-002034", "cat": "bottle_bbq",    "uid": "14"},
 {"clip": "clip-001964", "cat": "puzzle_toy",    "uid": "30"},
 {"clip": "clip-001970", "cat": "mug_white",     "uid": "9"},
 {"clip": "clip-001990", "cat": "spatula_red",   "uid": "6"}
]
EOF
```

- [ ] **Step 2: Add --arm/--config to run_batch.py**

In `compare/hot3d/run_batch.py`, replace the module constant `CFG` usage and `main()` head with:

```python
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("selection")
    ap.add_argument("--arm", default="icpj")
    ap.add_argument("--config", default="configs/real_forehoi_icp_joint.yaml")
    a = ap.parse_args()
    sel = json.load(open(a.selection))
    summary = []
    for item in sel:
        clip, uid, cat = item["clip"], str(item["uid"]), item["cat"]
        num = clip.split("-")[1]
        inp = f"{DS}/rc_input_{num}_{cat}"
        run = f"{RC}/runs/hot3d_{cat}_{num}_{a.arm}"
```

and inside the loop replace `CFG` with `a.config`; at the end write
`f"{HERE}/batch_summary_{a.arm}.json"` instead of `batch_summary.json`.
Also add the depth-lift INVALID check right after the pipeline call:

```python
            log_txt = ""
            lp = f"{run}.log"
            if os.path.exists(lp):
                log_txt = open(lp, errors="ignore").read()
            if "falling back to depth-lift" in log_txt:
                raise RuntimeError("INVALID: stage3 fell back to depth-lift")
```

and change the pipeline invocation to tee its output to that log:

```python
                with open(lp, "w") as lf:
                    subprocess.run([PY, "-m", "hoi_recon.cli", "--video",
                        f"{inp}/rgb.mp4", "--out", run, "--real", "--config",
                        a.config, "--depth", "gt", "--object-prompt",
                        f"{px:.1f}", f"{py_:.1f}"], check=True, cwd=RC,
                        env=env, stdout=lf, stderr=subprocess.STDOUT)
```

(Keep the existing overlay + eval + RESULT lines; overlay filename becomes
`rc_vs_gt_{cat}_{num}_{a.arm}.mp4`.)

- [ ] **Step 3: Smoke-test the driver on the already-complete baseline**

```bash
$PY compare/hot3d/run_batch.py /workspace/datasets/hot3d/bench6.json --arm icpj
```
Expected: all 6 clips skip the pipeline (stage8 exists), re-render overlays/evals quickly, and `compare/hot3d/batch_summary_icpj.json` appears with 6 rows. (The vase run dir is `hot3d_vase_icpj` — matching the existing dir name; verify no re-run happened.)

Note: the existing vase run is named `runs/hot3d_vase_icpj` while the driver
computes `runs/hot3d_vase_002500_icpj`. Symlink it rather than recompute:

```bash
ln -sfn /workspace/code/hoi_recon/render_and_compare/runs/hot3d_vase_icpj \
        /workspace/code/hoi_recon/render_and_compare/runs/hot3d_vase_002500_icpj
```

- [ ] **Step 4: Commit**

```bash
git add compare/hot3d/run_batch.py compare/hot3d/batch_summary_icpj.json
git commit -m "loop: arm-aware batch driver + frozen bench6 manifest"
```

---

### Task 2: Leaderboard + acceptance gate (pure logic, tested)

**Files:**
- Create: `compare/hot3d/leaderboard.py`
- Test: `compare/hot3d/tests/test_gate.py`
- Create: `compare/hot3d/LEADERBOARD.md` (generated)

**Interfaces:**
- Consumes: `batch_summary_<arm>.json` rows `{"clip","cat","chamfer_mm","centroid_cm","rot_traj_med","rot_traj_p90","rot_abs_med"}`.
- Produces: `gate(cand: dict, best: dict) -> (bool, str)` where each dict maps cat -> row; CLI `leaderboard.py check <arm>` prints `GATE PASS`/`GATE FAIL <reason>` and exit code 0/1; `leaderboard.py render` rewrites LEADERBOARD.md from all summaries; the file records which arm is `**best**`.

- [ ] **Step 1: Write the failing tests**

`compare/hot3d/tests/test_gate.py`:

```python
import copy
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from leaderboard import gate

BASE = {c: {"chamfer_mm": ch, "rot_traj_med": rt}
        for c, ch, rt in [("vase", 17.5, 19.4), ("mug_white", 60.7, 18.0),
                          ("spatula_red", 158.8, 32.8)]}

def test_worst_chamfer_improves_passes():
    cand = copy.deepcopy(BASE)
    cand["spatula_red"]["chamfer_mm"] = 25.0     # worst clip fixed
    ok, why = gate(cand, BASE)
    assert ok, why

def test_regression_blocks():
    cand = copy.deepcopy(BASE)
    cand["spatula_red"]["chamfer_mm"] = 25.0
    cand["vase"]["rot_traj_med"] = 25.0          # +29% > 20% regression
    ok, why = gate(cand, BASE)
    assert not ok and "vase" in why

def test_tie_needs_rot_improvement():
    cand = copy.deepcopy(BASE)                   # chamfer all tied
    ok, _ = gate(cand, BASE)
    assert not ok                                # nothing improved
    cand["mug_white"]["rot_traj_med"] = 10.0     # mean rot_traj drops
    ok, why = gate(cand, BASE)
    assert ok, why
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
$PY -m pytest compare/hot3d/tests/test_gate.py -q
```
Expected: FAIL / collection error `No module named 'leaderboard'`. (If pytest missing: `$PY -m pip install -q pytest`.)

- [ ] **Step 3: Implement leaderboard.py**

```python
"""Leaderboard + acceptance gate for the HOT3D improvement loop (see spec
docs/superpowers/specs/2026-07-08-hot3d-improvement-loop-design.md)."""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REG = 1.20          # per-clip regression tolerance
TIE_MM = 1.0        # worst-chamfer tie band


def load_arm(arm):
    p = f"{HERE}/batch_summary_{arm}.json"
    rows = json.load(open(p))
    return {r["cat"]: r for r in rows if "error" not in r}


def gate(cand, best):
    if set(cand) != set(best):
        return False, f"clip sets differ: {sorted(set(best) ^ set(cand))}"
    for c in best:
        for k in ("chamfer_mm", "rot_traj_med"):
            if cand[c][k] > best[c][k] * REG:
                return False, f"{c} regresses on {k}: {best[c][k]} -> {cand[c][k]}"
    wc_c = max(v["chamfer_mm"] for v in cand.values())
    wc_b = max(v["chamfer_mm"] for v in best.values())
    if wc_c < wc_b - TIE_MM:
        return True, f"worst-clip chamfer {wc_b:.1f} -> {wc_c:.1f} mm"
    if abs(wc_c - wc_b) <= TIE_MM:
        mr_c = sum(v["rot_traj_med"] for v in cand.values()) / len(cand)
        mr_b = sum(v["rot_traj_med"] for v in best.values()) / len(best)
        if mr_c < mr_b:
            return True, f"chamfer tied; mean rot_traj {mr_b:.1f} -> {mr_c:.1f} deg"
        return False, f"chamfer tied; mean rot_traj {mr_b:.1f} -> {mr_c:.1f} (no gain)"
    return False, f"worst-clip chamfer worsens {wc_b:.1f} -> {wc_c:.1f} mm"


def best_arm():
    p = f"{HERE}/BEST_ARM"
    return open(p).read().strip() if os.path.exists(p) else "icpj"


def render():
    lines = ["# HOT3D 6-clip leaderboard", "",
             "| arm | clip | chamfer mm | centroid cm | rot_traj | p90 | rot_abs |",
             "|---|---|---|---|---|---|---|"]
    for p in sorted(glob.glob(f"{HERE}/batch_summary_*.json")):
        arm = os.path.basename(p)[len("batch_summary_"):-len(".json")]
        tag = f"**{arm}**" if arm == best_arm() else arm
        for r in json.load(open(p)):
            if "error" in r:
                lines.append(f"| {tag} | {r['cat']} | ERROR | | | | |")
                continue
            lines.append(f"| {tag} | {r['cat']} | {r['chamfer_mm']} | "
                         f"{r['centroid_cm']} | {r['rot_traj_med']} | "
                         f"{r['rot_traj_p90']} | {r['rot_abs_med']} |")
    open(f"{HERE}/LEADERBOARD.md", "w").write("\n".join(lines) + "\n")
    print(f"rendered LEADERBOARD.md (best={best_arm()})")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "render":
        render()
    elif cmd == "check":
        ok, why = gate(load_arm(sys.argv[2]), load_arm(best_arm()))
        print(("GATE PASS: " if ok else "GATE FAIL: ") + why)
        if ok:
            open(f"{HERE}/BEST_ARM", "w").write(sys.argv[2])
            render()
        sys.exit(0 if ok else 1)
```

- [ ] **Step 4: Run tests, verify pass**

```bash
$PY -m pytest compare/hot3d/tests/test_gate.py -q
```
Expected: `3 passed`.

- [ ] **Step 5: Seed the leaderboard with the icpj baseline and commit**

```bash
printf icpj > compare/hot3d/BEST_ARM
$PY compare/hot3d/leaderboard.py render
git add compare/hot3d/leaderboard.py compare/hot3d/tests/test_gate.py \
        compare/hot3d/LEADERBOARD.md compare/hot3d/BEST_ARM
git commit -m "loop: leaderboard + lexicographic acceptance gate (tested)"
```

---

### Task 3: T1 — hand-aware SAM2 segmentation

**Files:**
- Modify: `render_and_compare/hoi_recon/backends/real_perception.py:336-364` (`segment_object`)
- Modify: `render_and_compare/hoi_recon/stages/stage1_detect_track.py:75-76` (pass hand boxes)
- Create: `render_and_compare/hoi_recon/mask_qa.py`
- Test: `compare/hot3d/tests/test_mask_qa.py`
- Create: `render_and_compare/configs/real_forehoi_icp_joint_seg.yaml` (arm `icpjs`)

**Interfaces:**
- Consumes: `detect_hands` output `hand_boxes[T,2,4] xyxy, hand_valid[T,2]` (already computed before `segment_object` in stage 1).
- Produces: `segment_object(cfg, frames_dir, frame_paths, prompt_xy, out_dir, hand_boxes=None, hand_valid=None)`; `mask_qa.qa_report(mask_paths, hand_boxes, hand_valid) -> dict` with keys `area[T]`, `tiou[T-1]`, `hand_overlap[T]`, `bad: bool`, `best_frame: int`. Behavior gated by `cfg.backend.hand_aware_seg` (default False).

- [ ] **Step 1: Literature pass (15 min, timeboxed)**

Search and skim; record 5-line notes at the top of the task commit message:
```bash
$PY /home/yijie/.claude/skills/paper_search/scripts/search_papers.py \
  --query "hand-held object segmentation video hand occlusion SAM negative prompt" \
  --start-year 2023 --end-year 2026 --max-papers 6 \
  --sources arxiv semantic_scholar
```
Plus WebSearch: "SAM2 video multi-object negative prompt subtract hand mask in-hand object segmentation 2025". Adopt any directly usable trick; otherwise proceed with the design below.

- [ ] **Step 2: Write failing tests for mask QA**

`compare/hot3d/tests/test_mask_qa.py`:

```python
import numpy as np, os, sys, tempfile
sys.path.insert(0, "/workspace/code/hoi_recon/render_and_compare")
from hoi_recon.mask_qa import qa_report

def _save(dirn, masks):
    ps = []
    for i, m in enumerate(masks):
        p = os.path.join(dirn, f"{i:05d}.npy"); np.save(p, m); ps.append(p)
    return ps

def test_stable_masks_pass():
    with tempfile.TemporaryDirectory() as d:
        m = np.zeros((64, 64), bool); m[20:40, 20:40] = True
        ps = _save(d, [m] * 10)
        r = qa_report(ps, None, None)
        assert not r["bad"] and r["best_frame"] in range(10)

def test_area_explosion_flags_bad():
    with tempfile.TemporaryDirectory() as d:
        small = np.zeros((64, 64), bool); small[20:40, 20:40] = True
        big = np.zeros((64, 64), bool); big[5:60, 5:60] = True   # 7.5x area
        ps = _save(d, [small] * 5 + [big] * 5)
        r = qa_report(ps, None, None)
        assert r["bad"] and r["best_frame"] < 5     # anchor from stable phase

def test_hand_overlap_flags_bad():
    with tempfile.TemporaryDirectory() as d:
        m = np.zeros((64, 64), bool); m[20:50, 20:50] = True
        ps = _save(d, [m] * 6)
        hb = np.tile(np.array([[[18., 18., 52., 52.], [np.nan]*4]]), (6, 1, 1))
        hv = np.tile(np.array([[True, False]]), (6, 1))
        r = qa_report(ps, hb, hv)
        assert r["hand_overlap"].mean() > 0.9 and r["bad"]
```

- [ ] **Step 3: Run tests, verify fail**

```bash
$PY -m pytest compare/hot3d/tests/test_mask_qa.py -q
```
Expected: `No module named 'hoi_recon.mask_qa'`.

- [ ] **Step 4: Implement `hoi_recon/mask_qa.py`**

```python
"""Mask quality gates for stage-1 object segmentation.

The two catastrophic HOT3D failures (mug+forearm merge, spatula->table leak)
were both silent stage-1 mask defects. This module quantifies the defects the
overlay videos showed: area jumps (leak), low temporal IoU (identity drift),
and high hand-box overlap (arm absorbed into the object mask)."""
import numpy as np

AREA_JUMP = 2.5        # x median area
TIOU_MIN = 0.45
HAND_OVERLAP_MAX = 0.55


def qa_report(mask_paths, hand_boxes, hand_valid):
    T = len(mask_paths)
    area = np.zeros(T)
    tiou = np.zeros(max(T - 1, 0))
    hov = np.zeros(T)
    prev = None
    for i, mp in enumerate(mask_paths):
        m = np.load(mp) if mp else np.zeros((1, 1), bool)
        area[i] = m.sum()
        if prev is not None and prev.shape == m.shape:
            u = (prev | m).sum()
            tiou[i - 1] = (prev & m).sum() / u if u else 0.0
        if hand_boxes is not None and hand_valid is not None and m.any():
            hb = np.zeros(m.shape, bool)
            H, W = m.shape
            for h in range(hand_boxes.shape[1]):
                b = hand_boxes[i, h]
                if hand_valid[i, h] and np.isfinite(b).all():
                    x0, y0, x1, y1 = np.clip(
                        b, 0, [W, H, W, H]).astype(int)
                    hb[y0:y1, x0:x1] = True
            hov[i] = (m & hb).sum() / m.sum()
        prev = m
    med = np.median(area[area > 0]) if (area > 0).any() else 0.0
    bad = bool((med > 0 and (area > AREA_JUMP * med).any())
               or (len(tiou) and np.median(tiou) < TIOU_MIN)
               or (hov.mean() > HAND_OVERLAP_MAX))
    # anchor candidate: large, stable, hand-free mask
    score = area / max(med, 1.0)
    score[area > AREA_JUMP * med] = 0
    score = score * (1.0 - hov)
    return {"area": area, "tiou": tiou, "hand_overlap": hov,
            "bad": bad, "best_frame": int(np.argmax(score))}
```

- [ ] **Step 5: Run tests, verify pass; commit the QA module**

```bash
$PY -m pytest compare/hot3d/tests/test_mask_qa.py -q   # expect 3 passed
git add render_and_compare/hoi_recon/mask_qa.py compare/hot3d/tests/test_mask_qa.py
git commit -m "loop T1: mask QA module (area/tiou/hand-overlap gates)"
```

- [ ] **Step 6: Hand-aware segment_object**

Modify `segment_object` in `real_perception.py` — new signature and body
(replacing lines 336-364), keeping the external-mask branch:

```python
def segment_object(cfg, frames_dir, frame_paths, prompt_xy, out_dir,
                   hand_boxes=None, hand_valid=None):
    """SAM 2 video segmentation from a point prompt.

    hand_aware_seg (cfg.backend, default False): hands are tracked as their
    own SAM2 objects (positive click at each valid hand-box centre on the
    frame where the box is most confident) and subtracted from the object
    mask per frame — SAM2 otherwise merges a held object with the arm (the
    HOT3D mug failure). A mask-QA pass (hoi_recon.mask_qa) then re-prompts
    from a cleaner anchor frame when the track is bad."""
    pattern = os.environ.get("RC_OBJECT_MASK_PATTERN")
    if pattern:
        return _external_object_masks(pattern, frame_paths, out_dir)
    hand_aware = bool(cfg.backend.get("hand_aware_seg", False)
                      if hasattr(cfg.backend, "get") else False)
    masks_dir = os.path.join(out_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    mask_paths = _run_sam2_once(cfg, frames_dir, len(frame_paths), masks_dir,
                                prompt_xy, 0, hand_boxes, hand_valid,
                                hand_aware)
    if hand_aware:
        from ..mask_qa import qa_report
        from ..logging_utils import log
        r = qa_report(mask_paths, hand_boxes, hand_valid)
        log(f"mask QA: bad={r['bad']} hand_overlap={r['hand_overlap'].mean():.2f} "
            f"best_frame={r['best_frame']}")
        if r["bad"] and r["best_frame"] != 0:
            m = np.load(mask_paths[r["best_frame"]])
            ys, xs = np.where(m)
            re_prompt = (float(xs.mean()), float(ys.mean()))
            log(f"mask QA re-prompt @ frame {r['best_frame']} {re_prompt}", "warn")
            mask_paths = _run_sam2_once(cfg, frames_dir, len(frame_paths),
                                        masks_dir, re_prompt,
                                        r["best_frame"], hand_boxes,
                                        hand_valid, hand_aware)
    return masks_dir, mask_paths


def _run_sam2_once(cfg, frames_dir, T, masks_dir, prompt_xy, prompt_frame,
                   hand_boxes, hand_valid, hand_aware):
    """One SAM2 video pass. Object = obj_id 1 (positive click); each hand =
    its own obj_id (2, 3) prompted by a box, tracked jointly and subtracted.
    Propagates forward AND backward from prompt_frame."""
    import torch
    predictor = _load_sam2(cfg)
    obj = {}
    with torch.inference_mode(), torch.autocast(_device(), dtype=torch.bfloat16):
        state = predictor.init_state(video_path=frames_dir)
        predictor.add_new_points_or_box(
            state, frame_idx=prompt_frame, obj_id=1,
            points=np.array([prompt_xy], np.float32),
            labels=np.array([1], np.int32))
        if hand_aware and hand_boxes is not None:
            for h in range(hand_boxes.shape[1]):
                if not hand_valid[prompt_frame, h]:
                    continue
                b = hand_boxes[prompt_frame, h]
                if not np.isfinite(b).all():
                    continue
                predictor.add_new_points_or_box(
                    state, frame_idx=prompt_frame, obj_id=2 + h,
                    box=b.astype(np.float32))
        def _consume(it):
            for fidx, ids, logits in it:
                ms = {int(o): (logits[k] > 0).squeeze().cpu().numpy().astype(bool)
                      for k, o in enumerate(ids)}
                m = ms.get(1, np.zeros_like(next(iter(ms.values()))))
                for o, hm in ms.items():
                    if o != 1:
                        m = m & ~hm
                obj[fidx] = m
        _consume(predictor.propagate_in_video(state))
        if prompt_frame > 0:
            _consume(predictor.propagate_in_video(state, reverse=True))
    mask_paths = [None] * T
    for fidx, m in obj.items():
        mp = os.path.join(masks_dir, f"{fidx:05d}.npy")
        np.save(mp, m)
        mask_paths[fidx] = mp
    return mask_paths
```

In `stage1_detect_track.py` change the call (line 75) to:

```python
    masks_dir, mask_paths = segment_object(cfg, frames_dir, frame_paths, prompt,
                                           ctx.stage_dir(NAME),
                                           hand_boxes=hand_boxes,
                                           hand_valid=hand_valid)
```

- [ ] **Step 7: Create the arm config**

`render_and_compare/configs/real_forehoi_icp_joint_seg.yaml` — copy of
`real_forehoi_icp_joint.yaml` with one added line under `backend:`:

```yaml
  hand_aware_seg: true   # T1: track hands as SAM2 objects, subtract + mask QA
```

- [ ] **Step 8: Screen on the mug clip + eyeball**

```bash
cd render_and_compare
printf '[{"clip": "clip-001970", "cat": "mug_white", "uid": "9"}]' \
  > /workspace/datasets/hot3d/screen_mug.json
$PY ../compare/hot3d/run_batch.py /workspace/datasets/hot3d/screen_mug.json \
  --arm icpjs --config configs/real_forehoi_icp_joint_seg.yaml
```
Then extract 3 frames of `compare/hot3d/rc_vs_gt_mug_white_001970_icpjs.mp4`
(frames 20/75/130) and LOOK at them: the orange estimate must be a mug
without an arm-shaped appendage. Also check the run log's `mask QA:` line.
If the mask still merges the arm: iterate (allowed variants, in order: box
prompts per-frame for hands; negative click at forearm centroid =
hand-box centre displaced away from the object prompt; erode object mask
where it borders the hand mask). Two failed variants → record findings in
BEST_STRATEGY.md, move to Task 4.

- [ ] **Step 9: Screen on spatula, then full bench, gate**

```bash
printf '[{"clip": "clip-001990", "cat": "spatula_red", "uid": "6"}]' \
  > /workspace/datasets/hot3d/screen_spatula.json
$PY ../compare/hot3d/run_batch.py /workspace/datasets/hot3d/screen_spatula.json \
  --arm icpjs --config configs/real_forehoi_icp_joint_seg.yaml
# eyeball rc_vs_gt_spatula_red_001990_icpjs.mp4 the same way, then:
$PY ../compare/hot3d/run_batch.py /workspace/datasets/hot3d/bench6.json \
  --arm icpjs --config configs/real_forehoi_icp_joint_seg.yaml
$PY ../compare/hot3d/leaderboard.py check icpjs
```
Expected on pass: `GATE PASS: worst-clip chamfer 158.8 -> <value> mm`.

- [ ] **Step 10: Commit (win or findings)**

```bash
git add -A render_and_compare/hoi_recon render_and_compare/configs \
        compare/hot3d
git commit -m "loop T1: hand-aware SAM2 segmentation (arm icpjs) — <GATE PASS/FAIL + numbers>"
```

---

### Task 4: T2 — photometric azimuth (anchor attitude search + in-loop color term)

**Files:**
- Modify: `render_and_compare/hoi_recon/object_icp.py` (`_joint_refine`, `refine_object_poses`)
- Modify: `render_and_compare/hoi_recon/stages/stage4_align.py` (pass vertex colors + frames dir)
- Create: `render_and_compare/configs/real_forehoi_icp_joint_photo.yaml` (arm `icpjp`, built on the best committed arm's config)

**Interfaces:**
- Consumes: stage-3 `obj_colors[N,3]` (float 0-1, vertex colors), stage-0 frames dir; `_joint_refine`'s existing tensors (`src`, `sil_idx`, `R0/t0`, projection code at lines 222-232).
- Produces: `refine_object_poses(..., obj_colors=None, frames_dir=None)`; config keys `w_photo` (default 0.0 = off), `attitude_search: 0` (N azimuth hypotheses, 0 = off).

- [ ] **Step 1: Literature pass (15 min)**

```bash
$PY /home/yijie/.claude/skills/paper_search/scripts/search_papers.py \
  --query "render-and-compare photometric 6D object pose refinement differentiable color" \
  --start-year 2023 --end-year 2026 --max-papers 6 --sources arxiv semantic_scholar
```
WebSearch: "texture-based azimuth disambiguation symmetric object pose 2025". Note findings, proceed.

- [ ] **Step 2: Point colors into the refiner**

In `stage4_align.py`, where `refine_object_poses` is called, add
`obj_colors=s3["obj_colors"], frames_dir=s0.assets["frames_dir"]` (s3/s0
are the loaded stage bundles; obj_colors shape (Nverts,3) in 0-1).
In `refine_object_poses`, sample per-surface-point colors exactly where
`src_pts` are sampled from the mesh (it uses `trimesh.sample.sample_surface`
— capture the face indices and barycentric-free approximation: take the
face's mean vertex color):

```python
    src_cols = None
    if obj_colors is not None:
        fc = np.asarray(obj_colors)[np.asarray(mesh.faces)].mean(1)  # per-face
        src_cols = fc[src_fidx]          # src_fidx from sample_surface
```
and pass `src_cols, frames_dir` through to `_joint_refine`.

- [ ] **Step 3: In-loop photometric term in `_joint_refine`**

Add after the DT/coverage setup (uses the already-projected `u, v` of the
silhouette subset; new precompute + loss term):

```python
    # photometric: LAB chroma of each projected colored point vs the image.
    # Differentiable via grid_sample on the (downsampled) chroma planes.
    w_photo = float(get("w_photo", 0.0))
    if w_photo > 0 and src_cols is not None and frames_dir is not None:
        import cv2
        labs = np.zeros((T, 2, Hd, Wd), np.float32)
        for i in valid:
            img = cv2.imread(os.path.join(frames_dir, f"{i:05d}.jpg"))
            img = cv2.resize(img, (Wd, Hd), interpolation=cv2.INTER_AREA)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
            labs[i] = lab[..., 1:].transpose(2, 0, 1) - 128.0
        labs_t = tt(labs)
        pc = np.asarray(src_cols, np.float32)[rng_idx := sil_idx.cpu().numpy()]
        lab_pts = cv2.cvtColor((pc[None] * 255).astype(np.uint8),
                               cv2.COLOR_RGB2LAB)[0].astype(np.float32)
        pt_ab = tt(lab_pts[:, 1:] - 128.0)          # [n_sil, 2]
```
and inside the inner loop, after `dt_s` is computed:

```python
            if w_photo > 0 and src_cols is not None and frames_dir is not None:
                im_ab = F.grid_sample(labs_t[vf], grid, align_corners=False,
                                      padding_mode="border")[:, :, 0]  # [F,2,n]
                d_ab = (im_ab.permute(0, 2, 1) - pt_ab.unsqueeze(0))
                inlier = (dt_s < 2.0).float()        # only points inside mask
                loss_p = ((d_ab ** 2).sum(-1).sqrt() * inlier).sum() \
                    / inlier.sum().clamp(min=1.0) / 10.0   # ~unit scale
                loss = loss + w_photo * loss_p
```
(`loss_p` joins the round log line.)

- [ ] **Step 4: Anchor attitude search in `refine_object_poses`**

Before the joint refine, when `attitude_search = int(get("attitude_search", 0)) > 1`:

```python
    if n_hyp > 1:
        a = int(np.nanargmin(resid_init))            # most reliable frame
        best = (np.inf, np.eye(3))
        for k in range(n_hyp):
            ang = 2 * np.pi * k / n_hyp
            Rh = poses0[a][:3, :3] @ _rotz(ang)      # azimuth about object up
            sc = _score_hypothesis(src_pts, targets[a], masks_dir, K, Rh,
                                   poses0[a][:3, 3], s0, src_cols,
                                   frames_dir, a)
            if sc < best[0]:
                best = (sc, _rotz(ang))
        G = best[1]
        for i in range(len(poses0)):                  # rotate whole track
            poses0[i][:3, :3] = poses0[i][:3, :3] @ G
```
with `_rotz(a) = np.array([[cos,-sin,0],[sin,cos,0],[0,0,1]])` and
`_score_hypothesis` = trimmed depth RMS (KDTree, as in `_joint_refine`'s
correspondence refresh) + mean out-of-mask DT of 500 projected points +
(if colors available) mean LAB-chroma distance of in-mask points — all
computed numpy-only at the single anchor frame. The object "up"/symmetry
axis for the azimuth grid: use the canonical axis with the SMALLEST
per-axis scale sensitivity — practically, run the grid about each of the 3
canonical axes with n_hyp/3 each when unsure (9-24 hypotheses total,
<10 s).

- [ ] **Step 5: Arm config + screen on cube and bottle**

`real_forehoi_icp_joint_photo.yaml` = best committed config + :

```yaml
object_icp:
  w_photo: 0.5
  attitude_search: 12
```

```bash
printf '[{"clip": "clip-001964", "cat": "puzzle_toy", "uid": "30"}]' \
  > /workspace/datasets/hot3d/screen_cube.json
$PY ../compare/hot3d/run_batch.py /workspace/datasets/hot3d/screen_cube.json \
  --arm icpjp --config configs/real_forehoi_icp_joint_photo.yaml
```
Eyeball the cube overlay (colored faces must land on the right sides) and
compare `rot_traj_med` in the RESULT line vs icpj's 82.7. Then screen
bottle_bbq (39.3 baseline). Tune `w_photo` in {0.2, 0.5, 1.0} if needed
(2-failure rule applies). Then full bench + gate + commit, exactly as Task
3 steps 9-10 with arm `icpjp`.

---

### Task 5: T3 — grasp-rigidity prior

**Files:**
- Create: `render_and_compare/hoi_recon/grasp_segments.py`
- Test: `compare/hot3d/tests/test_grasp_segments.py`
- Modify: `render_and_compare/hoi_recon/object_icp.py` (`_joint_refine`: new term), `stages/stage4_align.py` (pass wrist trajectories from stage-2 `hand_joints[T,21,3]`, joint 0 = wrist)
- Create: `render_and_compare/configs/real_forehoi_icp_joint_grasp.yaml` (arm `icpjgr`, on best committed config)

**Interfaces:**
- Consumes: stage-2 `hand_joints[T,21,3]` per side (wrist = index 0), init object poses `poses0[T,4,4]`.
- Produces: `stable_grasp_mask(wrist[T,3], obj_t[T,3], v_rel_max=0.015, win=5) -> bool[T]` (True where relative speed per frame < v_rel_max over a centred window); `_joint_refine` config key `w_grasp` (default 0.0).

- [ ] **Step 1: Write failing test**

`compare/hot3d/tests/test_grasp_segments.py`:

```python
import numpy as np, sys
sys.path.insert(0, "/workspace/code/hoi_recon/render_and_compare")
from hoi_recon.grasp_segments import stable_grasp_mask

def test_rigid_motion_detected():
    t = np.linspace(0, 1, 60)[:, None]
    wrist = np.concatenate([t, t * 0.5, 1.0 + 0 * t], 1)   # moving hand
    obj = wrist + np.array([0.03, 0.01, 0.02])             # rigid offset
    m = stable_grasp_mask(wrist, obj)
    assert m[10:50].all()

def test_independent_motion_rejected():
    rng = np.random.default_rng(0)
    wrist = rng.normal(0, 0.05, (60, 3)).cumsum(0)
    obj = np.zeros((60, 3))                                # static object
    m = stable_grasp_mask(wrist, obj)
    assert not m[10:50].any()
```

- [ ] **Step 2: Run tests (fail), implement, run tests (pass)**

`hoi_recon/grasp_segments.py`:

```python
"""Stable-grasp detection: frames where the object moves rigidly with a
wrist. During such segments the hand's rotation carries the azimuth signal
that symmetric object geometry hides from depth+silhouette (HOT3D batch
finding: rotation error is a direct function of shape symmetry)."""
import numpy as np


def stable_grasp_mask(wrist, obj_t, v_rel_max=0.015, win=5):
    rel = obj_t - wrist
    v = np.zeros(len(rel))
    v[1:] = np.linalg.norm(np.diff(rel, axis=0), axis=1)
    k = np.ones(win) / win
    vs = np.convolve(v, k, mode="same")
    moving = np.zeros(len(rel))
    moving[1:] = np.linalg.norm(np.diff(wrist, axis=0), axis=1)
    ms = np.convolve(moving, k, mode="same")
    # grasp = relative motion small WHILE the hand actually moves
    return (vs < v_rel_max) & (ms > v_rel_max)
```

```bash
$PY -m pytest compare/hot3d/tests/test_grasp_segments.py -q   # 2 passed
git add render_and_compare/hoi_recon/grasp_segments.py compare/hot3d/tests/test_grasp_segments.py
git commit -m "loop T3: stable-grasp detector (tested)"
```

- [ ] **Step 3: Rigidity term in `_joint_refine`**

`stage4_align.py`: compute per-side wrist trajectories `wrists[S,T,3]` from
stage-2 `hand_joints` (and per-side validity), call `stable_grasp_mask`
against `poses0[:,:3,3]`, pick the side with the most grasp frames, and pass
`grasp_mask[T] (bool)`, `wrist_R[T,3,3]` (HaMeR global orientation per
frame; if only joints are stored, build a wrist frame from joints 0,5,17:
x = j5-j0 normalized, z = x × (j17-j0) normalized, y = z × x) into
`refine_object_poses` → `_joint_refine`.

In `_joint_refine` add (setup):

```python
    w_grasp = float(get("w_grasp", 0.0))
    if w_grasp > 0 and grasp_mask is not None and grasp_mask.sum() >= 3:
        gi = np.where(grasp_mask[:-1] & grasp_mask[1:])[0]  # consecutive pairs
        gi_t = tt(gi, torch.long)
        Rw = tt(wrist_R)
        dRw = Rw[gi + 1] @ Rw[gi].transpose(1, 2)           # wrist deltas
```
and in the inner loop:

```python
            if w_grasp > 0 and grasp_mask is not None and grasp_mask.sum() >= 3:
                dRo = R[gi_t + 1] @ R[gi_t].transpose(1, 2)
                loss_g = ((dRo - dRw) ** 2).sum((1, 2)).mean() * 1600.0
                loss = loss + w_grasp * loss_g
```

- [ ] **Step 4: Arm config, screen, bench, gate, commit**

`real_forehoi_icp_joint_grasp.yaml` = best committed config + `w_grasp: 1.0`.
Screen on vase (its in-hand phase f100-150 is the known rot_traj blow-up:
74-122°) — the RESULT rot_traj_p90 must drop vs the committed arm; eyeball
the overlay's final 50 frames. Then bench6 + `leaderboard.py check icpjgr` +
commit, as in Task 3 steps 9-10.

---

### Task 6: T4 — learned tracker bake-off (bounded exploration)

**Files:**
- Create: `compare/hot3d/T4_NOTES.md` (findings)
- Possible: `render_and_compare/third_party/<method>/`, new conda env, stage-4 backend adapter

**Interfaces:**
- Consumes: frozen bench6 inputs; the committed best arm as the bar.
- Produces: either a new stage-4 backend + arm `<method>` on the leaderboard, or a T4_NOTES.md verdict explaining why none was integrated.

- [ ] **Step 1: Literature pass (30 min, wider)**

```bash
$PY /home/yijie/.claude/skills/paper_search/scripts/search_papers.py \
  --query "model-free unknown object 6DoF tracking reconstruction RGB-D video" \
  --start-year 2024 --end-year 2026 --max-papers 10
```
Plus WebSearch for: "BundleSDF successor 2025 2026", "neural object SLAM
hand occlusion", "HOT3D benchmark leaderboard object pose methods".
Write `compare/hot3d/T4_NOTES.md`: candidate table (method, inputs, code
availability, torch/CUDA pins vs Blackwell sm_120, expected fit).

- [ ] **Step 2: CHECKPOINT — report to user**

Present the candidate table + integration cost estimate. Per spec, any
integration projected >2 h needs user sign-off before building envs or
downloading weights. Wait for direction.

- [ ] **Step 3 (post-approval): integrate 1-2 winners as stage-4 backends**

Pattern to follow: SAM-3D's subprocess wiring (`run_object_sam3d` in
`real_perception.py` + `scripts/subprocess_entries/`) — new env, CLI entry
that reads frames/masks/depth/K from the run dir and writes
`poses[T,4,4] + mesh` npz; a `backend.object_pose: <method>` branch in
stage 4. Evaluate with `run_batch.py --arm <method>`, gate, commit or
record in T4_NOTES.md.

---

### Task 7: Campaign wrap-up

**Files:**
- Modify: `BEST_STRATEGY.md`, `compare/hot3d/LEADERBOARD.md`
- Modify: memory `hoi-recon-box-state.md`

- [ ] **Step 1: Final leaderboard render + BEST_STRATEGY section**

`$PY compare/hot3d/leaderboard.py render`. Add a "HOT3D improvement
campaign (2026-07)" section to BEST_STRATEGY.md: per-tier findings (win or
why not), final vs baseline table, updated next-build ranking.

- [ ] **Step 2: Commit, push, checkpoint report to user**

```bash
git add -A && git commit -m "loop: campaign wrap-up — final leaderboard + findings" && git push
```
Report: baseline → final numbers per clip, which tiers landed, what's next.
