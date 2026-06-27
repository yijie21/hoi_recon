# EgoAERO reproduction — SP3: Online quality assessment (design)

Date: 2026-06-27
Status: approved (brainstorming) → ready for implementation plan
Method folder: `egoaero/` (self-contained method in the HOI-reconstruction workbench)

## 0. Context

Second sub-project of the full EgoAERO reproduction (paper at
`egoaero/egoaero.pdf_by_PaddleOCR-VL-1.6.md`). Build order is **SP1 → SP3 → SP2 → SP4**.

- **SP1 — Asset-free hand-object reconstruction** (Sec 2.1 / App A–C): ✅ done (stages 0–7,
  faithful App-C contact optimization, mock-runnable).
- **SP3 — Online quality assessment (Sec 3 / App E)** ← *this spec*. Depends on SP1 output.
- SP2 — Two-stage residual RL policy (App D). SP4 — EgoDex-R dataset + collection loop (App F).

### Operating principle (unchanged across sub-projects)

> Faithful where the paper specifies; principled documented defaults at every gap (logged in
> `egoaero/ASSUMPTIONS.md`); mock data where none exists; runnable end-to-end in `--mock`.

## 1. Goal of SP3

Reproduce the EgoAERO **online ego data quality assessment** (App E): score a reconstructed
clip on *bounded recoverability* — whether stable hand-object contact can be recovered through
small, local, interpretable corrections — and emit one of three collection decisions:
**accept**, **repairable_accept**, or **recapture**, with per-finger diagnostics and failure
attribution.

### Key reuse insight

App E frames the assessment as a **constrained projection** of the coarse trajectory
`x_0 = {H_t, O_t, M_O}` with bounded per-finger corrections `‖ΔH_t^f‖ ≤ δ_max` and the object
pose **fixed** (`O_t = O_t^0`). That bounded projection is *exactly* the operation SP1's
**stage6 adaptive contact optimization** already performs (object pose/mesh/MANO fixed; only
the replay hand is corrected within App-C bounds). Therefore:

- `x_0` (coarse) hand = **stage5_ego_comp** output (pre-contact-correction hand).
- `x*` (repaired) hand = **stage6_contact** output (post-correction hand).
- `ΔH_t^f` = per-finger displacement between the stage5 and stage6 hand.

SP3 **does not re-run any optimization** — it reads stage5 (before) and stage6 (after) and
scores them. This keeps the reproduction DRY and faithful (the assessment scores the same
repair the pipeline applies).

## 2. Architecture

Two units plus config/wiring.

```
egoaero/egoaero/
  quality.py                  NEW — pure App-E scoring functions (no I/O), unit-tested
  stages/stage8_quality.py    NEW — thin stage: stage5+stage6 -> quality verdict bundle + quality.json
  config.py                   MODIFY — add cfg.quality defaults block
  pipeline.py                 MODIFY — append stage8_quality to STAGES (9 stages, 0..8)
```

### 2.1 `quality.py` — pure scoring functions

Imports `signed_distance`, `active_window`, `fingertip_pad_idx` from the stage6/contact code
and `egoaero.core.hand` (intra-package; no sibling-method imports). All functions are pure
(arrays in → values out), deterministic, and individually unit-testable.

- `per_finger_gap(hand_verts, finger_idx, obj_world_pts, obj_normals, window) -> dict[str, np.ndarray]`
  For each finger in `FINGERS`, the median **distance** of its fingertip-pad vertices to the
  object surface, per active frame `t ∈ W`. Returns finger → array of length `|W|`.
  (App E `g_t^{f}`; called once with the coarse hand → `g^before`, once with the repaired hand
  → `g^after`.) Distance uses `|signed_distance|` to the object surface.

- `per_finger_delta(coarse_verts, repaired_verts, finger_idx, window) -> dict[str, np.ndarray]`
  `‖ΔH_t^f‖` = mean Euclidean displacement of finger `f`'s fingertip-pad vertices between the
  coarse (stage5) and repaired (stage6) hand, per active frame. finger → array length `|W|`.

- `recoverability(gap_after, delta, eps_g, eps_delta) -> dict[str, float]`
  `Q_rec^f = (1/|W|) Σ_{t∈W} 1[ g_t^{f,after} < ε_g  ∧  ‖ΔH_t^f‖ < ε_Δ ]`. finger → fraction in
  `[0,1]`.

- `repair_budget(delta, delta_max) -> float`
  `B_repair = median_{t,f} ‖ΔH_t^f‖ / δ_max` over all active frames and fingers.

- `residual_after(pen_after_mm, gap_after, gap_ref_mm) -> float`
  `R_after` = normalized remaining residual = `(pen_after_mm / pen_ref) + (median_f median_t g^after / gap_ref)`.
  Combines residual penetration (from stage6 meta) and residual contact gap. `pen_ref`/`gap_ref`
  are documented normalizers so `R_after` is dimensionless and `Q` is well-scaled.

- `unresolved_ratio(gap_after, delta, object_moving, eps_g, eps_delta) -> float`
  `U_unresolved` heuristic: fraction of active frames in which the **object is moving** yet **no**
  finger achieves recoverable contact (`g^after < ε_g ∧ ‖ΔH^f‖ < ε_Δ`) — a proxy for failures
  not explainable by local correction (severe tracking/articulation error). `object_moving[t]`
  is a boolean per active frame (object translation speed above a documented threshold). The
  exact heuristic is a documented default (App E names the quantity but not its formula).

- `quality_score(R_after, B_repair, U_unresolved, alpha, beta, gamma) -> float`
  `Q = exp(−α·R_after − β·B_repair − γ·U_unresolved) ∈ (0,1]`.

- `decision(Q, per_finger_Q_rec, q_accept, q_repairable) -> (str, dict)`
  Returns one of `"accept"`, `"repairable_accept"`, `"recapture"` plus a `failure_attribution`
  dict (e.g. low-recoverability fingers, dominant residual term). Rule (documented default):
  `Q ≥ q_accept` → accept; `q_repairable ≤ Q < q_accept` → repairable_accept; else recapture.

### 2.2 `stage8_quality.py`

`NAME = "stage8_quality"; INDEX = 8`. `run(ctx) -> Bundle`:
- Loads **stage5_ego_comp** (coarse `hand_verts_t`/`hand_joints_t`, `obj_poses_t`, `obj_verts`,
  `obj_faces`, meta `finger_idx`, `stage_labels`) and **stage6_contact** (repaired
  `hand_verts_t`, meta `pen_after_mm`, `gap_after_mm`).
- Builds per-frame object world points+normals (reuse `stage6._obj_world`), the active window
  (`active_window(stage_labels)`), rehydrates `finger_idx` (float `z_norm`, int finger arrays).
- Computes `g^before` (coarse), `g^after` (repaired), `ΔH^f`, `Q_rec^f`, `B_repair`, `R_after`,
  `U_unresolved`, `Q`, and the `decision`.
- Prints a one-block verdict (decision, Q, per-finger Q_rec, B_repair).
- Returns a Bundle with `meta["quality"] = {Q, decision, per_finger:{gap_before_mm, gap_after_mm,
  Q_rec}, B_repair, R_after, U_unresolved, failure_attribution}` and writes the same dict to
  `<run>/quality.json`.
- No real-backend branch (pure geometry/metrics — runs identically in mock and real).

### 2.3 Config (`cfg.quality`, documented defaults)

```
quality:
  eps_g_m: 0.004            # contact-gap recoverability threshold (4 mm) — DOCUMENTED
  eps_delta_m: 0.012        # per-finger correction budget threshold (12 mm) — DOCUMENTED
  delta_max_m: 0.015        # = contact.max_finger_disp_m (budget normalizer) — DOCUMENTED
  alpha: 1.0                # R_after weight — DOCUMENTED
  beta: 0.5                 # B_repair weight — DOCUMENTED
  gamma: 1.0                # U_unresolved weight — DOCUMENTED
  pen_ref_mm: 50000.0       # R_after penetration normalizer (mock-scale) — DOCUMENTED
  gap_ref_mm: 40.0          # R_after gap normalizer — DOCUMENTED
  obj_move_thresh_mps: 0.01 # object-moving threshold for U_unresolved — DOCUMENTED
  q_accept: 0.6             # Q >= -> accept — DOCUMENTED
  q_repairable: 0.3         # q_repairable <= Q < q_accept -> repairable_accept — DOCUMENTED
```

All values are documented defaults (App E specifies none); each logged in `ASSUMPTIONS.md`.

## 3. Data flow

```
stage5_ego_comp (coarse) ┐
                         ├─► stage8_quality.run ─► Bundle.meta["quality"] + <run>/quality.json
stage6_contact (repaired)┘
```

The 4D-HOI **contract output is unchanged** — quality is a supplementary diagnostic, written
alongside (`quality.json`), not inside, the contract dir. `contract.validate` is untouched.

## 4. Faithfulness map

- `Q_rec^f`, `B_repair`, `Q = exp(−αR−βB−γU)`, gap before/after, the three decisions: **faithful**
  to App-E equations.
- `R_after` normalization, `U_unresolved` heuristic, `ε_g/ε_Δ/δ_max/α/β/γ`, and the decision
  thresholds: **documented defaults** (App E names them but gives no values/formulas). All in
  `ASSUMPTIONS.md`.

## 5. Testing

- **Unit (`quality.py`)**: `recoverability` counts the indicator correctly (construct gaps/deltas
  with a known number of passing frames); `repair_budget` = median/δ_max; `quality_score` is
  strictly decreasing in each of `R_after`, `B_repair`, `U_unresolved` and lies in `(0,1]`;
  `decision` returns each of the three labels at threshold boundaries; `per_finger_gap`/`_delta`
  return the right shapes and a known displacement on a constructed hand.
- **Stage (`stage8_quality`)**: mock run after stages 0–6 produces `decision ∈ {accept,
  repairable_accept, recapture}`, `Q ∈ (0,1]`, per-finger `Q_rec ∈ [0,1]`, and writes
  `quality.json`.
- **Smoke**: extend the end-to-end mock smoke test to run through stage8 and assert the quality
  verdict is present and well-formed.
- Tests CWD-safe, no weights/data.

## 6. Deliverables

1. `egoaero/egoaero/quality.py` (pure App-E scorers) + `stages/stage8_quality.py`.
2. `cfg.quality` defaults in `config.py`; `stage8_quality` wired into `STAGES`.
3. `<run>/quality.json` + `meta["quality"]` verdict; printed verdict block.
4. `ASSUMPTIONS.md` entries for every default + the `U_unresolved` heuristic.
5. Unit + stage + smoke tests passing.
6. `egoaero/README.md` updated (stage list 0–8; quality-assessment section).

## 7. Deferred

SP2 (RL policy, App D) and SP4 (EgoDex-R dataset + collection loop, App F). SP4's collection loop
will consume SP3's `decision` (accept/repair/recapture) — SP3 provides the scorer; SP4 wraps it
in the keep/repair/recapture loop.
