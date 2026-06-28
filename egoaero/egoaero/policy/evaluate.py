"""Rollout a trained two-stage policy and compute App-H metrics; ablation harness.

Rollout
-------
``rollout(task, pi_I, pi_R, seed=0)`` runs a single StageIIEnv episode with a
frozen Stage-I policy ``pi_I`` and a residual Stage-II policy ``pi_R``.  Either
may be ``None`` (→ zero action).  Returns a dict with:

  - ``obj_pos``    : (T, 3) object centroid positions
  - ``obj_R``      : (T, 3, 3) object rotation matrices  (converted from the env's
                     quat via mujoco's ``mju_quat2Mat``)
  - ``fingertips`` : (T, 5, 3) fingertip positions in world frame

Evaluate
--------
``evaluate(task, pi_I, pi_R, seeds=(0,1))`` computes App-H metrics averaged over
the given seeds:

  - ``Er``  : object rotation error  (degrees, geodesic)
  - ``Et``  : object translation error (cm)
  - ``Ej``  : fingertip error used as the joint-position metric (cm)
              [see ASSUMPTIONS.md — full per-joint robot↔human correspondence
               is not defined for the substitute hand; fingertip keypoints are used]
  - ``Eft`` : same fingertip error (cm)
  - ``SR``  : success rate in [0, 1]

Ablation
--------
``ablation(run_dir, hand_xml, policy_cfg, budget="smoke")`` runs three conditions:

  - ``full``          : train + evaluate with contact-optimised reconstruction
  - ``only_hand``     : evaluate with pi_R=None (Stage-I only, no residual)
  - ``wo_contact_opt``: placeholder — reuses ``full`` unless the caller provides a
                        separate ``run_dir`` whose reconstruction skipped stage-6
                        contact optimisation (see ASSUMPTIONS.md).
"""
from __future__ import annotations

import logging

import numpy as np

from . import metrics as M

log = logging.getLogger(__name__)


def rollout(task, pi_I, pi_R, seed: int = 0) -> dict:
    """Run a single StageIIEnv episode; return trajectory arrays.

    Parameters
    ----------
    task : egoaero.policy.task.Task
    pi_I : SB3-policy-like or None
        Frozen Stage-I policy.  ``None`` → zero base action.
    pi_R : SB3-policy-like or None
        Stage-II residual policy.  ``None`` → zero residual.
    seed : int

    Returns
    -------
    dict with keys ``obj_pos`` (T,3), ``obj_R`` (T,3,3), ``fingertips`` (T,5,3).
    """
    import mujoco as mj
    from .env import StageIIEnv

    # Build frozen pi_I callable for StageIIEnv
    if pi_I is not None:
        def _pi_I(obs):
            act, _ = pi_I.predict(obs, deterministic=True)
            return act
    else:
        def _pi_I(obs):  # noqa: E306
            return np.zeros(len(task.finger_act_ids), dtype=np.float32)

    env = StageIIEnv(task, _pi_I)
    obs, _ = env.reset(seed=seed)

    op: list[np.ndarray] = []
    oR: list[np.ndarray] = []
    ft: list[np.ndarray] = []

    done = False
    while not done:
        if pi_R is not None:
            a = pi_R.predict(obs, deterministic=True)[0]
        else:
            a = np.zeros(env.action_space.shape, dtype=np.float32)

        obs, _, term, trunc, _ = env.step(a)

        # _obj_pose() → (pos[3], quat[4])
        pos, quat = env._obj_pose()
        op.append(pos.copy())

        # Convert quaternion → 3×3 rotation matrix
        Rm = np.zeros(9, dtype=np.float64)
        mj.mju_quat2Mat(Rm, quat)
        oR.append(Rm.reshape(3, 3).copy())

        ft.append(env._fingertips().copy())

        done = bool(term or trunc)

    return {
        "obj_pos":   np.array(op,    dtype=np.float64),   # (T, 3)
        "obj_R":     np.array(oR,    dtype=np.float64),   # (T, 3, 3)
        "fingertips": np.array(ft,   dtype=np.float64),   # (T, 5, 3)
    }


def evaluate(task, pi_I, pi_R, seeds=(0, 1)) -> dict:
    """Compute App-H metrics over multiple rollout seeds.

    Parameters
    ----------
    task : egoaero.policy.task.Task
    pi_I, pi_R : SB3-policy-like or None
    seeds : sequence of int

    Returns
    -------
    dict with keys Er, Et, Ej, Eft (floats) and SR (float in [0,1]).
    """
    ref = task.ref
    T_ref = int(ref["T"])

    rows: list[tuple[float, float, float, float]] = []
    for s in seeds:
        rl = rollout(task, pi_I, pi_R, seed=s)
        n = min(T_ref, len(rl["obj_pos"]))

        Er = M.object_rotation_error(rl["obj_R"][:n], ref["obj_R"][:n])
        Et = M.object_translation_error(rl["obj_pos"][:n], ref["obj_pos"][:n])
        # Ej and Eft both use fingertip error — see ASSUMPTIONS.md
        Ej = M.mean_fingertip_error(rl["fingertips"][:n], ref["fingertips_h"][:n])
        Eft = Ej
        rows.append((Er, Et, Ej, Eft))

    arr = np.array(rows, dtype=np.float64)
    return {
        "Er":  float(arr[:, 0].mean()),
        "Et":  float(arr[:, 1].mean()),
        "Ej":  float(arr[:, 2].mean()),
        "Eft": float(arr[:, 3].mean()),
        "SR":  M.success_rate([tuple(r) for r in rows]),
    }


def ablation(run_dir: str, hand_xml: str, policy_cfg: dict, budget: str = "smoke") -> dict:
    """Train and evaluate under three ablation conditions.

    Conditions
    ----------
    full
        Contact-optimised reconstruction (the contract as-is).
    only_hand
        Stage-I policy alone; residual pi_R is set to None.
    wo_contact_opt
        Documented placeholder: uses the *same* ``run_dir`` (and therefore the
        same contact-optimised reconstruction) unless the caller passes a separate
        ``run_dir`` that was produced with stage-6 disabled.  Logged at WARNING.
        See ASSUMPTIONS.md.

    Parameters
    ----------
    run_dir : str
        Path to an SP1 reconstruction run directory.
    hand_xml : str
        Path to the Shadow Hand XML asset.
    policy_cfg : dict
        Policy config dict (e.g. from ``configs/policy.yaml``).
    budget : str
        ``"smoke"`` (default, ~seconds) or ``"full"`` (production).

    Returns
    -------
    dict with keys ``full``, ``only_hand``, ``wo_contact_opt``.
    """
    from .task import build_task
    from .train import train_two_stage

    out: dict[str, dict] = {}

    # --- full: contact-optimised run ---
    task = build_task(run_dir, hand_xml, policy_cfg)
    pi_I, pi_R = train_two_stage(task, policy_cfg, budget=budget)
    out["full"] = evaluate(task, pi_I, pi_R, seeds=(0,))

    # --- only_hand: Stage-I only (no residual) ---
    out["only_hand"] = evaluate(task, pi_I, None, seeds=(0,))

    # --- wo_contact_opt: placeholder (reuses full) ---
    log.warning(
        "ablation wo_contact_opt: no separate non-contact-opt run_dir provided; "
        "reusing 'full' result.  Pass a run_dir produced with --stages 0-5,7 to "
        "obtain a real comparison (see ASSUMPTIONS.md)."
    )
    out["wo_contact_opt"] = dict(out["full"])

    return out
