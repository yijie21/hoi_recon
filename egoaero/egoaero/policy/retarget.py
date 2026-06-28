"""Kinematic retargeting via damped-least-squares IK. FK is injected so the solver
is independent of MuJoCo (unit-tested with a toy chain). In the sim, fk_fn wraps the
Shadow-Hand forward kinematics (see hand_model.fk_fingertips)."""
from __future__ import annotations
import numpy as np


def _jacobian(fk_fn, q, eps=1e-5):
    """Compute Jacobian via finite differences.

    Args:
        fk_fn: FK callable, q[n_q] -> [K,3] fingertip positions
        q: Configuration vector [n_q]
        eps: Finite difference step

    Returns:
        J: Jacobian [3K, n_q]
        cur: Current end-effector positions (flattened) [3K]
    """
    base = fk_fn(q).reshape(-1)        # [3K]
    J = np.zeros((base.size, q.size))
    for j in range(q.size):
        dq = q.copy()
        dq[j] += eps
        J[:, j] = (fk_fn(dq).reshape(-1) - base) / eps
    return J, base


def ik_step(q, fk_fn, targets, damping):
    """One damped-least-squares IK iteration.

    DLS formula: dq = J^T (J J^T + damping^2 I)^-1 err

    Args:
        q: Current configuration [n_q]
        fk_fn: FK callable, q[n_q] -> [K,3] fingertip positions
        targets: Target positions [K,3]
        damping: Damping coefficient (regularization strength)

    Returns:
        q_new: Updated configuration [n_q]
    """
    J, cur = _jacobian(fk_fn, q)
    err = np.asarray(targets).reshape(-1) - cur
    JT = J.T
    dq = JT @ np.linalg.solve(J @ JT + (damping ** 2) * np.eye(J.shape[0]), err)
    return q + dq


def solve_ik(q0, fk_fn, targets, damping, iters):
    """Solve IK by iterating ik_step until convergence.

    Args:
        q0: Initial configuration [n_q]
        fk_fn: FK callable, q[n_q] -> [K,3] fingertip positions
        targets: Target positions [K,3]
        damping: Damping coefficient
        iters: Number of iterations

    Returns:
        q: Solved configuration [n_q]
    """
    q = np.asarray(q0, float).copy()
    for _ in range(int(iters)):
        q = ik_step(q, fk_fn, targets, damping)
    return q


def retarget_sequence(target_seq, fk_fn, q0, n_q, damping, iters):
    """Retarget a sequence of target poses by warm-starting each frame.

    Args:
        target_seq: Sequence of target positions [T, K, 3]
        fk_fn: FK callable, q[n_q] -> [K,3] fingertip positions
        q0: Initial configuration [n_q]
        n_q: Number of joints
        damping: Damping coefficient
        iters: Iterations per frame

    Returns:
        q_seq: Sequence of solved configurations [T, n_q]
    """
    q = np.asarray(q0, float).copy()
    out = np.zeros((len(target_seq), n_q))
    for t in range(len(target_seq)):
        q = solve_ik(q, fk_fn, target_seq[t], damping, iters)   # warm-start from prev frame
        out[t] = q
    return out
