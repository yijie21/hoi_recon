"""App-D reward terms for EgoAERO two-stage residual RL. Pure numpy, no heavy deps."""
from __future__ import annotations
import numpy as np


def geodesic_rad(Ra, Rb):
    """Rotation geodesic distance in radians.

    Args:
        Ra, Rb: 3x3 rotation matrices

    Returns:
        float: geodesic distance in radians, clipped to valid range
    """
    Rrel = np.asarray(Ra).T @ np.asarray(Rb)
    cos = (np.trace(Rrel) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def r_wrist(p, R, pdot, p_h, R_h, pdot_h, lam_p, lam_R, lam_v):
    """Wrist tracking reward (exponential of negative weighted squared errors).

    Args:
        p, R, pdot: current position (3,), rotation (3x3), velocity (3,)
        p_h, R_h, pdot_h: desired position (3,), rotation (3x3), velocity (3,)
        lam_p, lam_R, lam_v: weight on position error, rotation error, velocity error

    Returns:
        float: reward in [0, 1]
    """
    dp = np.sum((np.asarray(p) - np.asarray(p_h)) ** 2)
    dR = geodesic_rad(R, R_h) ** 2
    dv = np.sum((np.asarray(pdot) - np.asarray(pdot_h)) ** 2)
    return float(np.exp(-lam_p * dp - lam_R * dR - lam_v * dv))


def r_finger(x_kpts, x_kpts_h, lam_k):
    """Finger keypoint tracking reward (mean of exponentials).

    Args:
        x_kpts: current keypoints (K, 3)
        x_kpts_h: desired keypoints (K, 3)
        lam_k: weight on keypoint error

    Returns:
        float: mean reward across all keypoints
    """
    x = np.asarray(x_kpts)
    xh = np.asarray(x_kpts_h)
    d2 = np.sum((x - xh) ** 2, axis=1)
    return float(np.mean(np.exp(-lam_k * d2)))


def r_smooth(a, a_prev, torque, qdot, lam_a, lam_tau):
    """Smoothness reward (acceleration and torque penalty).

    Args:
        a, a_prev: current and previous acceleration (4,)
        torque: joint torques (4,)
        qdot: joint velocities (4,)
        lam_a: weight on acceleration change
        lam_tau: weight on torque power

    Returns:
        float: smoothness reward
    """
    da = np.sum((np.asarray(a) - np.asarray(a_prev)) ** 2)
    power = np.sum(np.abs(np.asarray(torque) * np.asarray(qdot)))
    return float(np.exp(-lam_a * da - lam_tau * power))


def r_stage1(r_wrist_v, r_finger_v, r_smooth_v, w_w, w_f, w_s):
    """Stage-I reward: weighted sum of wrist, finger, and smoothness terms.

    Args:
        r_wrist_v: wrist tracking reward
        r_finger_v: finger keypoint reward
        r_smooth_v: smoothness reward
        w_w, w_f, w_s: weights for wrist, finger, smoothness

    Returns:
        float: weighted sum of rewards
    """
    return float(w_w * r_wrist_v + w_f * r_finger_v + w_s * r_smooth_v)
