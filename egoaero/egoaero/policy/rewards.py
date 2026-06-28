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


def r_obj(p_o, R_o, podot, p_ref, R_ref, podot_ref, mu_p, mu_R, mu_v):
    """Object pose reward (exponential of negative weighted squared errors).

    Args:
        p_o, R_o, podot: current position (3,), rotation (3x3), velocity (3,)
        p_ref, R_ref, podot_ref: desired position (3,), rotation (3x3), velocity (3,)
        mu_p, mu_R, mu_v: weight on position error, rotation error, velocity error

    Returns:
        float: reward in [0, 1]
    """
    dp = np.sum((np.asarray(p_o) - np.asarray(p_ref)) ** 2)
    dR = geodesic_rad(R_o, R_ref) ** 2
    dv = np.sum((np.asarray(podot) - np.asarray(podot_ref)) ** 2)
    return float(np.exp(-mu_p * dp - mu_R * dR - mu_v * dv))


def r_contact(dists, forces, active, mu_d, mu_F):
    """Contact reward (mean exponential over active fingers).

    Returns 0.0 if active finger set is empty, otherwise computes mean reward
    over active fingers as exp(-mu_d * d^2) * (1 - exp(-mu_F * |F|)).

    Args:
        dists: per-finger distances (N,)
        forces: per-finger forces (N,)
        active: list/array of active finger indices
        mu_d: weight on distance decay
        mu_F: weight on force activation

    Returns:
        float: mean contact reward, or 0.0 if no active fingers
    """
    active = list(active)
    if not active:
        return 0.0
    d = np.asarray(dists, float)
    f = np.asarray(forces, float)
    vals = [np.exp(-mu_d * d[i] ** 2) * (1.0 - np.exp(-mu_F * abs(f[i]))) for i in active]
    return float(np.mean(vals))


def r_res(delta_a, mu_delta):
    """Residual action reward (exponential of negative weighted squared action changes).

    Args:
        delta_a: action residuals (M,)
        mu_delta: weight on action change penalty

    Returns:
        float: residual action reward
    """
    return float(np.exp(-mu_delta * np.sum(np.asarray(delta_a) ** 2)))


def r_stage2(r1, r_obj_v, r_contact_v, r_res_v, eta_I, eta_o, eta_c, eta_delta):
    """Stage-II reward: weighted sum of Stage-I and new contact/residual terms.

    Args:
        r1: Stage-I reward
        r_obj_v: object pose reward
        r_contact_v: contact reward
        r_res_v: residual action reward
        eta_I, eta_o, eta_c, eta_delta: weights for Stage-I, object, contact, residual

    Returns:
        float: weighted sum of Stage-II components
    """
    return float(eta_I * r1 + eta_o * r_obj_v + eta_c * r_contact_v + eta_delta * r_res_v)
