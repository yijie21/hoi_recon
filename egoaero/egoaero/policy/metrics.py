"""App-H evaluation metrics for EgoAERO policy rollouts. Pure numpy. Units: deg, cm."""
from __future__ import annotations
import numpy as np
from .rewards import geodesic_rad


def object_rotation_error(R_seq, R_ref_seq):
    """Geodesic distance between rotation sequences in degrees.

    Args:
        R_seq: (N, 3, 3) rotation matrix sequence
        R_ref_seq: (N, 3, 3) reference rotation matrix sequence

    Returns:
        Mean geodesic error in degrees
    """
    R_seq = np.asarray(R_seq)
    R_ref_seq = np.asarray(R_ref_seq)
    degs = [np.degrees(geodesic_rad(R_seq[t], R_ref_seq[t])) for t in range(len(R_seq))]
    return float(np.mean(degs))


def object_translation_error(p_seq, p_ref_seq):
    """Mean translation error in centimeters.

    Args:
        p_seq: (N, 3) position sequence in meters
        p_ref_seq: (N, 3) reference position sequence in meters

    Returns:
        Mean L2 error in cm
    """
    d = np.linalg.norm(np.asarray(p_seq) - np.asarray(p_ref_seq), axis=-1)
    return float(np.mean(d) * 100.0)   # m -> cm


def mean_joint_error(xj_seq, xj_ref_seq):
    """Mean joint position error in centimeters.

    Args:
        xj_seq: (..., 3) joint positions in meters
        xj_ref_seq: (..., 3) reference joint positions in meters

    Returns:
        Mean L2 error in cm
    """
    d = np.linalg.norm(np.asarray(xj_seq) - np.asarray(xj_ref_seq), axis=-1)
    return float(np.mean(d) * 100.0)


def mean_fingertip_error(xf_seq, xf_ref_seq):
    """Mean fingertip position error in centimeters.

    Args:
        xf_seq: (..., 3) fingertip positions in meters
        xf_ref_seq: (..., 3) reference fingertip positions in meters

    Returns:
        Mean L2 error in cm
    """
    d = np.linalg.norm(np.asarray(xf_seq) - np.asarray(xf_ref_seq), axis=-1)
    return float(np.mean(d) * 100.0)


def success(Er, Et, Ej, Eft, tau_r=30.0, tau_t=3.0, tau_j=8.0, tau_ft=6.0):
    """Check if all error metrics are below thresholds.

    Args:
        Er: rotation error in degrees
        Et: translation error in cm
        Ej: joint error in cm
        Eft: fingertip error in cm
        tau_r: rotation threshold in degrees (default 30)
        tau_t: translation threshold in cm (default 3)
        tau_j: joint threshold in cm (default 8)
        tau_ft: fingertip threshold in cm (default 6)

    Returns:
        True if all errors are below thresholds, False otherwise
    """
    return bool(Er < tau_r and Et < tau_t and Ej < tau_j and Eft < tau_ft)


def success_rate(rows, taus=None):
    """Compute success rate from rows of (Er, Et, Ej, Eft) errors.

    Args:
        rows: list of (Er, Et, Ej, Eft) tuples
        taus: (tau_r, tau_t, tau_j, tau_ft) thresholds (default (30, 3, 8, 6))

    Returns:
        Fraction of successful rows (0.0 to 1.0)
    """
    taus = taus or (30.0, 3.0, 8.0, 6.0)
    if not rows:
        return 0.0
    ok = [success(*row, *taus) for row in rows]
    return float(np.mean(ok))
