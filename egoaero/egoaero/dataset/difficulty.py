"""Heuristic difficulty score (1..5) for an EgoDex-R sequence. The paper uses an
MLLM-based evaluator (App F); this is a documented deterministic substitute combining
occlusion, object motion, residual quality, and (inversely) contact richness."""
from __future__ import annotations


def difficulty_score(quality_report, recon_summary, weights):
    """Score sequence difficulty from 1 (easy) to 5 (hard).

    Args:
        quality_report: Dict with 'R_after' (residual quality, ~3.0 scale) and
                       'U_unresolved' (unresolved ratio, [0,1]).
        recon_summary: Dict with 'occlusion' ([0,1]), 'obj_motion_m' (meters),
                      and 'contact_richness' ([0,1]).
        weights: Dict with 'w_occlusion', 'w_motion', 'w_residual', 'w_contact'
                (all relative weights, typically 1.0 each).

    Returns:
        int in range [1, 5], where 1 is easiest and 5 is hardest.
    """
    occ = float(recon_summary.get("occlusion", 0.0))                 # [0,1]
    motion = min(float(recon_summary.get("obj_motion_m", 0.0)) / 0.5, 1.0)  # normalize ~0.5 m
    contact = float(recon_summary.get("contact_richness", 0.0))      # [0,1]
    resid = min(float(quality_report.get("R_after", 0.0)) / 3.0, 1.0)       # normalize ~3.0
    unresolved = float(quality_report.get("U_unresolved", 0.0))      # [0,1]

    hard = (weights["w_occlusion"] * occ
            + weights["w_motion"] * motion
            + 0.5 * weights["w_residual"] * (resid + unresolved)
            - weights["w_contact"] * contact)

    # max possible "hard" (contact=0): w_occ + w_motion + w_residual ; normalize to [0,1]
    denom = weights["w_occlusion"] + weights["w_motion"] + weights["w_residual"]
    frac = max(0.0, min(hard / denom, 1.0)) if denom > 0 else 0.0
    return int(round(1 + 4 * frac))                                  # 1..5
