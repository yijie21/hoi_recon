# hoi_recon/choir_fine/presets.py
"""Named energy-term weight presets for the CHOIR Stage-3 optimizer.

choir_faithful = CHOIR arXiv:2605.20992 §7.3 weights, VERBATIM and LOCKED (guarded by
tests/test_choir_fine_presets.py). combined_v2 = faithful + our improvement toggles.
Each term name is a key the optimizer's term registry looks up."""
from __future__ import annotations

import copy

# every energy term the Stage-3 optimizer can apply (weight 0 => inactive)
TERMS = [
    "contact", "template", "bridge", "gap", "patch",      # contact family
    "pen", "sil",                                          # penetration, object silhouette
    "anc_2d", "anc_anat", "anc_pose_h", "anc_pose_o",      # anchors
    "temp_pose_vel", "temp_obj_vel", "temp_wrist_anchor",  # temporal (velocity)
    "temp_hand_tr_vel", "temp_root_R_vel",
    "temp_pose_acc", "temp_hand_tr_acc", "temp_root_R_acc",  # temporal (acceleration)
    "hand_sil",                                            # our improvement (T3)
]

# CHOIR §7.3 verbatim. The contact-family stabilizers (template/bridge/gap/patch) are
# "annealed" in CHOIR; start them at 0 and let the optimizer schedule them.
CHOIR_FAITHFUL = {
    "contact": 1000.0, "template": 0.0, "bridge": 0.0, "gap": 0.0, "patch": 0.0,
    "pen": 500.0, "sil": 500.0,
    "anc_2d": 0.5, "anc_anat": 30.0, "anc_pose_h": 100.0, "anc_pose_o": 100.0,
    "temp_pose_vel": 500.0, "temp_obj_vel": 500.0, "temp_wrist_anchor": 200.0,
    "temp_hand_tr_vel": 200.0, "temp_root_R_vel": 500.0,
    "temp_pose_acc": 1000.0, "temp_hand_tr_acc": 5000.0, "temp_root_R_acc": 3000.0,
    "hand_sil": 0.0,
}
LRS_FAITHFUL = {"object": 3e-4, "finger": 5e-4, "wrist": 5e-5}
ITERS_FAITHFUL = 800

# contact-cache rebuild + anchor-build constants (CHOIR §7.3 / §7.2)
CONTACT_CACHE = {"dist_m": 0.05, "normal_deg": 60.0, "topk": 8, "softmax_sigma": 0.01}
ANCHOR_BUILD = {"n_surface": 10000, "knn": 50, "dist_m": 0.02, "normal_deg": 60.0}

# combined_v2 = faithful + our improvement toggles (each independently overridable)
COMBINED_V2 = {**copy.deepcopy(CHOIR_FAITHFUL), "hand_sil": 1.0}

_PRESETS = {"choir_faithful": CHOIR_FAITHFUL, "combined_v2": COMBINED_V2}


def get_preset(name: str) -> dict:
    """Return a deep copy of the named preset's weight dict. Raises KeyError if unknown."""
    if name not in _PRESETS:
        raise KeyError(f"unknown preset '{name}'; have {sorted(_PRESETS)}")
    return copy.deepcopy(_PRESETS[name])
