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
