# hoi_recon/choir_fine/anatomical.py
"""MANO anatomical constraint (CHOIR L^h_anat). Fingers flex about a single joint axis;
penalize the off-flexion rotation components — twist (about the bone axis) and splay
(abduction) — leaving bend (the flexion axis) free. Operates on MANO hand_pose in
axis-angle. The flexion axis is taken as component index 2 (MANO convention); penalizing
the other two components is a documented approximation of CHOIR's twist-splay-bend term."""
from __future__ import annotations

import torch

BEND_AXIS = 2          # axis-angle component that finger flexion lives on (MANO)


def anatomical_loss(hand_pose):
    """hand_pose: (...,15,3) or (...,45) axis-angle. Returns a scalar tensor."""
    aa = hand_pose.reshape(*hand_pose.shape[:-1], 15, 3) if hand_pose.shape[-1] == 45 \
        else hand_pose.reshape(-1, 15, 3) if hand_pose.dim() == 1 else hand_pose
    off = [i for i in range(3) if i != BEND_AXIS]      # twist + splay axes
    return (aa[..., off] ** 2).mean()
