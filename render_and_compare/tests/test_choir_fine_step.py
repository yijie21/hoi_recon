# tests/test_choir_fine_step.py
import torch
import pytest
from hoi_recon.choir_fine import step, presets, registry, terms_torch as T


def _state(Tf=4):
    """Minimal synthetic optimization state (tiny tensors)."""
    return {
        "hand_c": torch.zeros(Tf, 3, 3), "anchors": torch.zeros(Tf, 3, 2, 3),
        "anc_w": torch.full((Tf, 3, 2), 0.5), "conf": torch.ones(Tf, 3),
        "pen_hand": torch.zeros(Tf, 5, 3), "pen_surf": torch.zeros(Tf, 5, 3),
        "pen_normal": torch.ones(Tf, 5, 3),
        "joints_cam": torch.tensor([[[0.0, 0.0, 1.0]]]).repeat(Tf, 21, 1),
        "kp2d": torch.zeros(Tf, 21, 2), "K": torch.eye(3), "kp_valid": torch.ones(Tf),
        "mano_pose": torch.zeros(Tf, 15, 3),
        "hand_verts": torch.zeros(Tf, 8, 3), "hand_verts_init": torch.zeros(Tf, 8, 3),
        "o_t": torch.zeros(Tf, 3), "o_t0": torch.zeros(Tf, 3),
        "o_r6": torch.zeros(Tf, 6), "o_r60": torch.zeros(Tf, 6),
        "p6": torch.zeros(Tf, 15, 6), "transl": torch.zeros(Tf, 3),
        "g6": torch.zeros(Tf, 6),
        "wrist": torch.zeros(Tf, 3), "wrist_init": torch.zeros(Tf, 3),
    }


def test_returns_all_geometric_term_keys():
    v = step.compute_geometric_terms(_state())
    expected = {"contact", "pen", "anc_2d", "anc_anat", "anc_pose_h", "anc_pose_o",
                "temp_pose_vel", "temp_obj_vel", "temp_wrist_anchor", "temp_hand_tr_vel",
                "temp_root_R_vel", "temp_pose_acc", "temp_hand_tr_acc", "temp_root_R_acc"}
    assert set(v) == expected
    for name, val in v.items():
        assert torch.is_tensor(val) and val.dim() == 0, name


def test_values_match_underlying_term_functions():
    s = _state()
    s["anchors"][:, :, 0, 2] = 0.1                       # anchor 0 is 0.1m away in z
    v = step.compute_geometric_terms(s)
    ref = T.contact_loss(s["hand_c"], s["anchors"], s["anc_w"], s["conf"])
    assert float(v["contact"]) == pytest.approx(float(ref))


def test_assembles_with_a_real_preset():
    """The geometric dict + zeroed render terms must assemble against a real preset
    (every value key has a weight) and be differentiable."""
    s = _state()
    s["o_t"] = s["o_t"].clone().requires_grad_(True)
    v = step.compute_geometric_terms(s)
    v_full = {**v, "sil": torch.zeros(()), "hand_sil": torch.zeros(()),
              "template": torch.zeros(()), "bridge": torch.zeros(()),
              "gap": torch.zeros(()), "patch": torch.zeros(())}
    total = registry.assemble_energy(presets.get_preset("choir_faithful"), v_full)
    assert torch.is_tensor(total)
    total.backward()                                     # must not raise
