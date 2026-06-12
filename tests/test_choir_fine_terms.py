# tests/test_choir_fine_terms.py
import torch
import pytest
from hoi_recon.choir_fine import terms_torch as T


def test_contact_loss_zero_when_anchors_coincide():
    hand_c = torch.zeros(1, 2, 3)                     # (T=1, Nc=2, 3)
    anchors = torch.zeros(1, 2, 4, 3)                 # K=4 anchors all at the hand vert
    weights = torch.full((1, 2, 4), 0.25)
    conf = torch.ones(1, 2)
    assert float(T.contact_loss(hand_c, anchors, weights, conf)) == 0.0


def test_contact_loss_is_weighted_distance():
    hand_c = torch.zeros(1, 1, 3)
    anchors = torch.zeros(1, 1, 2, 3)
    anchors[0, 0, 0, 2] = 0.1                         # anchor 0 is 0.1 away in z
    weights = torch.tensor([[[1.0, 0.0]]])            # all weight on anchor 0
    conf = torch.ones(1, 1)
    # loss = conf * (w0 * 0.1^2) / conf = 0.01
    assert float(T.contact_loss(hand_c, anchors, weights, conf)) == pytest.approx(0.01)


def test_contact_loss_confidence_normalizes():
    # two verts, one with confidence 0 -> ignored
    hand_c = torch.zeros(1, 2, 3)
    anchors = torch.zeros(1, 2, 1, 3)
    anchors[0, 0, 0, 0] = 0.2                         # vert 0 anchor 0.2 away
    anchors[0, 1, 0, 0] = 1.0                         # vert 1 far, but conf 0
    weights = torch.ones(1, 2, 1)
    conf = torch.tensor([[1.0, 0.0]])
    assert float(T.contact_loss(hand_c, anchors, weights, conf)) == pytest.approx(0.04)


def test_penetration_zero_outside():
    hand = torch.tensor([[[0.0, 0.0, 0.02]]])         # 2cm outside (above) the surface
    surf = torch.zeros(1, 1, 3)
    nrm = torch.tensor([[[0.0, 0.0, 1.0]]])
    assert float(T.penetration_loss(hand, surf, nrm)) == 0.0


def test_penetration_clamps_with_tolerance():
    hand = torch.tensor([[[0.0, 0.0, -0.01]]])        # 1cm inside
    surf = torch.zeros(1, 1, 3)
    nrm = torch.tensor([[[0.0, 0.0, 1.0]]])
    # signed = (surf-hand).n = 0.01 ; (0.01 - eps0.005).clamp(0,0.04) = 0.005
    assert float(T.penetration_loss(hand, surf, nrm)) == pytest.approx(0.005)


def test_penetration_normalizes_normal():
    hand = torch.tensor([[[0.0, 0.0, -0.01]]])
    surf = torch.zeros(1, 1, 3)
    nrm = torch.tensor([[[0.0, 0.0, 2.0]]])           # non-unit normal must not scale depth
    assert float(T.penetration_loss(hand, surf, nrm)) == pytest.approx(0.005)


def test_terms_are_differentiable():
    hand_c = torch.zeros(1, 1, 3, requires_grad=True)
    anchors = torch.ones(1, 1, 1, 3)
    loss = T.contact_loss(hand_c, anchors, torch.ones(1, 1, 1), torch.ones(1, 1))
    loss.backward()
    assert hand_c.grad is not None and torch.isfinite(hand_c.grad).all()
