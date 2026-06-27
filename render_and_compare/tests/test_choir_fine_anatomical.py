import torch
from hoi_recon.choir_fine import anatomical


def test_pure_bend_is_low():
    pose = torch.zeros(1, 15, 3)
    pose[..., 2] = 0.8                          # bend axis only
    assert float(anatomical.anatomical_loss(pose)) < 1e-6


def test_twist_is_penalized():
    pose = torch.zeros(1, 15, 3)
    pose[..., 0] = 0.8                          # twist about bone axis
    assert float(anatomical.anatomical_loss(pose)) > 0.1


def test_splay_is_penalized():
    pose = torch.zeros(1, 15, 3)
    pose[..., 1] = 0.8                          # splay / abduction
    assert float(anatomical.anatomical_loss(pose)) > 0.1


def test_accepts_flat_45_shape():
    pose = torch.zeros(45)                      # (15*3,) flat is accepted
    pose[2::3] = 0.5                            # bend components -> low
    assert float(anatomical.anatomical_loss(pose)) < 1e-6


def test_loss_is_scalar_and_differentiable():
    pose = torch.zeros(1, 15, 3, requires_grad=True)
    loss = anatomical.anatomical_loss(pose + torch.tensor([0.5, 0.0, 0.0]))  # twist offset
    assert loss.dim() == 0                     # scalar
    loss.backward()
    assert pose.grad is not None and torch.isfinite(pose.grad).all()
