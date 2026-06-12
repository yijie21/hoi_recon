# tests/test_choir_fine_registry.py
import torch
import pytest
from hoi_recon.choir_fine import registry


def test_assemble_sums_weighted_active_terms():
    weights = {"contact": 10.0, "pen": 500.0, "sil": 0.0}
    values = {"contact": torch.tensor(2.0), "pen": torch.tensor(0.1),
              "sil": torch.tensor(9.9)}
    total = registry.assemble_energy(weights, values)
    # 10*2 + 500*0.1 + (sil weight 0 -> skipped) = 20 + 50 = 70
    assert float(total) == pytest.approx(70.0)


def test_assemble_skips_zero_weight_terms_entirely():
    """A zero-weight term must not contribute even if its value is huge/NaN-prone."""
    weights = {"contact": 1.0, "sil": 0.0}
    values = {"contact": torch.tensor(1.0), "sil": torch.tensor(float("inf"))}
    total = registry.assemble_energy(weights, values)
    assert float(total) == pytest.approx(1.0)          # inf*0 skipped, not nan


def test_assemble_raises_on_value_without_weight():
    with pytest.raises(KeyError):
        registry.assemble_energy({"contact": 1.0}, {"contact": torch.tensor(1.0),
                                                    "mystery": torch.tensor(1.0)})


def test_assemble_is_differentiable():
    x = torch.tensor(3.0, requires_grad=True)
    total = registry.assemble_energy({"a": 2.0}, {"a": x * x})
    total.backward()
    assert float(x.grad) == pytest.approx(12.0)        # d(2*x^2)/dx = 4x = 12


def test_assemble_returns_zero_tensor_when_no_active_terms():
    out = registry.assemble_energy({"a": 0.0}, {"a": torch.tensor(5.0, requires_grad=True)})
    assert torch.is_tensor(out) and float(out) == 0.0
