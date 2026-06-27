# hoi_recon/choir_fine/registry.py
"""Weighted energy-term registry: sum the active (non-zero-weight) terms of an optimization
step. The optimizer computes each term's scalar value into a dict, and this assembles the
total loss using a preset's weight dict (hoi_recon.choir_fine.presets). Zero-weight terms
are skipped entirely (their value never enters the graph), so an inactive term cannot inject
NaN/inf or waste a backward pass."""
from __future__ import annotations

import torch


def assemble_energy(weights, values) -> torch.Tensor:
    """weights: {term_name: float}. values: {term_name: scalar tensor}. Returns the summed
    weighted total (a scalar tensor, or 0.0 if no active terms). Raises KeyError if a value
    has no corresponding weight (guards against typos / unregistered terms)."""
    missing = [k for k in values if k not in weights]
    if missing:
        raise KeyError(f"term value(s) without a weight: {missing}")
    total = None
    for name, val in values.items():
        w = weights[name]
        if w != 0:                       # exact-zero test: skip inactive terms entirely
            total = w * val if total is None else total + w * val
    if total is None:                    # no active term -> a differentiable zero tensor
        ref = next(iter(values.values()), None)
        if ref is not None:
            return torch.zeros((), dtype=ref.dtype, device=ref.device)
        return torch.zeros(())
    return total
