"""Bridge flow-matching: refine a coarse trajectory x0 toward the GT x1.

Interpolant (in NORMALIZED state space):
    x_t = (1-t) * (x0 + sigma0 * eps) + t * x1,   eps ~ N(0,I)
i.e. a Gaussian-smoothed source, a straight path to x1, noise that vanishes exactly at
t=1. The network predicts the ENDPOINT x1 (model.py residual init makes an untrained net
the identity refiner); the implied velocity is v = (x1_pred - x_t) / (1 - t). Training is
masked endpoint-MSE; sampling integrates v by Euler over sliding temporal windows with an
optional guidance gradient. Everything here is space-agnostic — pass normalized x0/x1 and
denormalize the result; extra_losses that need metric poses receive a `normalizer`.
"""
import torch

from .state import unpack_state

_OBJ_ROT = slice(0, 6)
_OBJ_TRANS = slice(6, 9)
_HAND = slice(9, 57)


def _weight_vector(block_weights, device):
    w = torch.ones(57, device=device)
    if block_weights:
        w[_OBJ_ROT] = block_weights.get("obj_rot", 1.0)
        w[_OBJ_TRANS] = block_weights.get("obj_trans", 1.0)
        w[_HAND] = block_weights.get("hand", 1.0)
    return w


def bridge_loss(model, x0, x1, valid, cond_tokens, presence, rng=None, sigma0=0.1,
                block_weights=None, extra_losses=None, normalizer=None, batch=None):
    """Masked endpoint-MSE bridge loss. Returns (scalar_loss, logs_dict).

    extra_losses: list of callables(x1_pred_unpacked_denorm: dict, batch) -> (name, scalar),
    summed into the loss (empty by default) — the hook for later FK/contact terms.
    """
    B, T, Dd = x1.shape
    device = x1.device
    t = torch.rand(B, generator=rng, device=device)
    eps = torch.randn(B, T, Dd, generator=rng, device=device)
    tt = t[:, None, None]
    x_t = (1 - tt) * (x0 + sigma0 * eps) + tt * x1
    x1_pred = model(x_t, t, cond_tokens, presence)

    w = _weight_vector(block_weights, device)
    m = valid.float()
    err2 = (x1_pred - x1) ** 2
    loss = (err2 * w * m).sum() / m.sum().clamp_min(1.0)

    logs = {"loss": float(loss.detach()), "t_mean": float(t.mean())}
    e2d = err2.detach()
    for name, sl in (("obj_rot", _OBJ_ROT), ("obj_trans", _OBJ_TRANS), ("hand", _HAND)):
        msl = m[..., sl]
        logs[f"mse_{name}"] = float((e2d[..., sl] * msl).sum() / msl.sum().clamp_min(1.0))

    if extra_losses:
        pred = normalizer.inverse(x1_pred) if normalizer is not None else x1_pred
        unp = unpack_state(pred)
        for fn in extra_losses:
            name, val = fn(unp, batch)
            loss = loss + val
            logs[name] = float(val.detach()) if torch.is_tensor(val) else float(val)
    return loss, logs


def _window_bounds(T, window, overlap):
    if T <= window:
        return [(0, T)]
    stride = max(window - overlap, 1)
    starts = list(range(0, T - window + 1, stride))
    if starts[-1] != T - window:
        starts.append(T - window)
    return [(s, s + window) for s in starts]


def _blend_weights(wl, overlap, device):
    """Trapezoidal cross-fade weights, >=1 everywhere (so every frame gets coverage);
    linear ramps of width `overlap` blend adjacent windows."""
    idx = torch.arange(wl, device=device)
    tri = torch.minimum(idx + 1, wl - idx).float()
    return torch.minimum(tri, torch.tensor(float(overlap + 1), device=device))


@torch.no_grad()
def sample(model, x0, cond, presence, n_steps=8, window=32, overlap=16, sigma0=0.1,
           guidance=(), seed=None):
    """x0 [T,57] -> refined x1 [T,57] (same normalized space as x0).

    cond: callable(sl: slice) -> cond_tokens [1,N,d] for that window's frames (the chosen
    API — the caller slices its own per-frame conditioning). guidance: iterable of
    (fn, scale); each step subtracts scale * grad_x fn(x) from the velocity to descend the
    energy fn. Windows of length `window` (stride window-overlap) are Euler-integrated from
    t=0..1 starting at x0_win + sigma0*eps (seedable) and cross-faded on overlap.
    """
    T, Dd = x0.shape
    device = x0.device
    gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    out = torch.zeros(T, Dd, device=device)
    wsum = torch.zeros(T, 1, device=device)
    dt = 1.0 / n_steps
    for s, e in _window_bounds(T, window, overlap):
        wl = e - s
        x = x0[s:e] + sigma0 * torch.randn(wl, Dd, generator=gen, device=device)
        pres = presence[s:e].unsqueeze(0)
        ctok = cond(slice(s, e))
        for i in range(n_steps):
            ti = i * dt
            x1p = model(x.unsqueeze(0), torch.full((1,), ti, device=device), ctok, pres)[0]
            v = (x1p - x) / max(1.0 - ti, 1e-3)
            for fn, scale in guidance:
                with torch.enable_grad():
                    xg = x.detach().requires_grad_(True)
                    g = torch.autograd.grad(fn(xg).sum(), xg)[0]
                v = v - scale * g
            x = x + dt * v
        w = _blend_weights(wl, overlap, device)[:, None]
        out[s:e] += w * x
        wsum[s:e] += w
    return out / wsum.clamp_min(1e-8)
