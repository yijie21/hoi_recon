"""DiT-style temporal transformer — the flow field of the bridge refiner.

Given a noised/interpolated trajectory x_t, the flow time t, and cross-attention tokens
from the observation encoders, it predicts the ENDPOINT trajectory x1 (clean GT). Two
deliberate init choices make an *untrained* model the identity refiner (x1_pred == x_t):
AdaLN-Zero gates zero every block's contribution, and a zero-init output head is ADDED as
a residual to x_t. So training starts from "return the coarse input unchanged" and only
ever learns the correction on top — stable, and it gives the data-corruption calibration a
known baseline to measure against.
"""
import torch
import torch.nn as nn

from .geometry import sinusoidal_embedding
from .state import D as STATE_D

DEFAULTS = {"d": 512, "depth": 8, "heads": 8, "mlp_ratio": 4, "t_freq": 256}


def modulate(x, shift, scale):  # DiT AdaLN modulation; shift/scale [B,d] -> broadcast over T
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """pre-LN self-attn over T -> cross-attn to cond tokens -> MLP, each AdaLN-Zero gated.
    Self-attn and MLP use AdaLN(shift,scale,gate); cross-attn uses a zero-init per-block
    gate — so at init every sub-layer contributes exactly zero."""

    def __init__(self, d, heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm_c = nn.LayerNorm(d, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_ratio * d), nn.GELU(), nn.Linear(mlp_ratio * d, d))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        self.cross_gate = nn.Parameter(torch.zeros(d))

    def forward(self, x, c, cond):
        s_sa, c_sa, g_sa, s_mlp, c_mlp, g_mlp = self.adaLN(c).chunk(6, dim=-1)
        h = modulate(self.norm1(x), s_sa, c_sa)
        x = x + g_sa.unsqueeze(1) * self.self_attn(h, h, h, need_weights=False)[0]
        x = x + self.cross_gate * self.cross_attn(self.norm_c(x), cond, cond, need_weights=False)[0]
        x = x + g_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), s_mlp, c_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d, elementwise_affine=False)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.head = nn.Linear(d, STATE_D)  # zero-init -> residual starts at identity

    def forward(self, x, c):
        shift, scale = self.adaLN(c).chunk(2, dim=-1)
        return self.head(modulate(self.norm(x), shift, scale))


class HOIDiT(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        cfg = {**DEFAULTS, **(config or {})}
        self.cfg = cfg
        d = cfg["d"]
        self.d, self.t_freq = d, cfg["t_freq"]
        self.x_proj = nn.Linear(STATE_D + 2, d)  # +2 per-hand presence flags
        self.t_mlp = nn.Sequential(nn.Linear(cfg["t_freq"], d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([DiTBlock(d, cfg["heads"], cfg["mlp_ratio"]) for _ in range(cfg["depth"])])
        self.final = FinalLayer(d)
        self._init_zero()

    def _init_zero(self):
        for blk in self.blocks:
            nn.init.zeros_(blk.adaLN[-1].weight)
            nn.init.zeros_(blk.adaLN[-1].bias)
        for m in (self.final.adaLN[-1], self.final.head):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x_t, t, cond_tokens, presence):
        """x_t [B,T,57], t [B] in [0,1], cond_tokens [B,N,d], presence [B,T,2] bool
        -> x1_pred [B,T,57]."""
        B, T, _ = x_t.shape
        h = self.x_proj(torch.cat([x_t, presence.float()], dim=-1))
        h = h + sinusoidal_embedding(torch.arange(T, device=x_t.device), self.d)[None]
        c = self.t_mlp(sinusoidal_embedding(t, self.t_freq))
        for blk in self.blocks:
            h = blk(h, c, cond_tokens)
        return x_t + self.final(h, c)  # residual endpoint prediction
