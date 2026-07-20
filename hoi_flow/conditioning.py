"""Pluggable observation-conditioning for the flow refiner.

The refiner must run with *any subset* of modalities available (RGB usually; depth/seg
only when a sensor or a stage-1 mask exists; obj_points/K always cheap). So each modality
is an independent encoder emitting a fixed number of d_model tokens, and cross-attention
in the DiT consumes their concatenation. A dropped modality contributes zero tokens
(modality dropout at train time -> robustness to missing sensors at test time); if every
modality is dropped we still emit one learned null token so cross-attention has a key.

Token budget (per batch element):
  rgb        16 tokens / frame   (DINOv2 or tinycnn patch grid, avg-pooled to a 4x4 grid)
  depth      16 tokens / frame
  seg_mask   16 tokens / frame
  obj_points 16 static tokens    (PointNet-lite, 16 chunked max-pools)
  K           1 static token
=> with T frames and all modalities: 16*T*3 + 16 + 1 tokens.
Per-frame tokens get a sinusoidal FRAME-index embedding (so attention knows which frame a
token describes) + a learned per-modality type embedding; static tokens get type only. The
frame index is LOCAL to the frames handed in `batch` (encode a window -> indices 0..w-1),
which keeps windowed sampling consistent with the model's own local frame positions.

RGB note: at 256px the DINOv2 grid is deliberately pooled to a coarse 4x4 in v1 — RGB is
*global* evidence here, not pixel-precise cues. Pixel precision is stage-1's job.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import sinusoidal_embedding

MODALITIES = ["rgb", "depth", "seg_mask", "obj_points", "K"]
_PER_FRAME = {"rgb", "depth", "seg_mask"}
_TYPE_ID = {m: i for i, m in enumerate(MODALITIES)}
_NULL_TYPE = len(MODALITIES)
TOKENS_PER_FRAME = 16


class _PatchConv(nn.Module):
    """Strided conv -> adaptive 4x4 avg-pool -> 16 tokens of d_model. The adaptive pool
    makes it resolution-agnostic (R=32 in tests, 256 in production)."""

    def __init__(self, in_ch, d):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 2, 1), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.GELU(),
        )
        self.proj = nn.Linear(128, d)

    def forward(self, x):  # [N,in_ch,R,R] -> [N,16,d]
        x = F.adaptive_avg_pool2d(self.conv(x), 4)  # [N,128,4,4]
        x = x.flatten(2).transpose(1, 2)            # [N,16,128]
        return self.proj(x)


class _TinyCNNRGB(nn.Module):
    """Small strided CNN — the download-free RGB backbone (tests / no-network)."""

    def __init__(self, d):
        super().__init__()
        self.patch = _PatchConv(3, d)

    def forward(self, rgb):  # [B,T,3,R,R] -> [B,T,16,d]
        B, T = rgb.shape[:2]
        return self.patch(rgb.flatten(0, 1)).reshape(B, T, TOKENS_PER_FRAME, -1)


class _DINORGB(nn.Module):
    """Frozen DINOv2 ViT-S/14 via torch.hub, lazy-loaded at first forward (one network
    fetch, then cached). Eval + no_grad; only the linear projection trains. 224px patch
    grid (16x16) avg-pooled to 4x4 -> 16 tokens."""

    def __init__(self, d):
        super().__init__()
        self._dino = None
        self.proj = nn.Linear(384, d)

    def _load(self, device):
        if self._dino is None:
            m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
            m.eval().to(device)
            for p in m.parameters():
                p.requires_grad_(False)
            self._dino = m

    def forward(self, rgb):  # [B,T,3,R,R] -> [B,T,16,d]
        B, T = rgb.shape[:2]
        self._load(rgb.device)
        x = F.interpolate(rgb.flatten(0, 1), size=224, mode="bilinear", align_corners=False)
        with torch.no_grad():
            feats = self._dino.forward_features(x)["x_norm_patchtokens"]  # [BT,256,384]
        g = feats.transpose(1, 2).reshape(-1, 384, 16, 16)
        g = F.adaptive_avg_pool2d(g, 4).flatten(2).transpose(1, 2)        # [BT,16,384]
        return self.proj(g).reshape(B, T, TOKENS_PER_FRAME, -1)


class _DepthEnc(nn.Module):
    """Metric depth (0 = no return) -> normalized depth + a valid-channel -> patch tokens."""

    def __init__(self, d):
        super().__init__()
        self.patch = _PatchConv(2, d)

    def forward(self, depth):  # [B,T,R,R] meters -> [B,T,16,d]
        B, T = depth.shape[:2]
        valid = (depth > 0).float()
        dn = (depth / 1.5).clamp(0.0, 4.0) * valid            # 0 where invalid
        x = torch.stack([dn, valid], dim=2).flatten(0, 1)     # [BT,2,R,R]
        return self.patch(x).reshape(B, T, TOKENS_PER_FRAME, -1)


class _SegEnc(nn.Module):
    """Instance class map (0..4) -> class embedding -> patch tokens."""

    def __init__(self, d):
        super().__init__()
        self.embed = nn.Embedding(5, 8)
        self.patch = _PatchConv(8, d)

    def forward(self, seg):  # [B,T,R,R] long -> [B,T,16,d]
        B, T = seg.shape[:2]
        e = self.embed(seg.long()).permute(0, 1, 4, 2, 3).flatten(0, 1)  # [BT,8,R,R]
        return self.patch(e).reshape(B, T, TOKENS_PER_FRAME, -1)


class _ObjPointsEnc(nn.Module):
    """PointNet-lite: shared pointwise MLP then 16 chunked max-pools
    (adaptive_max_pool1d) -> 16 permutation-robust static tokens for object shape."""

    def __init__(self, d):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, d))

    def forward(self, pts):  # [B,Nv,3] -> [B,16,d]
        f = self.mlp(pts).transpose(1, 2)                      # [B,d,Nv]
        return F.adaptive_max_pool1d(f, TOKENS_PER_FRAME).transpose(1, 2)


class _KEnc(nn.Module):
    """Flatten intrinsics -> MLP -> 1 static token."""

    def __init__(self, d):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(9, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, K):  # [B,3,3] -> [B,1,d]
        return self.mlp(K.reshape(K.shape[0], 9)).unsqueeze(1)


class CondEncoders(nn.Module):
    """Encode a batch of enabled modalities into cross-attention tokens.

    config: dict enabling modalities by truthy key (e.g. {"rgb": True, "K": True}) plus an
    optional "rgb_backbone" in {"dino","tinycnn"} (default "dino"). Modalities enabled but
    absent from a given `batch` are treated as dropped.
    forward(batch, drop) -> (tokens [B,N,d], meta) where meta = {"layout": [(name,count)...],
    "n_tokens": N}.
    """

    def __init__(self, d_model, config):
        super().__init__()
        self.d = d_model
        self.encoders = nn.ModuleDict()
        rgb_backbone = config.get("rgb_backbone", "dino")
        for name in MODALITIES:
            if not config.get(name):
                continue
            if name == "rgb":
                self.encoders[name] = _TinyCNNRGB(d_model) if rgb_backbone == "tinycnn" else _DINORGB(d_model)
            elif name == "depth":
                self.encoders[name] = _DepthEnc(d_model)
            elif name == "seg_mask":
                self.encoders[name] = _SegEnc(d_model)
            elif name == "obj_points":
                self.encoders[name] = _ObjPointsEnc(d_model)
            elif name == "K":
                self.encoders[name] = _KEnc(d_model)
        self.type_emb = nn.Embedding(_NULL_TYPE + 1, d_model)
        self.null_token = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, batch, drop=None):
        drop = drop or {}
        ref = next(iter(batch.values()))
        B, device = ref.shape[0], ref.device
        toks, layout = [], []
        for name, enc in self.encoders.items():
            if drop.get(name) or name not in batch:
                continue
            t = enc(batch[name])                                       # per-frame [B,T,16,d]; static [B,C,d]
            type_e = self.type_emb(torch.tensor(_TYPE_ID[name], device=device))
            if name in _PER_FRAME:
                T = t.shape[1]
                frame_e = sinusoidal_embedding(torch.arange(T, device=device), self.d)  # [T,d]
                t = (t + frame_e[None, :, None, :] + type_e).reshape(B, T * TOKENS_PER_FRAME, self.d)
            else:
                t = t + type_e
            toks.append(t)
            layout.append((name, t.shape[1]))
        if not toks:  # everything dropped -> single learned null token
            null = self.null_token.expand(B, 1, self.d) + self.type_emb(torch.tensor(_NULL_TYPE, device=device))
            return null, {"layout": [("null", 1)], "n_tokens": 1}
        tokens = torch.cat(toks, dim=1)
        return tokens, {"layout": layout, "n_tokens": tokens.shape[1]}
