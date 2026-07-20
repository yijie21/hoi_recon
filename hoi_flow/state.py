"""The 57-D per-frame HOI state vector — the single contract the flow model operates on.

Layout (D=57), everything in the pinhole CAMERA frame:
  [0:6]   object rotation (6D)
  [6:9]   object translation (m)
  left hand  (h=0): [9:15]  wrist 6D, [15:18] wrist trans, [18:33] 15 MANO thetas
  right hand (h=1): [33:39] wrist 6D, [39:42] wrist trans, [42:57] 15 MANO thetas

Absent hands / frames arrive as NaN; pack_state zeros them in x and marks valid=False so
the loss can mask them and normalization statistics can ignore them. Roundtrip
pack -> unpack is exact for valid entries when the input rotations are orthonormal.
"""
import json

import torch

from .geometry import matrix_to_rot6d, rot6d_to_matrix

D = 57
_HAND_BLOCK = 24  # 6 wrist-rot6d + 3 wrist-trans + 15 thetas

# rotation-6D dims (identity-normalized): object + both wrists
ROT6D_DIMS = list(range(0, 6)) + list(range(9, 15)) + list(range(33, 39))


def _hand_off(h):
    return 9 + _HAND_BLOCK * h


def pack_state(obj_pose, hand_wrist, hand_thetas):
    """(obj_pose [T,4,4], hand_wrist [T,2,4,4], hand_thetas [T,2,15])
    -> (x [T,57], valid [T,57] bool). NaN inputs -> 0 in x, False in valid."""
    T = obj_pose.shape[0]
    dev = obj_pose.device
    x = torch.zeros(T, D, dtype=torch.float32, device=dev)
    valid = torch.zeros(T, D, dtype=torch.bool, device=dev)

    # validate only the [:3,:4] sub-block we actually read (R + t) — robust to however the
    # SE3 bottom row is stored; an absent frame/hand is all-NaN so this still marks it invalid.
    obj_ok = torch.isfinite(obj_pose[:, :3, :4]).all(dim=(-1, -2))  # [T]
    x[:, 0:6] = torch.nan_to_num(matrix_to_rot6d(obj_pose[:, :3, :3]))
    x[:, 6:9] = torch.nan_to_num(obj_pose[:, :3, 3])
    valid[:, 0:9] = obj_ok[:, None]

    for h in (0, 1):
        o = _hand_off(h)
        wrist, thetas = hand_wrist[:, h], hand_thetas[:, h]         # [T,4,4], [T,15]
        wrist_ok = torch.isfinite(wrist[:, :3, :4]).all(dim=(-1, -2))  # [T]
        theta_ok = torch.isfinite(thetas).all(dim=-1)               # [T]
        x[:, o:o + 6] = torch.nan_to_num(matrix_to_rot6d(wrist[:, :3, :3]))
        x[:, o + 6:o + 9] = torch.nan_to_num(wrist[:, :3, 3])
        x[:, o + 9:o + 24] = torch.nan_to_num(thetas)
        valid[:, o:o + 9] = wrist_ok[:, None]
        valid[:, o + 9:o + 24] = theta_ok[:, None]

    x = torch.where(valid, x, torch.zeros_like(x))  # hard-zero invalid dims (no stray finite)
    return x, valid


def _se3(R, t):
    shp = R.shape[:-2]
    M = torch.zeros(*shp, 4, 4, dtype=R.dtype, device=R.device)
    M[..., :3, :3] = R
    M[..., :3, 3] = t
    M[..., 3, 3] = 1.0
    return M


def unpack_state(x):
    """x [...,57] (leading dims e.g. [T] or [B,T]) -> dict of camera-frame quantities.
    rot6d -> orthonormal R. Inverse of pack_state on valid entries."""
    obj_pose = _se3(rot6d_to_matrix(x[..., 0:6]), x[..., 6:9])
    wrists, thetas = [], []
    for h in (0, 1):
        o = _hand_off(h)
        wrists.append(_se3(rot6d_to_matrix(x[..., o:o + 6]), x[..., o + 6:o + 9]))
        thetas.append(x[..., o + 9:o + 24])
    return {
        "obj_pose": obj_pose,                       # [...,4,4]
        "hand_wrist": torch.stack(wrists, dim=-3),  # [...,2,4,4]
        "hand_thetas": torch.stack(thetas, dim=-2),  # [...,2,15]
    }


class Normalizer:
    """Per-dim z-scoring of the state, EXCEPT rotation-6D dims which get the identity
    transform (mean 0, std 1). Why skip rotations: a 6D rotation already sits at unit
    scale on a nonlinear manifold; per-dim mean/std would shear it off that manifold so
    rot6d_to_matrix no longer recovers the intended rotation. Only translations and MANO
    thetas — genuine Euclidean quantities with meaningful per-dim spread — are standardized.
    """

    def __init__(self, mean=None, std=None):
        self.mean = torch.zeros(D) if mean is None else torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.ones(D) if std is None else torch.as_tensor(std, dtype=torch.float32)

    def fit(self, xs, valids=None):
        """xs: list of [.,57] tensors. valids (optional): matching bool masks so absent-hand
        zeros don't bias the statistics."""
        X = torch.cat([x.reshape(-1, D) for x in xs], 0).float()
        if valids is not None:
            M = torch.cat([v.reshape(-1, D) for v in valids], 0).float()
            cnt = M.sum(0).clamp_min(1.0)
            mean = (X * M).sum(0) / cnt
            std = (((X - mean) ** 2) * M).sum(0).div(cnt).sqrt()
        else:
            mean, std = X.mean(0), X.std(0)
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        mean[ROT6D_DIMS] = 0.0
        std[ROT6D_DIMS] = 1.0
        self.mean, self.std = mean, std
        return self

    def transform(self, x):
        return (x - self.mean.to(x)) / self.std.to(x)

    def inverse(self, x):
        return x * self.std.to(x) + self.mean.to(x)

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"mean": self.mean.tolist(), "std": self.std.tolist()}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d["mean"], d["std"])
