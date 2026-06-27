"""Procedural geometry-level hand: vertex set, per-finger index groups,
fingertip pads, thenar region, and MANO-style finger-chain weights for the
App-C local contact correction. Real MANO arrives via backends/real.py."""
from __future__ import annotations
import numpy as np

FINGERS = ["thumb", "index", "middle", "ring", "little"]
_LENGTHS = {"thumb": 0.045, "index": 0.060, "middle": 0.065, "ring": 0.060, "little": 0.050}
_BASE_X = {"thumb": -0.035, "index": -0.0175, "middle": 0.0, "ring": 0.0175, "little": 0.035}


def procedural_hand(n: int = 778, seed: int = 0):
    rng = np.random.default_rng(seed)
    verts, fidx, z_along = [], {}, []
    cur = 0
    n_palm = n // 3
    verts.append(rng.uniform([-0.040, -0.012, -0.030], [0.040, 0.012, 0.005], (n_palm, 3)))
    fidx["palm"] = np.arange(cur, cur + n_palm); cur += n_palm
    z_along.append(np.zeros(n_palm))
    n_fing = n - n_palm; per = n_fing // 5
    for k, f in enumerate(FINGERS):
        cnt = per if k < 4 else n_fing - per * 4
        L = _LENGTHS[f]
        z = rng.uniform(0.0, L, cnt)
        x = _BASE_X[f] + rng.uniform(-0.006, 0.006, cnt)
        y = rng.uniform(-0.008, 0.008, cnt)
        verts.append(np.stack([x, y, z], 1))
        fidx[f] = np.arange(cur, cur + cnt); cur += cnt
        z_along.append(z / L)                       # 0 at base .. 1 at tip
    V = np.concatenate(verts, 0).astype(np.float64)
    Z = np.concatenate(z_along, 0)
    fidx["z_norm"] = Z                              # per-vertex normalized along-finger coord
    # 21 joints: wrist + 4 per finger
    joints = [[0.0, 0.0, -0.020]]
    for f in FINGERS:
        for fr in (0.25, 0.5, 0.75, 1.0):
            joints.append([_BASE_X[f], 0.0, fr * _LENGTHS[f]])
    return V, np.asarray(joints, float), fidx


def fingertip_pad_idx(fidx, finger):
    z = np.asarray(fidx["z_norm"])
    idx = np.asarray(fidx[finger])
    return idx[z[idx] > 0.7]                         # distal pad = top 30% of finger


def thenar_idx(fidx):
    # hukou/thenar: palm vertices near the thumb base (negative x side)
    return fidx["palm"]


def finger_chain_weights(verts, fidx, finger):
    z = np.asarray(fidx["z_norm"])
    w = np.zeros(len(verts))
    fi = np.asarray(fidx[finger])
    w[fi] = np.clip(z[fi], 0.0, 1.0)                # distal-heavy, palm≈0
    return w
