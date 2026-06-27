"""Stage 6 (§2.1.5 / App C): adaptive contact optimization — geometry-level,
bounded correction of the replay hand. Object pose/mesh and MANO articulation
are unchanged. App C constants are used verbatim (see config.contact)."""
from __future__ import annotations
import numpy as np
from ..bundle import Bundle
from ..core.geometry import knn, vertex_normals, transform_points
from ..core import hand as H

NAME = "stage6_contact"; INDEX = 6


def active_window(stage_labels):
    """Return indices of frames whose label is grasp/move/place."""
    keep = {"grasp", "move", "place"}
    return [i for i, l in enumerate(stage_labels) if str(l) in keep]


def signed_distance(points, obj_pts, obj_normals):
    """Approximate signed distance of each query point to the object surface.

    Uses nearest-vertex + that vertex's outward normal to project the offset.
    Sign convention: + outside, − inside (penetration).

    Returns
    -------
    s : (N,) signed distances
    nn : (N, 3) nearest-surface outward normals
    """
    d, idx = knn(points, obj_pts, k=1)
    idx = idx[:, 0]
    nn = obj_normals[idx]
    nearest = obj_pts[idx]
    s = np.sum((points - nearest) * nn, axis=1)   # signed: + outside, − inside
    return s, nn


def whole_hand_translation(regions, s_list, n_list, gaps, weights, max_trans):
    """App C: d_t^k = mean(-n * ReLU(s - g_k)); aggregate by region weights; clip.

    Parameters
    ----------
    regions  : list of (M_k, 3) contact-region point arrays (unused in the sum
               itself but kept for API symmetry / future use)
    s_list   : list of (M_k,) signed distances for each region
    n_list   : list of (M_k, 3) nearest-surface normals for each region
    gaps     : list of scalar gap thresholds g_k (metres)
    weights  : list of scalar region importance weights w_k
    max_trans: scalar clip radius (metres)

    Returns
    -------
    delta : (3,) translation to apply to the whole hand
    """
    dks = []
    ws = []
    for _pts, s, n, g, w in zip(regions, s_list, n_list, gaps, weights):
        relu = np.maximum(s - g, 0.0)                 # ReLU(s - g_k)
        dk = np.mean(-n * relu[:, None], axis=0)      # pull toward surface
        dks.append(dk)
        ws.append(w)
    if not dks:
        return np.zeros(3)
    raw = np.average(np.stack(dks, 0), axis=0, weights=np.array(ws, dtype=float))
    norm = np.linalg.norm(raw)
    if norm <= max_trans or norm < 1e-12:
        return raw
    return raw * (max_trans / norm)


def select_opposing_finger(hand_verts, finger_idx, obj_pts):
    """Choose the finger (index/middle/ring/little) whose distal pad is closest
    (median distance) to the object surface — the 'opposing' finger for a
    pinch-style contact correction.

    Parameters
    ----------
    hand_verts : (V, 3) hand vertex array
    finger_idx : dict mapping finger name -> vertex index array
    obj_pts    : (P, 3) object surface points

    Returns
    -------
    finger name string
    """
    best, bestd = "index", np.inf
    for fmap_key in ["index", "middle", "ring", "little"]:
        pad = H.fingertip_pad_idx(finger_idx, fmap_key)
        if len(pad) == 0:
            continue
        d, _ = knn(hand_verts[pad], obj_pts, k=1)
        m = float(np.median(d))
        if m < bestd:
            bestd, best = m, fmap_key
    return best
