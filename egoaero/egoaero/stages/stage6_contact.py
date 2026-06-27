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


def triangular_smooth(deltas, window, boundary_frames):
    """Finite-window triangular kernel smoothing + boundary taper (App C)."""
    T = deltas.shape[0]; half = window // 2
    k = np.array([half + 1 - abs(i - half) for i in range(window)], float)
    k = k / k.sum()
    out = np.zeros_like(deltas)
    for c in range(deltas.shape[1]):
        out[:, c] = np.convolve(np.pad(deltas[:, c], half, mode="edge"), k, "valid")[:T]
    b = boundary_frames
    if b > 0 and T > 0:                                  # taper at active-segment ends
        taper = np.ones(T)
        ramp = np.linspace(0, 1, min(b, T))
        taper[:len(ramp)] = ramp; taper[-len(ramp):] = ramp[::-1]
        out = out * taper[:, None]
    return out


def penetration_pushback(points, s, normals, eps, max_pb):
    """App C: penetrating set s<-eps; push-back along normals, clipped."""
    pen = s < -eps
    if pen.sum() == 0:
        return np.zeros(3)
    depth = np.maximum(-eps - s[pen], 0.0)
    r = np.mean(normals[pen] * depth[:, None], axis=0)
    norm = np.linalg.norm(r)
    return r if norm <= max_pb or norm < 1e-12 else r * (max_pb / norm)


def _obj_world(obj_verts, obj_faces, pose):
    ow = transform_points(obj_verts, pose)
    return ow, vertex_normals(ow, obj_faces)


def run(ctx) -> Bundle:
    cfg = ctx.cfg; cc = cfg.contact
    s5 = ctx.load("stage5_ego_comp")
    hv = s5["hand_verts_t"].copy(); hj = s5["hand_joints_t"].copy()
    obj_poses = s5["obj_poses_t"]; ov = s5["obj_verts"]; of = s5["obj_faces"].astype(int)
    fidx = {k: (np.asarray(v, float) if k == "z_norm" else np.asarray(v, int))
            for k, v in s5.meta["finger_idx"].items()}
    labels = s5.meta["stage_labels"]; T = hv.shape[0]
    win = active_window(labels)

    def pen_gap(vh):
        pens, gaps = [], []
        for i in range(T):
            ow, on = _obj_world(ov, of, obj_poses[i])
            s, _ = signed_distance(vh[i], ow, on)
            pens.append(np.maximum(-s, 0).sum())
            pad = H.fingertip_pad_idx(fidx, "thumb")
            gaps.append(np.median(np.abs(s[pad])) if len(pad) else 0.0)
        return float(np.mean(pens) * 1000), float(np.median(gaps) * 1000)

    pen_b, gap_b = pen_gap(hv)
    raw_delta = np.zeros((T, 3))
    for i in win:
        ow, on = _obj_world(ov, of, obj_poses[i])
        thumb = H.fingertip_pad_idx(fidx, "thumb")
        opp_f = select_opposing_finger(hv[i], fidx, ow)
        opp = H.fingertip_pad_idx(fidx, opp_f)
        huk = H.thenar_idx(fidx)
        regions, s_list, n_list, gaps, weights = [], [], [], [], []
        for pts_idx, g, w in [(thumb, cc.contact_gap_m, cc.region_weights["thumb"]),
                              (opp, cc.opp_gap_m, cc.region_weights["opp"]),
                              (huk, cc.thenar_gap_m, cc.region_weights["hukou"])]:
            if len(pts_idx) == 0:
                continue
            s, nn = signed_distance(hv[i][pts_idx], ow, on)
            regions.append(hv[i][pts_idx]); s_list.append(s); n_list.append(nn)
            gaps.append(g); weights.append(w)
        raw_delta[i] = whole_hand_translation(regions, s_list, n_list, gaps, weights,
                                              cc.max_global_trans_m)
    delta = triangular_smooth(raw_delta, cc.smooth_window, cc.boundary_frames)
    for i in range(T):
        hv[i] += delta[i]; hj[i] += delta[i]

    # local finger correction (thumb + opposing finger) weighted by finger chain
    for i in win:
        ow, on = _obj_world(ov, of, obj_poses[i])
        opp_f = select_opposing_finger(hv[i], fidx, ow)
        for f, g in [("thumb", cc.contact_gap_m), (opp_f, cc.opp_gap_m)]:
            pad = H.fingertip_pad_idx(fidx, f)
            if len(pad) == 0:
                continue
            s, nn = signed_distance(hv[i][pad], ow, on)
            off = np.mean(-nn * np.maximum(s - g, 0)[:, None], axis=0)
            off = np.clip(off, -cc.max_finger_disp_m, cc.max_finger_disp_m)
            # vertex-level correction weighted by finger chain
            w = H.finger_chain_weights(hv[i], fidx, f)
            hv[i] += w[:, None] * off
            # joint-level correction: distal ramp [0.25,0.5,0.75,1.0] on 4 joints of finger f
            k = H.FINGERS.index(f)
            base = k * 4 + 1
            jw = np.zeros(hj[i].shape[0])
            for qi, alpha in enumerate([0.25, 0.5, 0.75, 1.0]):
                jw[base + qi] = alpha
            hj[i] += jw[:, None] * off
        # penetration push-back (whole hand)
        s_all, n_all = signed_distance(hv[i], ow, on)
        r = penetration_pushback(hv[i], s_all, n_all, cc.pen_eps_m, cc.max_pushback_m)
        hv[i] += r; hj[i] += r

    pen_a, gap_a = pen_gap(hv)
    # contact mask: hand verts within contact gap of the object surface
    cmask = np.zeros((T, hv.shape[1]), bool)
    for i in range(T):
        ow, on = _obj_world(ov, of, obj_poses[i])
        s, _ = signed_distance(hv[i], ow, on)
        cmask[i] = np.abs(s) < (cc.contact_gap_m * cc.contact_mask_radius_factor)
    return Bundle(arrays={"hand_verts_t": hv, "hand_joints_t": hj, "contact_mask": cmask,
                          "obj_poses_t": obj_poses, "obj_verts": ov, "obj_faces": of},
                  meta={"pen_before_mm": pen_b, "pen_after_mm": pen_a,
                        "gap_before_mm": gap_b, "gap_after_mm": gap_a,
                        "finger_idx": s5.meta["finger_idx"], "stage_labels": labels})
