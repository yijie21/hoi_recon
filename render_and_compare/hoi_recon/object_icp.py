"""Locked-scale rigid ICP object placement: register the canonical object mesh
per frame onto the depth point cloud inside the object mask.

Replaces the depth-lift-anchored object trajectory with a direct geometric
registration against the depth substrate. Scale is deliberately NOT a per-frame
free variable: partial-view fits cannot observe it (a 53% oversized mesh fits
the visible front at unchanged residual — see compare/hoi4d/gate2/sam3d_icp/),
so the object stays a rigid body. What IS observable is ONE global scale from
the union of all registered frames (`global_scale_refit`): the fused
multi-frame cloud covers enough of the object to pin the metric size that the
per-frame front views hide (kettle_N15: stage-3's bbox heuristic was 13%
undersized, per-frame bias showed only −0.7 cm; the fused refit recovered
s=1.13 and cut the per-frame residual 6.3→3.7 mm — Route B of RESULTS.md).

Validated on kettle_N15 vs GT depth: visible-surface MAE 2.97 → 1.12 cm,
wiggle 1.58 → 0.79 cm over stages 4-7 (rigid), plus the scale-refit gains
above. Numpy/scipy/cv2/trimesh only — runs in-process in the main env.
"""
from __future__ import annotations

import os

import numpy as np

from .logging_utils import log

# Attitude-search spread gate (LAB chroma units): apply the chroma-scored
# azimuth correction only when the hypotheses' chroma scores spread at least
# this much — i.e. the object has distinctive texture that actually
# discriminates orientation. Calibrated on the mesh-controlled 6-clip HOT3D
# A/B (bottle spread 31 → helped; masher 10 / uniform objects 1-2 → hurt or
# random). Below it the depth-ICP attitude is kept.
SPREAD_MIN = 18.0


def _attitude_apply(best_is_identity, spread, spread_min=SPREAD_MIN):
    """Decide whether to apply the winning attitude hypothesis: it must be a
    non-identity winner AND the chroma spread across hypotheses must clear
    spread_min (distinctive texture actually discriminated orientation)."""
    return (not best_is_identity) and (spread >= spread_min)


def _umeyama_rigid(src, dst):
    """Rigid (R, t) minimizing ||R@src + t - dst|| (Kabsch)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    cov = (dst - mu_d).T @ (src - mu_s) / len(src)
    U, _, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    return R, mu_d - R @ mu_s


def _icp(src_pts, src_tree, tgt, R0, t0, trim, iters, rot_free=True):
    """Trimmed rigid ICP; correspondences target -> canonical mesh samples
    (the target is a partial front view, so every target point has a match).
    rot_free=False keeps R fixed at R0 and solves translation only — use when
    rotation comes from image evidence (stage-3 silhouette tracker): the
    top-down depth patch is near rotation-symmetric AND trimming discards the
    few disambiguating points (spout/handle), so free rotation walks into a
    wrong basin (33-97 deg off on kettle_N15) that depth residuals can't see."""
    R, t = R0.copy(), t0.copy()
    prev = np.inf
    res = prev
    for _ in range(iters):
        d, j = src_tree.query((tgt - t) @ R, workers=-1)
        keep = d <= np.quantile(d, trim)
        if rot_free:
            R, t = _umeyama_rigid(src_pts[j[keep]], tgt[keep])
        else:
            t = tgt[keep].mean(0) - R @ src_pts[j[keep]].mean(0)
        res = float(np.sqrt(np.mean(
            np.sum((src_pts[j[keep]] @ R.T + t - tgt[keep]) ** 2, 1))))
        if abs(prev - res) < 1e-6:
            break
        prev = res
    return R, t, res


def _solve_shared_scale(src_tree, src_pts, qs, s0, trim=0.95, iters=5):
    """One global scale about the canonical origin from the FUSED cloud (all
    frames' canonical-frame targets q, poses frozen): q ~ s*m. No per-frame
    centering and a mild trim — per-frame centering + tight trims remove the
    very evidence (radial extent mismatch, boundary correspondences) that
    makes global scale observable; the validated Route-B solve is exactly
    this fixed-pose fused-cloud fit (compare/hoi4d/gate2/sam3d_icp)."""
    q = np.concatenate(qs)
    s = s0
    for _ in range(iters):
        _, j = src_tree.query(q / s, workers=-1)
        m = src_pts[j]
        r = np.linalg.norm(s * m - q, axis=1)
        keep = r <= np.quantile(r, trim)
        s_new = float((m[keep] * q[keep]).sum() / (m[keep] * m[keep]).sum())
        if abs(s_new / s - 1.0) < 1e-4:
            return s_new
        s = s_new
    return s


def _so3_exp(r):
    """Batched axis-angle -> rotation matrices, torch [T,3] -> [T,3,3]."""
    import torch
    th = r.norm(dim=1, keepdim=True).clamp(min=1e-8)
    k = r / th
    K = torch.zeros(len(r), 3, 3, dtype=r.dtype, device=r.device)
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    I = torch.eye(3, dtype=r.dtype, device=r.device).expand_as(K)
    s, c = th.sin().unsqueeze(-1), th.cos().unsqueeze(-1)
    return I + s * K + (1 - c) * (K @ K)


def _rotz(a):
    """Rotation about the canonical +z axis by angle `a` (radians)."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_axis(axis, a):
    """Rotation by `a` (radians) about canonical axis 0/1/2 (x/y/z)."""
    c, s = np.cos(a), np.sin(a)
    R = np.eye(3)
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    R[i, i], R[i, j] = c, -s
    R[j, i], R[j, j] = s, c
    return R


def _attitude_hypotheses(n_hyp):
    """Azimuth hypotheses for the anchor attitude search.

    The object 'up'/symmetry axis is unknown, so — as the brief prescribes when
    unsure — we grid the azimuth about ALL THREE canonical axes with n_hyp//3
    equally-spaced non-zero rotations each, plus the identity (no correction).
    Returns 1 + 3*(n_hyp//3) numpy 3x3 rotations (e.g. n_hyp=12 -> 13). Fully
    deterministic (fixed angular grid, no randomness)."""
    per = max(1, int(n_hyp) // 3)          # non-identity rotations per axis
    hyps = [np.eye(3)]
    for axis in range(3):
        for k in range(1, per + 1):
            hyps.append(_rot_axis(axis, 2.0 * np.pi * k / (per + 1)))
    return hyps


def _score_hypothesis(src_pts, tgt, dt, lab_img, K, R, t, s, src_cols, rng,
                      trim):
    """Cheap numpy score for one attitude hypothesis at a single anchor frame
    (lower is better). Returns (score, chroma_mean, depth_rms_mm).

    The whole point of the attitude search is to resolve the AZIMUTH DOF that
    depth + silhouette are blind to: a revolution / near-symmetric object has
    ~equal depth RMS at every azimuth, so a depth-weighted score is flat in
    azimuth and trivially returns identity. So WHEN colors are available the
    score is CHROMA-DOMINANT — raw-LAB chroma mismatch decides azimuth, with
    depth RMS (mm) + out-of-mask DT (px) kept only as a mild 0.1-weight validity
    gate that still rejects a wildly mislocated hypothesis. With NO colors we
    fall back to the depth-dominant geometric score (unchanged behaviour).
    `chroma_mean` is np.nan when colors are unavailable / no in-mask points.

    `dt` is the pre-computed full-res out-of-mask distance transform and
    `lab_img` the pre-computed full-res LAB image (float32) or None — both
    hoisted out of the hypothesis loop by the caller (frame-constant). `src_cols`
    is in [0,1]. Kept light: KDTree depth RMS on <=2000 mesh points, image terms
    on <=500 projected points. `rng` is fresh-seeded per hypothesis so every
    hypothesis is scored on the SAME point subset (fair + deterministic)."""
    import cv2
    from scipy.spatial import cKDTree

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    src_s = src_pts * s
    n = len(src_s)
    idx = rng.choice(n, min(2000, n), replace=False)
    Xc = src_s[idx] @ R.T + t                       # posed points, camera frame
    # (a) trimmed depth RMS (mm): nearest posed point to each target point
    d, _ = cKDTree(Xc).query(tgt, workers=-1)
    keep = d <= np.quantile(d, trim)
    depth_rms = float(np.sqrt(np.mean(d[keep] ** 2)) * 1000.0)
    # image-space terms on a <=500 subset of the posed points
    pj = np.arange(len(Xc)) if len(Xc) <= 500 \
        else rng.choice(len(Xc), 500, replace=False)
    Xp = Xc[pj]
    z = np.clip(Xp[:, 2], 0.1, None)
    H, W = dt.shape
    ui = np.clip(np.round(Xp[:, 0] / z * fx + cx).astype(int), 0, W - 1)
    vi = np.clip(np.round(Xp[:, 1] / z * fy + cy).astype(int), 0, H - 1)
    dt_pt = dt[vi, ui]
    dt_mean = float(dt_pt.mean())
    # (c) LAB-chroma mismatch of in-mask projected points, in RAW LAB units
    chroma_mean = np.nan
    if lab_img is not None and src_cols is not None:
        inmask = dt_pt < 1.0
        if inmask.any():
            cols = np.asarray(src_cols, np.float32)[idx][pj]     # [0,1]
            lab_pts = cv2.cvtColor((cols[None] * 255).astype(np.uint8),
                                   cv2.COLOR_RGB2LAB)[0].astype(np.float32)
            ia = lab_img[vi, ui, 1:] - 128.0
            pa = lab_pts[:, 1:] - 128.0
            chroma = np.sqrt(((ia - pa) ** 2).sum(1))
            chroma_mean = float(chroma[inmask].mean())
    if np.isfinite(chroma_mean):
        # chroma-dominant: azimuth decided by color; depth+DT only gate position
        score = chroma_mean + 0.1 * depth_rms + 0.1 * dt_mean
    else:
        # geometry-only fallback (no colors, or no in-mask points)
        score = depth_rms + dt_mean
    return score, chroma_mean, depth_rms


def _joint_refine(src_pts, targets, masks_dir, K, poses0, s0, hand_boxes,
                  hand_valid, get, src_cols=None, frames_dir=None,
                  grasp_mask=None, wrist_R=None):
    """Joint depth + silhouette refinement (torch, GPU if available): per-frame
    rigid poses AND one global per-axis (anisotropic) canonical scale.

    Fixes the two blind spots of pure depth ICP measured on kettle_N15:
    (a) mesh overhang outside the mask is invisible to depth residuals — a
    distance-transform out-of-silhouette penalty (bilinearly sampled, so it is
    differentiable w.r.t. the projected points) sees it; the allowed region is
    object mask UNION valid hand boxes, because the object legitimately
    projects behind the occluding hand. (b) rotation is unobservable from the
    near-symmetric depth patch — the silhouette term supplies the image-space
    gradient, with a small prior keeping poses near the image-based init.
    Per-frame isotropic scale stays locked; the anisotropic scale is shared by
    ALL frames, so lateral axes get their evidence from the silhouettes while
    the depth-observed axes keep the fused-cloud evidence. Pixels and mm are
    naturally comparable here (~1.1 mm/px at 1.5 m), so unit weights balance.

    (c) T3 grasp-rigidity prior: on stable-grasp frame pairs (grasp_mask,
    from hoi_recon.grasp_segments.stable_grasp_mask) the object's rotation
    delta between consecutive frames is pulled toward the wrist's rotation
    delta (wrist_R, per-frame MANO global orientation) — the co-rotation
    signal that resolves azimuth on symmetric objects while firmly grasped,
    which depth+silhouette alone cannot see. Off unless w_grasp>0 AND
    grasp_mask has >=3 True frames (config key w_grasp, default 0.0).

    Returns (poses[T,4,4], s_axes[3], resid[T])."""
    import cv2
    import torch
    import torch.nn.functional as F
    from scipy.spatial import cKDTree

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = int(get("sil_downsample", 4))
    rounds = int(get("joint_rounds", 4))
    inner = int(get("joint_iters", 80))
    trim = float(get("trim", 0.8))
    n_sil = int(get("joint_sil_points", 2000))
    n_cov = int(get("sil_coverage_px", 300))
    erode = int(get("erode_px", 5))
    w_depth = float(get("w_depth", 1.0))
    w_sil = float(get("w_sil", 1.0))
    w_cov = float(get("w_cov", 0.3))
    w_rot = float(get("w_rot_prior", 30.0))
    w_temp = float(get("w_temp", 0.5))
    s_lo, s_hi = 0.7, 1.4

    # T3 grasp-rigidity prior: only active with w_grasp>0 AND a usable mask
    # (>=3 True frames, so at least one consecutive pair can exist) — this is
    # the flag-off purity contract: w_grasp=0 (default) or no/short grasp_mask
    # must leave every other loss term byte-identical to before this feature.
    w_grasp = float(get("w_grasp", 0.0))
    use_grasp = bool(w_grasp > 0 and grasp_mask is not None
                      and np.asarray(grasp_mask).sum() >= 3)

    T = len(poses0)
    valid = [i for i in range(T) if targets[i] is not None]
    if not valid:
        raise RuntimeError(
            "object ICP: no frames with valid mask∧depth — segmentation "
            "likely failed")
    m0 = np.load(os.path.join(masks_dir, f"{valid[0]:05d}.npy"))
    H, W = m0.shape
    Hd, Wd = H // D, W // D
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode + 1,) * 2)
    rng = np.random.default_rng(0)

    # per-frame: DT of the allowed region (0 inside mask+hand, px outside) and
    # coverage targets sampled from the eroded object mask
    dts = np.zeros((T, Hd, Wd), np.float32)
    cov_px = np.zeros((T, n_cov, 2), np.float32)
    cov_ok = np.zeros((T, n_cov), bool)
    for i in valid:
        m = np.load(os.path.join(masks_dir, f"{i:05d}.npy"))
        allowed = m.copy()
        if hand_boxes is not None and hand_valid is not None:
            for h in range(hand_boxes.shape[1]):
                b = hand_boxes[i, h]
                if hand_valid[i, h] and np.isfinite(b).all():
                    x0, y0, x1, y1 = np.clip(b, 0, [W, H, W, H]).astype(int)
                    allowed[y0:y1, x0:x1] = True
        a_ds = cv2.resize(allowed.astype(np.uint8), (Wd, Hd),
                          interpolation=cv2.INTER_AREA) > 0
        dts[i] = cv2.distanceTransform(
            (~a_ds).astype(np.uint8), cv2.DIST_L2, 3) * D
        ys, xs = np.nonzero(cv2.erode(m.astype(np.uint8), ker) > 0)
        if len(ys):
            sel = rng.choice(len(ys), min(n_cov, len(ys)), replace=False)
            cov_px[i, :len(sel)] = np.stack([xs[sel], ys[sel]], 1)
            cov_ok[i, :len(sel)] = True

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    tt = lambda a, d=torch.float32: torch.as_tensor(a, dtype=d, device=dev)

    gi_t = dRw = None
    if use_grasp:
        gm = np.asarray(grasp_mask)
        gi = np.where(gm[:-1] & gm[1:])[0]     # consecutive stable-grasp pairs
        use_grasp = len(gi) > 0
        if use_grasp:
            gi_t = tt(gi, torch.long)
            Rw = tt(wrist_R)
            dRw = Rw[gi_t + 1] @ Rw[gi_t].transpose(1, 2)   # wrist rotation deltas

    src = tt(src_pts)
    sil_idx = tt(rng.choice(len(src_pts), n_sil, replace=False), torch.long)
    R0 = tt(poses0[:, :3, :3])
    t0 = tt(poses0[:, :3, 3])
    Pcat = tt(np.concatenate([targets[i] for i in valid]))
    fid = tt(np.concatenate([np.full(len(targets[i]), i) for i in valid]),
             torch.long)
    vf = tt(np.array(valid), torch.long)
    dts_t = tt(dts).unsqueeze(1)
    cov_t, cov_ok_t = tt(cov_px), tt(cov_ok, torch.bool)

    # photometric term: LAB *chroma* (a,b) of each projected colored mesh point
    # vs the image at its projection. Differentiable via grid_sample on the
    # downsampled a,b planes (shares the silhouette subset + its projection grid).
    # Chroma (not luminance L) so it is robust to shading/exposure. Off unless a
    # textured mesh (src_cols) AND the frames dir are both available.
    w_photo = float(get("w_photo", 0.0))
    use_photo = w_photo > 0 and src_cols is not None and frames_dir is not None
    labs_t = pt_ab = None
    if use_photo:
        labs = np.zeros((T, 2, Hd, Wd), np.float32)
        for i in valid:
            img = cv2.imread(os.path.join(frames_dir, f"{i:05d}.jpg"))
            if img is None:
                continue
            img = cv2.resize(img, (Wd, Hd), interpolation=cv2.INTER_AREA)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
            labs[i] = lab[..., 1:].transpose(2, 0, 1) - 128.0     # a,b centered
        labs_t = tt(labs)
        pc = np.asarray(src_cols, np.float32)[sil_idx.cpu().numpy()]
        lab_pts = cv2.cvtColor((pc[None] * 255).astype(np.uint8),
                               cv2.COLOR_RGB2LAB)[0].astype(np.float32)
        pt_ab = tt(lab_pts[:, 1:] - 128.0)                        # [n_sil, 2]

    r = torch.zeros(T, 3, device=dev, requires_grad=True)
    dt_t = torch.zeros(T, 3, device=dev, requires_grad=True)
    ls = torch.full((3,), float(np.log(s0)), device=dev, requires_grad=True)
    opt = torch.optim.Adam([{"params": [r], "lr": 1e-2},
                            {"params": [dt_t], "lr": 3e-3},
                            {"params": [ls], "lr": 5e-3}])

    def poses_now():
        R = _so3_exp(r) @ R0
        return R, t0 + dt_t

    j_cat = w_cat = None
    for rnd in range(rounds):
        with torch.no_grad():               # refresh trimmed correspondences
            R, t = poses_now()
            s_np = ls.exp().cpu().numpy()
            tree = cKDTree(src_pts * s_np)
            js, ws = [], []
            for i in valid:
                q = (targets[i] - t[i].cpu().numpy()) @ R[i].cpu().numpy()
                d, j = tree.query(q, workers=-1)
                js.append(j)
                ws.append((d <= np.quantile(d, trim)).astype(np.float32))
            j_cat = tt(np.concatenate(js), torch.long)
            w_cat = tt(np.concatenate(ws))
        for it in range(inner):
            opt.zero_grad()
            R, t = poses_now()
            s = ls.exp()
            # depth term (mm): posed scaled samples vs their matched targets
            m_pts = (R[fid] @ (src[j_cat] * s).unsqueeze(-1)).squeeze(-1) \
                + t[fid]
            rn = (m_pts - Pcat).norm(dim=1) * 1000.0
            loss_d = (F.smooth_l1_loss(rn, torch.zeros_like(rn), beta=10.0,
                                       reduction="none") * w_cat).sum() \
                / w_cat.sum()
            # project silhouette subset for all valid frames
            Xs = torch.einsum("tab,nb->tna", R[vf], src[sil_idx] * s) \
                + (t0 + dt_t)[vf].unsqueeze(1)
            z = Xs[..., 2].clamp(min=0.1)
            u = Xs[..., 0] / z * fx + cx
            v = Xs[..., 1] / z * fy + cy
            # out-of-silhouette: bilinear DT sample (px)
            gx = ((u / D + 0.5) / Wd) * 2 - 1
            gy = ((v / D + 0.5) / Hd) * 2 - 1
            grid = torch.stack([gx, gy], -1).unsqueeze(1)
            dt_s = F.grid_sample(dts_t[vf], grid, align_corners=False,
                                 padding_mode="border").squeeze(1).squeeze(1)
            loss_s = F.smooth_l1_loss(dt_s, torch.zeros_like(dt_s), beta=5.0)
            # coverage: every confident mask pixel near some projected point
            uv = torch.stack([u, v], -1)
            dmin = torch.cdist(cov_t[vf], uv).min(dim=2).values
            cok = cov_ok_t[vf]
            loss_c = (dmin * cok).sum() / cok.sum()
            # priors: stay near image-based rotations; temporal smoothness on
            # the ABSOLUTE trajectory (the init rotations themselves jitter
            # frame-to-frame, so smoothing the deltas alone cannot help)
            loss_r = (r[vf] ** 2).sum(1).mean()
            tv = (t0 + dt_t)[vf] * 1000.0
            Rv = R[vf] * 40.5           # Frobenius ~ sqrt(2)*rad -> ~deg units
            loss_t = ((tv[2:] - 2 * tv[1:-1] + tv[:-2]) ** 2).mean() \
                + ((Rv[2:] - 2 * Rv[1:-1] + Rv[:-2]) ** 2).sum((1, 2)).mean()
            loss = w_depth * loss_d + w_sil * loss_s + w_cov * loss_c \
                + w_rot * loss_r + w_temp * loss_t
            # photometric: mean L2 LAB-CHROMA mismatch of IN-mask projected
            # points, sampled with the SAME `grid` as the DT term, in RAW LAB
            # units (no ÷ scaling). Magnitude balance: after grid_sample on the
            # downsampled planes + the inlier average, loss_p reads ~2-6 —
            # comparable to loss_d (~4 mm) and loss_s (~2 px). The old ÷10 pinned
            # it near ~0.2 so at w_photo=0.5 it was near-invisible; dropping the
            # ÷10 and raising w_photo to 4.0 (config) makes photometry the
            # DOMINANT per-frame azimuth signal, which is the point: depth +
            # silhouette are ~blind to azimuth on revolution / near-symmetric
            # objects. Restricting to inliers (dt_s<2px) avoids penalizing
            # overhang points (whose image color is background, not the object).
            loss_p_val = 0.0
            if use_photo:
                im_ab = F.grid_sample(labs_t[vf], grid, align_corners=False,
                                      padding_mode="border")[:, :, 0]  # [F,2,n]
                d_ab = im_ab.permute(0, 2, 1) - pt_ab.unsqueeze(0)     # [F,n,2]
                inlier = (dt_s < 2.0).float()
                # eps inside sqrt: sqrt'(0)=inf, and inf*inlier(0) -> NaN would
                # poison the whole backward when a point's color exactly matches
                # the image (baked colors / border padding). eps keeps the
                # gradient finite while preserving L2-chroma semantics.
                loss_p = (((d_ab ** 2).sum(-1) + 1e-8).sqrt() * inlier).sum() \
                    / inlier.sum().clamp(min=1.0)
                loss = loss + w_photo * loss_p
                loss_p_val = loss_p.item()
            # T3 grasp-rigidity: while stably grasped, the object's rotation
            # delta between consecutive frames should match the wrist's —
            # supplies the per-frame azimuth signal depth+silhouette can't
            # see on symmetric objects. 1600x brings the raw squared-Frobenius
            # delta mismatch (typically ~1e-3-1e-2 for near-aligned deltas)
            # into the same rough magnitude as the other terms (mm/px scale).
            loss_g_val = 0.0
            if use_grasp:
                dRo = R[gi_t + 1] @ R[gi_t].transpose(1, 2)
                loss_g = ((dRo - dRw) ** 2).sum((1, 2)).mean() * 1600.0
                loss = loss + w_grasp * loss_g
                loss_g_val = loss_g.item()
            loss.backward()
            opt.step()
            with torch.no_grad():
                ls.clamp_(np.log(s_lo), np.log(s_hi))
        log(f"joint refine round {rnd + 1}/{rounds}: depth={loss_d.item():.2f}mm "
            f"sil={loss_s.item():.2f}px cov={loss_c.item():.2f}px "
            + (f"photo={loss_p_val:.2f} " if use_photo else "")
            + (f"grasp={loss_g_val:.2f} " if use_grasp else "")
            + f"s_axes={np.round(ls.exp().detach().cpu().numpy(), 4)}")

    with torch.no_grad():
        R, t = poses_now()
        R, t = R.cpu().numpy(), t.cpu().numpy()
        s_axes = ls.exp().cpu().numpy().astype(float)
    poses = poses0.copy()
    resid = np.full(T, np.nan)
    tree = cKDTree(src_pts * s_axes)
    for i in valid:
        poses[i] = np.eye(4)
        poses[i][:3, :3], poses[i][:3, 3] = R[i], t[i]
        d, _ = tree.query((targets[i] - t[i]) @ R[i], workers=-1)
        keep = d <= np.quantile(d, trim)
        resid[i] = float(np.sqrt(np.mean(d[keep] ** 2)))
    return poses, s_axes, resid


def refine_object_poses(obj_verts, obj_faces, poses0, depth_dir, masks_dir, K,
                        opts=None, hand_boxes=None, hand_valid=None,
                        obj_colors=None, frames_dir=None,
                        grasp_mask=None, wrist_R=None):
    """Return (poses[T,4,4], stats) — per-frame rigid registration of the
    canonical mesh onto masked depth. Frame 0 initializes from poses0[0];
    frame t from the t-1 result (poses0[t] as re-init after gaps). `grasp_mask`
    [T] bool / `wrist_R` [T,3,3] (per-frame wrist global orientation) feed the
    T3 grasp-rigidity term in `_joint_refine` (config key w_grasp, default
    0.0 = off); both may be None when unavailable. Frames
    without usable depth keep their poses0. With `global_scale_refit`, the
    registration alternates with a shared-scale solve over all frames'
    pooled correspondences; stats["global_scale"] then carries the factor
    the CALLER must apply to the canonical verts (poses stay det=1 rigid)."""
    import cv2
    import trimesh
    from scipy.spatial import cKDTree

    o = opts or {}
    get = o.get if hasattr(o, "get") else lambda k, d: getattr(o, k, d)
    trim = float(get("trim", 0.8))
    iters = int(get("iters", 60))
    erode = int(get("erode_px", 5))
    n_tgt = int(get("max_points", 3000))
    n_src = int(get("mesh_samples", 20000))
    min_pts = int(get("min_points", 200))
    scale_refit = bool(get("global_scale_refit", False))
    passes = int(get("scale_refit_rounds", 2)) if scale_refit else 1
    rot_free = str(get("rotation", "free")) == "free"
    fg_band = float(get("fg_band_m", 0.15))

    mesh = trimesh.Trimesh(obj_verts, np.asarray(obj_faces, int), process=False)
    sampled = trimesh.sample.sample_surface(mesh, n_src, seed=0)
    src_pts = np.asarray(sampled[0])
    src_fidx = np.asarray(sampled[1])          # source face of each sampled point
    # per-surface-point color (mean of the sampled face's vertex colors), always
    # normalized to [0,1]; None for the depth-lift fallback mesh (no colors).
    # SAM-3D vertex colors arrive as uint8 [0,255] (real_perception.run_object_sam3d
    # -> stage3 "colors"); other/future producers may already be [0,1]. Normalize
    # defensively: treat any array whose max exceeds 1.5 as 0-255 and divide.
    src_cols = None
    if obj_colors is not None:
        oc = np.asarray(obj_colors, np.float32)
        if oc.size and float(oc.max()) > 1.5:
            oc = oc / 255.0
        fc = oc[np.asarray(mesh.faces)].mean(1)   # per-face mean color, [0,1]
        src_cols = fc[src_fidx]                # [n_src, 3] in 0-1
    frames_dir = frames_dir or None            # treat "" (no frames) as absent
    rng = np.random.default_rng(0)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode + 1,) * 2)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    T = len(poses0)
    targets = [None] * T
    for i in range(T):
        dp = os.path.join(depth_dir, f"{i:05d}.npy")
        mp = os.path.join(masks_dir, f"{i:05d}.npy")
        if not (os.path.exists(dp) and os.path.exists(mp)):
            continue
        g = np.load(dp).astype(np.float32)
        m = (cv2.erode(np.load(mp).astype(np.uint8), ker) > 0) \
            & (g > 0.25) & (g < 5.0)
        if m.sum() < min_pts:
            continue
        ys, xs = np.nonzero(m)
        z = g[ys, xs]
        # foreground band: real sensor depth bleeds background values into the
        # mask at the boundary (halo/misregistration — measured ~8% of the
        # ERODED kettle_N15 mask is >12cm off the object); reject them here or
        # they survive the trims and inflate the scale solve.
        if fg_band > 0:
            keep = np.abs(z - np.median(z)) < fg_band
            if keep.sum() >= min_pts:
                ys, xs, z = ys[keep], xs[keep], z[keep]
        P = np.stack([(xs - cx) / fx * z, (ys - cy) / fy * z, z], 1)
        if len(P) > n_tgt:
            P = P[rng.choice(len(P), n_tgt, replace=False)]
        targets[i] = P

    src_tree = cKDTree(src_pts)

    def registration_pass(s):
        src_s = src_pts * s
        tree = cKDTree(src_s)
        poses = poses0.copy()
        resid = np.full(T, np.nan)
        qs = []
        R, t = None, None
        for i, P in enumerate(targets):
            if P is None:
                R, t = None, None
                continue
            if R is None:                   # (re)start from the prior trajectory
                R, t = poses0[i][:3, :3].copy(), poses0[i][:3, 3].copy()
            if not rot_free:                # image-informed per-frame rotation
                R = poses0[i][:3, :3].copy()
            R, t, resid[i] = _icp(src_s, tree, P, R, t, trim, iters, rot_free)
            poses[i] = np.eye(4)
            poses[i][:3, :3], poses[i][:3, 3] = R, t
            if scale_refit:                 # this frame's cloud in canonical frame
                qs.append((P - t) @ R)
        return poses, resid, qs

    s = 1.0
    for p in range(passes):
        poses, resid, qs = registration_pass(s)
        if p == passes - 1 or not qs:
            break
        s_new = _solve_shared_scale(src_tree, src_pts, qs, s)
        if abs(s_new / s - 1.0) < 1e-3:
            break
        s = s_new

    # Anchor attitude search: the near-symmetric depth patch leaves azimuth
    # unobservable, so a wrong-but-locally-stable rotation basin can survive ICP.
    # Before the (local, gradient) joint refine, do a global discrete search over
    # azimuth hypotheses about all three canonical axes, scored numpy-only at the
    # single MOST-reliable frame (lowest ICP residual), and apply the winning
    # canonical-frame rotation G to the WHOLE track (so it stays temporally
    # consistent). G left-composed on the right: R_i <- R_i @ G.
    n_hyp = int(get("attitude_search", 0))
    if n_hyp > 1 and np.isfinite(resid).any():
        a = int(np.nanargmin(resid))
        hyps = _attitude_hypotheses(n_hyp)
        best_sc, best_G = np.inf, np.eye(3)
        R_a, t_a = poses[a][:3, :3], poses[a][:3, 3]
        # hoist the frame-constant anchor mask DT + LAB image out of the loop
        m_a = np.load(os.path.join(masks_dir, f"{a:05d}.npy")) > 0
        dt_a = cv2.distanceTransform((~m_a).astype(np.uint8), cv2.DIST_L2, 3)
        lab_a = None
        if src_cols is not None and frames_dir is not None:
            img_a = cv2.imread(os.path.join(frames_dir, f"{a:05d}.jpg"))
            if img_a is not None:
                lab_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2LAB).astype(np.float32)
        best_ch = best_dp = np.nan
        chromas = []
        for G in hyps:
            # fresh rng(0) per hypothesis -> identical point subset for a fair,
            # deterministic comparison
            sc, ch, dp = _score_hypothesis(
                src_pts, targets[a], dt_a, lab_a, K, R_a @ G, t_a, s, src_cols,
                np.random.default_rng(0), trim)
            chromas.append(ch)
            if sc < best_sc:
                best_sc, best_G, best_ch, best_dp = sc, G, ch, dp
        # Spread-gate: only trust (and apply) the winning rotation when the
        # chroma scores across hypotheses actually SPREAD — i.e. the object has
        # distinctive texture and one azimuth matches much better. Measured on
        # the mesh-controlled 6-clip HOT3D A/B: spread cleanly separates the one
        # trustworthy correction (bottle 31 LAB, chamfer 19.9->9.9mm) from
        # near-random ones on uniform/mismatched objects (vase/mug/cube 1-2 LAB;
        # masher 10 LAB actually hurt chamfer). Below SPREAD_MIN the depth-ICP
        # attitude is kept (identity) — the search self-disables, never harms.
        fin = np.array([c for c in chromas if np.isfinite(c)])
        spread = float(fin.max() - fin.min()) if fin.size else 0.0
        wants = not np.allclose(best_G, np.eye(3))
        applied = _attitude_apply(not wants, spread)
        if applied:
            for i in range(T):
                poses[i][:3, :3] = poses[i][:3, :3] @ best_G
        spread_msg = (f", chroma spread={spread:.2f} LAB over {fin.size} hyps"
                      if fin.size else "")
        if wants and not applied:
            verdict = f"identity (spread {spread:.2f} < {SPREAD_MIN} — non-discriminative)"
        elif applied:
            verdict = "correction applied"
        else:
            verdict = "identity (no change)"
        log(f"attitude search: {len(hyps)} hyps at anchor frame {a}, "
            f"best score={best_sc:.2f} (chroma={best_ch:.2f} LAB, "
            f"depth={best_dp:.1f}mm), azimuth {verdict}" + spread_msg)

    s_axes = None
    if bool(get("joint_silhouette", False)):
        poses, s_axes, resid = _joint_refine(
            src_pts, targets, masks_dir, K, poses, s, hand_boxes, hand_valid,
            get, src_cols=src_cols, frames_dir=frames_dir,
            grasp_mask=grasp_mask, wrist_R=wrist_R)
        s = 1.0                 # the per-axis scale subsumes the isotropic one

    ok = ~np.isnan(resid)
    n_reg = int(ok.sum())
    moved = np.linalg.norm(poses[ok][:, :3, 3] - poses0[ok][:, :3, 3], axis=1)
    stats = {"frames_registered": n_reg, "frames_total": int(T),
             "icp_resid_mm_med": float(np.nanmedian(resid) * 1000),
             "icp_resid_mm_p90": float(np.nanpercentile(resid, 90) * 1000),
             "moved_cm_med": float(np.median(moved) * 100) if n_reg else 0.0,
             "global_scale": float(s)}
    if s_axes is not None:
        stats["global_scale_axes"] = [float(x) for x in s_axes]
    log(f"object ICP: {n_reg}/{T} frames registered; residual "
        f"med={stats['icp_resid_mm_med']:.1f}mm p90={stats['icp_resid_mm_p90']:.1f}mm; "
        f"moved med={stats['moved_cm_med']:.1f}cm vs prior trajectory"
        + (f"; global scale refit s={s:.4f}" if scale_refit and s_axes is None
           else "")
        + (f"; joint per-axis scale {np.round(s_axes, 4)}"
           if s_axes is not None else ""))
    return poses, stats
