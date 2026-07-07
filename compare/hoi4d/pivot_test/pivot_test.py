"""b5-pivot kill test (v2) — GT-free foreground-anchored per-frame depth gauge.

Follow-up to the b6 kill test (../kill_test/RESULTS.md, verdict KILL): the
per-frame depth gauge that drives HOI foreground depth error and jitter lives
in the FOREGROUND (2-DOF oracle: 51% MAE cut, wiggle 4.0 -> 0.36 cm) but is
NOT recoverable from the background. Pivot question:

    Can the per-frame gauge be estimated WITHOUT GT, from foreground-intrinsic
    anchors?  (If yes -> "foreground-anchored gauge + GS photometric
    refinement" direction is GO.)

Every anchor sees GT only through the clip-global gauge (a_g, b_g) — exactly
the information the baseline already uses; all per-frame signal is GT-free.
All anchor kappas share ONE level convention: geometric mean 1 over the clip
(DC sits at the global gauge; DC recovery is scored separately, see below).

ANCHORS (kappa_t multiplies globally-gauged depth z_g = a_g*mono + b_g):
    rig-o : object visible-extent rigidity. Backproject eroded object-mask
            pixels from z_g; robust radius r_t = median |P - med(P)| must be
            constant for a rigid object => kappa_t ∝ 1/r_t.
    rig-h : same statistic on the hand mask (hand size constant but hand
            articulates — negative-control flavour).
    rig-u : extent of the hand∪object union (may be steadier through grasp).
    flow  : chained pairwise scale. Farneback flow between consecutive frames
            (both directions, object-bbox crop); correspondences inside the
            object mask; rho_t = ratio-of-medians of radii of corresponding
            point sets (mutually-visible points only => occlusion-robust);
            log-chained, geomean-normalised. Drift is measured, not hidden.
    hyb   : banded MAP smoother fusing rig-o absolute observations at
            high-visibility knot frames (obj px >= 0.75*clip max, rule-fixed)
            with flow relative increments elsewhere (lambda = 25, rule-fixed):
            minimise sum_knots (s_t - y_t)^2 + 25 * sum_t (s_{t+1}-s_t-d_t)^2
            in log-kappa. (Demoted from primary after the SYNTHETIC selftest
            showed the knot rule degenerates to all-knots on high-visibility
            clips, swamping the stronger flow signal — decided blind to real
            data.)
    eb    : PRIMARY (as eb-s). Empirical-Bayes fusion+shrinkage of rig-o and
            flow: cov of the two anchors estimates the shared gauge signal
            (GT-free); disagreement shrinks kappa -> 1 (= the baseline), which
            protects criterion (c) by construction. Chosen as primary on
            synthetic-selftest evidence only (flow-family corr 0.88 vs rig
            0.51; shrinkage safety), before any real-clip evaluation.
    *-s   : med-5 temporally smoothed variant.

CONTROLS / ORACLES (computed per evaluation region on its own pixels):
    smooth : evidence-free floor — per-frame depth shift toward the med-5 of
             the region's median-depth trace. Anchors must beat this.
    ko     : 1-DOF kappa-oracle — best per-frame pure scale (median GT/z_g).
             THE fair ceiling for scale anchors (review MUST-FIX #1).
    so     : shift-only oracle — best per-frame shift (median GT - z_g).
             With ko, decomposes which parameterisation the gauge error wants.
    oracle : 2-DOF per-frame affine on the region's own pixels (b6 oracle).

DIAGNOSTICS (make a KILL interpretable):
    cv_r_gt         std/mean of the SAME extent statistic on GT depth over the
                    SAME pixels — occlusion/truncation pollution of the extent
                    signal, independent of gauge error (rig-family ceiling).
    kappa spectrum  variance of ko-kappa split into med-5 band vs residual —
                    the fraction a temporally-coherent anchor can address at
                    all (review MUST-FIX #2). If mostly high-freq: structural.
    corr / corr_lp  corr_t(kappa_anchor, kappa_ko) raw and vs med5(kappa_ko)
                    (band-limited, threshold to beat: b6's failed 0.26).
    drift           linear slope of log kappa per 100 frames (flow vs ko).
    coverage        % frames each anchor actually measured (vs interpolated /
                    held); hyb knot count.

METRICS (protocol identical to ../kill_test): per-frame MAE vs GT and
wiggle = std_t(mean signed error), per region (obj / hand / union);
R = 1 - sum(MAE_variant)/sum(MAE_global); additionally MAE_dc0 / R_dc0 with
the clip-constant signed bias removed (DC-vs-shape decoupling: DC is
recoverable by one well-observed frame or a GS refinement stage downstream).

PRE-REGISTERED CRITERION v2 — frozen after the synthetic selftest and an
external design review, BEFORE any real-clip evaluation; kettle_N22 is the
machinery pilot clip (verdict must hold with and without it).
PRIMARY config: eb_s. All others are secondary/reported.
Over clips with R_obj(ko) >= 0.10 (others reported but excluded from ratios):
  GO iff (prim = eb_s)
    (a) pooled-headroom  sum_clips(MAE_g - MAE_prim) / sum_clips(MAE_g - MAE_ko)
        >= 0.5  AND  median per-clip R_obj(prim)/R_obj(ko) >= 0.5
    (b) median per-clip wiggle_obj(global)/wiggle_obj(prim) >= 2.5
        (med5-ko's reduction is reported as the coherent-anchor ceiling)
    (c) no clip's object OR union MAE degraded > 5% vs the global baseline
    (d) prim beats the smooth control on pooled object MAE.
  CONDITIONAL GO if (b), (c), (d) pass and (a) passes only after DC removal
    (pooled ratio on MAE_dc0 >= 0.5): shape is recovered, DC needs an
    absolute anchor stage — a design conclusion, not a kill.
  Else KILL, with diagnostics naming the failure mode (extent pollution vs
  flow drift vs high-freq-dominated gauge vs shift-shaped gauge).

Usage:
  python pivot_test.py --clip /workspace/hoi4d/clips/kettle_N22_S157_T1 [--selftest]
Outputs <clip>/pivot_test/{result.json, figure.png}. Reuses cached MoGe depth
at <clip>/kill_test/moge_depth.npy (CPU-only otherwise).
"""
import argparse, glob, json, os, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kill_test"))
from kill_test import huber_affine, load_gt_depth

ZMIN, ZMAX, STATIC_TOL = 0.25, 5.0, 0.05
DILATE_PX, ERODE_PX = 17, 2          # kill_test's 35/5 at full res, halved
MIN_REGION_PX = 150
HYB_LAMBDA = 25.0                    # rule: (sigma_rig/sigma_flow)^2 ~ (5%/1%)^2
HYB_KNOT_FRAC = 0.75                 # knot = frames with >= 0.75 * max obj px
PRIMARY = "eb_s"
SECONDARY = ("rig_o", "rig_o_s", "rig_h_s", "rig_u_s", "flow", "flow_s",
             "fuse", "eb", "hyb", "hyb_s")


# ------------------------------------------------------------------ loading
def load_clip(clip):
    mono = np.load(os.path.join(clip, "kill_test", "moge_depth.npy")).astype(np.float32)
    T, h, w = mono.shape
    n_rgb = len(glob.glob(os.path.join(clip, "rgb", "*.jpg")))
    T = min(T, n_rgb)
    mono = mono[:T]
    gt, hand, obj = [], [], []
    for i in range(T):
        g = load_gt_depth(clip, i)
        gt.append(cv2.resize(g, (w, h), interpolation=cv2.INTER_NEAREST))
        hm, om = [cv2.imread(os.path.join(clip, "masks", f"frame_{i:06d}_masks", f"{n}.png"),
                             cv2.IMREAD_GRAYSCALE) for n in ("hand", "object")]
        for src, dst in ((hm, hand), (om, obj)):
            m = (src > 127) if src is not None else np.zeros(g.shape, bool)
            dst.append(cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0)
    gt = np.stack(gt); hand = np.stack(hand); obj = np.stack(obj)
    K = np.load(os.path.join(clip, "intrin.npy")).astype(np.float64)
    W_full = cv2.imread(os.path.join(clip, "rgb", "000000.jpg")).shape[1]
    K2 = K * (w / W_full); K2[2, 2] = 1.0
    return mono, gt, hand, obj, K2


def load_gray(clip, T, shape_hw):
    h, w = shape_hw
    return [cv2.resize(cv2.imread(os.path.join(clip, "rgb", f"{i:06d}.jpg"),
                                  cv2.IMREAD_GRAYSCALE), (w, h)) for i in range(T)]


# ------------------------------------------------------------------ geometry
def backproject(ys, xs, z, K):
    X = (xs - K[0, 2]) / K[0, 0] * z
    Y = (ys - K[1, 2]) / K[1, 1] * z
    return np.stack([X, Y, z], 1)


def extent_radius(mask, depth, K, stride=2):
    """Robust visible-extent radius of masked pixels backprojected at `depth`."""
    ys, xs = np.nonzero(mask)
    if ys.size < MIN_REGION_PX:
        return np.nan, ys.size
    ys, xs = ys[::stride], xs[::stride]
    P = backproject(ys.astype(np.float64), xs.astype(np.float64), depth[ys, xs], K)
    c = np.median(P, 0)
    return float(np.median(np.linalg.norm(P - c, axis=1))), ys.size


def interp_nan(x):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.sum() == 0:
        return np.ones_like(x)
    return np.interp(np.arange(len(x)), np.nonzero(ok)[0], x[ok])


def med5(x):
    xp = np.pad(x, 2, mode="edge")
    return np.median(np.stack([xp[i:i + len(x)] for i in range(5)]), 0)


def gmean_norm(k):
    k = np.maximum(np.asarray(k, float), 1e-6)
    return k / np.exp(np.mean(np.log(k)))


def lin_slope_per100(x):
    """Linear-trend slope of a trace, per 100 frames."""
    t = np.arange(len(x), dtype=float)
    ok = np.isfinite(x)
    if ok.sum() < 4:
        return None
    return float(np.polyfit(t[ok], np.asarray(x, float)[ok], 1)[0] * 100)


# ------------------------------------------------------------------ anchors
def rigidity_kappa(masks, zg, K):
    """kappa_t from visible-extent rigidity (geomean-normalised); r trace; coverage."""
    r = np.array([extent_radius(masks[t], zg[t], K)[0] for t in range(len(zg))])
    ok = np.isfinite(r)
    kappa = np.full(len(r), np.nan)
    kappa[ok] = 1.0 / r[ok]
    return gmean_norm(interp_nan(kappa)), r, float(ok.mean())


def _pair_scale(gray0, gray1, mask0, mask1, valid0, valid1, zg0, zg1, K, kd):
    """One-direction pairwise relative scale (ratio-of-medians). None if unusable."""
    m0 = mask0 & valid0
    if m0.sum() < MIN_REGION_PX:
        return None, 0
    h, w = gray0.shape
    ys, xs = np.nonzero(m0)
    y0, y1 = max(ys.min() - 40, 0), min(ys.max() + 40, h)
    x0, x1 = max(xs.min() - 40, 0), min(xs.max() + 40, w)
    fl = cv2.calcOpticalFlowFarneback(gray0[y0:y1, x0:x1], gray1[y0:y1, x0:x1],
                                      None, 0.5, 3, 25, 3, 5, 1.2, 0)
    ys_s, xs_s = ys[::3], xs[::3]
    du = fl[ys_s - y0, xs_s - x0, 0]; dv = fl[ys_s - y0, xs_s - x0, 1]
    xd = np.rint(xs_s + du).astype(int); yd = np.rint(ys_s + dv).astype(int)
    inb = (xd >= 0) & (xd < w) & (yd >= 0) & (yd < h)
    # dilate the mask alone, THEN require validity at the target pixel —
    # dilating (mask & valid) would re-admit invalid zero-depth pixels
    m1 = (cv2.dilate(mask1.astype(np.uint8), kd) > 0) & valid1
    keep = inb.copy()
    keep[inb] = m1[yd[inb], xd[inb]]
    if keep.sum() < 80:
        return None, int(keep.sum())
    ys_s, xs_s, xd, yd = ys_s[keep], xs_s[keep], xd[keep], yd[keep]
    P0 = backproject(ys_s.astype(float), xs_s.astype(float), zg0[ys_s, xs_s], K)
    P1 = backproject(yd.astype(float), xd.astype(float), zg1[yd, xd], K)
    r0 = np.linalg.norm(P0 - np.median(P0, 0), axis=1)
    r1 = np.linalg.norm(P1 - np.median(P1, 0), axis=1)
    # paired median-of-log-ratios: pairing cancels the shared-shape variance;
    # the radius floor + log-median handle the near-centroid heavy tail
    big = (r0 > 0.02) & (r1 > 0.005)
    if big.sum() < 50:
        return None, int(keep.sum())
    return float(np.exp(np.median(np.log(r1[big] / r0[big])))), int(big.sum())


def flow_kappa(gray, obj_masks, valid, zg, K):
    """Bidirectionally chained pairwise relative scale from Farneback flow."""
    T = len(gray)
    d = np.zeros(T - 1)              # log kappa increments
    measured = np.zeros(T - 1, bool)
    n_pairs = np.zeros(T - 1, int)
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    for t in range(T - 1):
        rf, nf = _pair_scale(gray[t], gray[t + 1], obj_masks[t], obj_masks[t + 1],
                             valid[t], valid[t + 1], zg[t], zg[t + 1], K, kd)
        rb, nb = _pair_scale(gray[t + 1], gray[t], obj_masks[t + 1], obj_masks[t],
                             valid[t + 1], valid[t], zg[t + 1], zg[t], K, kd)
        logs = [np.log(r) for r in (rf,) if r] + [-np.log(r) for r in (rb,) if r]
        if logs:
            # rho = size ratio t->t+1; kappa increment is its negative
            d[t] = -float(np.mean(logs))
            measured[t] = True
            n_pairs[t] = max(nf, nb)
    kappa = np.exp(np.concatenate([[0.0], np.cumsum(d)]))
    return gmean_norm(kappa), d, measured, n_pairs


def hybrid_kappa(k_rig, r_valid, obj_px, d_flow, flow_measured):
    """MAP smoother in log-kappa: rig-o absolute obs at high-visibility knots +
    flow increments as the dynamics. Banded normal equations, solved dense."""
    T = len(k_rig)
    knots = r_valid & (obj_px >= HYB_KNOT_FRAC * obj_px.max())
    if knots.sum() < 2:
        return gmean_norm(np.exp(np.concatenate([[0.0], np.cumsum(d_flow)]))), int(knots.sum())
    y = np.log(np.maximum(k_rig, 1e-6))
    A = np.zeros((T, T)); b = np.zeros(T)
    for t in np.nonzero(knots)[0]:
        A[t, t] += 1.0
        b[t] += y[t]
    lam = HYB_LAMBDA
    for t in range(T - 1):
        dt = d_flow[t] if flow_measured[t] else 0.0
        w = lam if flow_measured[t] else lam * 0.1
        A[t, t] += w; A[t + 1, t + 1] += w
        A[t, t + 1] -= w; A[t + 1, t] -= w
        b[t] -= w * dt; b[t + 1] += w * dt
    s = np.linalg.solve(A + 1e-9 * np.eye(T), b)
    return gmean_norm(np.exp(s)), int(knots.sum())


def eb_fuse(k1, k2):
    """GT-free empirical-Bayes fusion+shrinkage of two noisy gauge anchors.

    Log-domain: x_i = s + n_i, shared signal s, independent noises (rig noise =
    mask/visibility truncation; flow noise = flow-error drift). cov(x1,x2)
    estimates var(s); precisions weight the combination; the posterior shrinks
    toward kappa=1, so disagreeing anchors fall back to the global baseline."""
    x1, x2 = np.log(np.maximum(k1, 1e-6)), np.log(np.maximum(k2, 1e-6))
    x1, x2 = x1 - x1.mean(), x2 - x2.mean()
    s2 = max(float(np.cov(x1, x2)[0, 1]), 0.0)
    n1 = max(float(x1.var()) - s2, 1e-8)
    n2 = max(float(x2.var()) - s2, 1e-8)
    xb = (x1 / n1 + x2 / n2) / (1 / n1 + 1 / n2)
    nc = 1.0 / (1 / n1 + 1 / n2)
    return np.exp(s2 / (s2 + nc) * xb)


# ------------------------------------------------------------------ analysis
def analyze(clip_dir, mono, gt, hand, obj, K, gray):
    T = len(mono)
    valid = (gt > ZMIN) & (gt < ZMAX) & (mono > 0) & np.isfinite(mono)
    dyn = hand | obj
    gt_nan = np.where(valid, gt, np.nan)
    static = np.abs(gt_nan - np.nanmedian(gt_nan, 0)[None]) < STATIC_TOL
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DILATE_PX + 1,) * 2)
    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ERODE_PX + 1,) * 2)

    regions = {"obj": [], "hand": [], "union": []}
    pooled_x, pooled_y = [], []
    for t in range(T):
        bg_t = valid[t] & ~(cv2.dilate(dyn[t].astype(np.uint8), kd) > 0) & static[t]
        xb = mono[t][bg_t][::3]
        pooled_x.append(xb[::4]); pooled_y.append(gt[t][bg_t][::3][::4])
        for name, m in (("obj", obj[t]), ("hand", hand[t]), ("union", dyn[t])):
            regions[name].append((cv2.erode(m.astype(np.uint8), ke) > 0) & valid[t])
    g = huber_affine(np.concatenate(pooled_x), np.concatenate(pooled_y))
    if g is None:
        raise RuntimeError("global bg fit failed")
    ag, bg_ = g
    zg = ag * mono + bg_

    # 2-DOF oracle per region (per-frame affine on the region's own pixels)
    oracle = {name: [huber_affine(mono[t][regions[name][t]], gt[t][regions[name][t]])
                     for t in range(T)] for name in regions}

    # ------------------------------------------------- anchors (GT-free)
    obj_e = regions["obj"]
    k_rig_o, r_obj, cov_o = rigidity_kappa(obj_e, zg, K)
    k_rig_h, _, cov_h = rigidity_kappa(regions["hand"], zg, K)
    k_rig_u, _, cov_u = rigidity_kappa(regions["union"], zg, K)
    k_flow, d_flow, flow_meas, n_pairs = flow_kappa(gray, obj, valid, zg, K)
    obj_px = np.array([m.sum() for m in obj_e], float)
    r_valid = np.isfinite(r_obj)
    k_hyb, n_knots = hybrid_kappa(k_rig_o, r_valid, obj_px, d_flow, flow_meas)
    kappas = {"rig_o": k_rig_o, "rig_o_s": med5(k_rig_o),
              "rig_h": k_rig_h, "rig_h_s": med5(k_rig_h),
              "rig_u_s": med5(k_rig_u),
              "flow": k_flow, "flow_s": med5(k_flow),
              "fuse": med5(gmean_norm(np.sqrt(np.maximum(k_rig_o, 1e-6)
                                              * np.maximum(k_flow, 1e-6)))),
              "eb": eb_fuse(k_rig_o, k_flow),
              "eb_s": eb_fuse(med5(k_rig_o), med5(k_flow)),
              "hyb": k_hyb, "hyb_s": med5(k_hyb)}

    # ------------------------------------------- per-region metric engine
    def region_metrics(name):
        zm_trace = np.array([np.median(zg[t][regions[name][t]])
                             if regions[name][t].sum() >= MIN_REGION_PX else np.nan
                             for t in range(T)])
        smooth_shift = med5(interp_nan(zm_trace)) - interp_nan(zm_trace)

        def frame_preds(t):
            m = regions[name][t]
            if m.sum() < MIN_REGION_PX:
                return None
            xf, yf = mono[t][m], gt[t][m]
            zgt = ag * xf + bg_
            entries = {"global": zgt, "smooth": zgt + smooth_shift[t],
                       "ko": float(np.median(yf / np.maximum(zgt, 1e-6))) * zgt,
                       "so": zgt + float(np.median(yf - zgt))}
            o = oracle[name][t]
            if o is not None:
                entries["oracle"] = o[0] * xf + o[1]
            for v, k in kappas.items():
                entries[v] = k[t] * zgt
            return entries, yf
        # pass 1: per-frame mae/bias + clip DC per variant
        rows = {}
        for t in range(T):
            fp = frame_preds(t)
            if fp is None:
                continue
            entries, yf = fp
            for v, pred in entries.items():
                res = pred - yf
                rows.setdefault(v, []).append(
                    (t, float(np.mean(np.abs(res))), float(np.mean(res))))
        if "global" not in rows:
            return {}
        dc = {v: float(np.mean([b for _, _, b in rr])) for v, rr in rows.items()}
        # pass 2: DC-removed MAE
        mae0 = {v: [] for v in rows}
        for t in range(T):
            fp = frame_preds(t)
            if fp is None:
                continue
            entries, yf = fp
            for v, pred in entries.items():
                mae0[v].append(float(np.mean(np.abs(pred - dc[v] - yf))))
        mg = {t: mae for t, mae, _ in rows["global"]}
        out = {}
        for v, rr in rows.items():
            rr = [r for r in rr if r[0] in mg]
            mae = np.array([r[1] for r in rr]); bias = np.array([r[2] for r in rr])
            base = np.array([mg[r[0]] for r in rr])
            m0 = np.array(mae0[v][:len(rr)])
            out[v] = {"MAE_cm": float(mae.mean() * 100),
                      "R_MAE": float(1 - mae.sum() / base.sum()),
                      "MAE_dc0_cm": float(m0.mean() * 100),
                      "wiggle_cm": float(bias.std() * 100),
                      "frames": len(rr)}
        gsum = sum(r[1] for r in rows["global"])
        for v in out:
            out[v]["R_dc0"] = float(1 - np.sum(mae0[v]) / np.sum(mae0["global"]))
        return out

    metrics = {name: region_metrics(name) for name in regions}

    # ------------------------------------------------------- diagnostics
    r_gt = np.array([extent_radius(obj_e[t], gt[t], K)[0] for t in range(T)])
    ok_r = np.isfinite(r_gt)
    cv_r_gt = float(r_gt[ok_r].std() / r_gt[ok_r].mean()) if ok_r.sum() > 3 else None
    r_gt_u = np.array([extent_radius(regions["union"][t], gt[t], K)[0] for t in range(T)])
    r_gt_u = r_gt_u[np.isfinite(r_gt_u)]
    cv_r_gt_u = float(r_gt_u.std() / r_gt_u.mean()) if r_gt_u.size > 3 else None

    # ko-kappa trace on the object (the 1-DOF oracle gauge)
    k_ko = np.full(T, np.nan)
    for t in range(T):
        m = obj_e[t]
        if m.sum() >= MIN_REGION_PX:
            zgt = zg[t][m]
            k_ko[t] = float(np.median(gt[t][m] / np.maximum(zgt, 1e-6)))
    okk = np.isfinite(k_ko)
    k_ko_i = interp_nan(k_ko)
    k_ko_lp = med5(k_ko_i)
    hf = k_ko_i - k_ko_lp
    var_tot = float(k_ko_i[okk].var())
    spectrum = {"var_total": var_tot,
                "lowfreq_frac": float(1 - hf[okk].var() / var_tot) if var_tot > 0 else None}
    corr_kappa, corr_kappa_lp = {}, {}
    for v, k in kappas.items():
        if okk.sum() > 3:
            corr_kappa[v] = float(np.corrcoef(k[okk], k_ko[okk])[0, 1])
            corr_kappa_lp[v] = float(np.corrcoef(k[okk], k_ko_lp[okk])[0, 1])

    res = {"clip": os.path.basename(clip_dir.rstrip("/")), "frames": T,
           "global_gauge": {"a": ag, "b": bg_},
           "diag": {"cv_r_gt_obj": cv_r_gt, "cv_r_gt_union": cv_r_gt_u,
                    "ko_spectrum": spectrum,
                    "corr_kappa_ko": corr_kappa, "corr_kappa_ko_lp": corr_kappa_lp,
                    "drift_per100": {"flow": lin_slope_per100(np.log(np.maximum(k_flow, 1e-6))),
                                     "ko": lin_slope_per100(np.log(np.maximum(k_ko_i, 1e-6)))},
                    "coverage": {"rig_o": cov_o, "rig_h": cov_h, "rig_u": cov_u,
                                 "flow": float(flow_meas.mean()), "hyb_knots": n_knots},
                    "flow_pairs_median": int(np.median(n_pairs)) if len(n_pairs) else 0,
                    "obj_px_median": float(np.median(obj_px))},
           "metrics": metrics}
    traces = {"k_ko": k_ko_i, "r_obj": r_obj, "r_gt": r_gt, **kappas}
    return res, traces


# ------------------------------------------------------------------ selftest
def selftest(clip):
    """Synthesize mono from GT with a known per-frame gauge; the oracle/ko
    machinery must recover it; anchor tracking is printed (informative)."""
    mono, gt, hand, obj, K = load_clip(clip)
    T = len(gt)
    rng = np.random.default_rng(0)
    a_true = 1.0 + 0.15 * np.sin(np.arange(T) / 4.0)
    b_true = 0.06 * np.cos(np.arange(T) / 6.0)
    syn = np.zeros_like(gt)
    for t in range(T):
        m = gt[t] > 0
        syn[t][m] = (gt[t][m] - b_true[t]) / a_true[t]
        syn[t][m] *= 1 + rng.normal(0, 0.01, m.sum())
        syn[t][m] += rng.normal(0, 0.008, m.sum())
    gray = load_gray(clip, T, gt.shape[1:])
    res, traces = analyze(clip, syn, gt, hand, obj, K, gray)
    zm = np.array([np.median(syn[t][obj[t] & (syn[t] > 0)]) for t in range(T)])
    k_true = (a_true * zm + b_true) / (res["global_gauge"]["a"] * zm + res["global_gauge"]["b"])
    for v in ("rig_o", "rig_o_s", "rig_u_s", "flow", "flow_s", "fuse", "eb", "eb_s", "hyb"):
        c = np.corrcoef(traces[v], k_true)[0, 1]
        print(f"[selftest] corr({v:7s}, kappa_true) = {c:5.3f}   "
              f"R_obj={res['metrics']['obj'][v]['R_MAE']:6.3f}")
    print(f"[selftest] cv(r_gt) = {res['diag']['cv_r_gt_obj']:.4f} (extent noise floor)  "
          f"knots={res['diag']['coverage']['hyb_knots']}")
    ro = res["metrics"]["obj"]["oracle"]["R_MAE"]
    rk = res["metrics"]["obj"]["ko"]["R_MAE"]
    print(f"[selftest] oracle(2dof) R_obj = {ro:.3f}, ko(1dof) R_obj = {rk:.3f} "
          f"(expect ~0.88 / slightly lower: 8mm+1% noise floor over ~9cm gauge error)")
    # ko < oracle here is EXPECTED: the synthetic's b_t = +-6 cm shift is
    # outside the pure-scale model class (the review's 1-vs-2-DOF point).
    assert ro > 0.85 and rk > 0.6, "SELFTEST FAILED: oracle machinery broken"
    print("[selftest] PASS (oracle machinery ok)")


# ------------------------------------------------------------------ figure
def make_figure(res, traces, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = res["frames"]; ts = np.arange(T)
    fig, axes = plt.subplots(3, 1, figsize=(9, 8))
    axes[0].plot(ts, traces["k_ko"], color="#1baf7a", ls="--", label="kappa ko (1-DOF oracle)")
    axes[0].plot(ts, traces["rig_o"], color="#2a78d6", lw=.8, alpha=.5, label="rig-o")
    axes[0].plot(ts, traces["hyb"], color="#eda100", lw=1.0, alpha=.7, label="hyb")
    axes[0].plot(ts, traces["flow_s"], color="#d64550", lw=1.4, label="flow-s (PRIMARY*)")
    axes[0].plot(ts, traces["eb_s"], color="#8145d6", label="eb-s")
    axes[0].axhline(1.0, color="#898781", lw=1)
    axes[0].set_ylabel("kappa_t"); axes[0].legend(fontsize=7, ncol=3)
    cv = res["diag"]["cv_r_gt_obj"]
    ck = res["diag"]["corr_kappa_ko"].get("flow_s")
    axes[0].set_title(f"{res['clip']}: GT-free gauge anchors vs ko oracle "
                      f"(cv(r_gt)={cv:.3f}, corr flow-s={ck:.2f}, "
                      f"lowfreq={res['diag']['ko_spectrum']['lowfreq_frac']:.2f})")
    axes[1].plot(ts, traces["r_obj"] * 100, color="#2a78d6", label="r_t (globally-gauged mono)")
    axes[1].plot(ts, traces["r_gt"] * 100, color="#1baf7a", label="r_t (GT depth, same pixels)")
    axes[1].set_xlim(axes[0].get_xlim())
    axes[1].set_ylabel("object extent radius (cm)"); axes[1].legend(fontsize=8)
    axes[1].set_xlabel("frame")
    m = res["metrics"]["obj"]
    labels = [("global", "#898781"), ("smooth", "#b5b2ac"), ("rig_o_s", "#2a78d6"),
              ("hyb", "#eda100"), ("eb_s", "#8145d6"), ("flow_s", "#d64550"),
              ("ko", "#7ad6b8"), ("oracle", "#1baf7a")]
    bars = [m[v]["MAE_cm"] for v, _ in labels if v in m]
    names = [f"{v}\nR={m[v]['R_MAE']:.2f}\nw={m[v]['wiggle_cm']:.2f}" for v, _ in labels if v in m]
    cols = [c for v, c in labels if v in m]
    axes[2].bar(range(len(bars)), bars, color=cols)
    axes[2].set_xticks(range(len(bars))); axes[2].set_xticklabels(names, fontsize=7)
    axes[2].set_ylabel("object MAE (cm)")
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(args.clip)
        return
    mono, gt, hand, obj, K = load_clip(args.clip)
    gray = load_gray(args.clip, len(mono), mono.shape[1:])
    res, traces = analyze(args.clip, mono, gt, hand, obj, K, gray)
    outdir = os.path.join(args.clip, "pivot_test")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "result.json"), "w") as f:
        json.dump(res, f, indent=1)
    try:
        make_figure(res, traces, os.path.join(outdir, "figure.png"))
    except Exception as e:
        print("figure failed:", e)
    print(json.dumps({"clip": res["clip"], "diag": res["diag"],
                      "obj": res["metrics"]["obj"]}, indent=1))


if __name__ == "__main__":
    main()
