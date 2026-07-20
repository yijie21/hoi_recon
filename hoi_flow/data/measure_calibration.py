"""Measure the REAL per-frame error of our coarse HOI stack vs HOT3D mocap GT, so a
flow-matching refiner can be trained on *calibrated* corruptions of clean GT rather than
on guessed noise. This script produces `calibration.json`; `corrupt.py` consumes it.

WHY each measurement is shaped the way it is
--------------------------------------------
* Object est poses pose a SAM-3D mesh; GT poses pose the HOT3D eval GLB. The two live in
  DIFFERENT canonical frames, so pose matrices are not directly comparable. We first solve
  a per-clip constant alignment A (4x4), robustified by dropping flip frames, then measure
  the residual per frame. Sanity: median translation after alignment must be ~5-25 mm
  (leaderboard chamfer scale), not metres.
* Translation error is decomposed ALONG the camera view-ray (depth, weakly constrained by
  a single view) vs PERPENDICULAR (image plane, well constrained). Depth dominates.
* Rotation error separates a smooth AR-correlated wobble from FLIP EVENTS (symmetric
  objects jumping >~60 deg and later reverting) — the two need different corruption models.
* Noise temporal structure is fit as AR(1)+white: lag-1 & lag-2 autocorrelation identify
  the slow (AR) vs fast (white/jitter) split (rho_ar = r2/r1, var_ar/var = r1^2/r2).
* Hand: est is MANO (778v), GT is UmeTrack (diff topology) -> correspondence-free. We take
  the per-frame translation offset (est centroid - matched GT centroid) as placement error
  and the post-offset one-sided chamfer as the articulation residual.
* MANO GT reliability: HOT3D stores per-frame MANO (thetas + wrist_xform). We FK them and
  compare to the UmeTrack GT landmarks. Raw vs translation-aligned vs rigid-aligned distances
  disentangle a global-frame convention gap from true articulation error.

Run (env rc5090):  python -m hoi_flow.data.measure_calibration
"""
import glob
import json
import os
import re
import sys

import numpy as np

RUNS = "/workspace/code/hoi_recon/render_and_compare/runs"
HOT3D = "/workspace/datasets/hot3d"
OUT = os.path.join(os.path.dirname(__file__), "calibration.json")

# clip -> object arm shipped. 5x fpauto (learned core), potato_masher is icpjgr (registration).
CLIPS = {
    "bottle_bbq_002034": "fpauto",
    "mug_white_001970": "fpauto",
    "vase_002500": "fpauto",
    "spatula_red_001990": "fpauto",
    "puzzle_toy_001964": "fpauto",
    "potato_masher_002349": "icpjgr",
}
FPAUTO = [c for c, a in CLIPS.items() if a == "fpauto"]
FLIP_DEG = 60.0


# ----------------------------------------------------------------------------- helpers
def rc_dir(clip):
    obj, num = re.match(r"(.+)_(\d+)$", clip).groups()
    return f"{HOT3D}/rc_input_{num}_{obj}"


def geodesic_deg(R):  # [N,3,3] -> [N] rotation angle
    tr = np.trace(R, axis1=1, axis2=2)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def align_A(Test, Tgt, keep):
    """A = argmin_A sum_t ||Test_t @ A - Tgt_t||_F over frames `keep` (closed form)."""
    Re, te = Test[:, :3, :3], Test[:, :3, 3]
    tg = Tgt[:, :3, 3]
    M = np.einsum("tij,tik->jk", Re[keep], Tgt[keep, :3, :3])  # sum Re^T Rg
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    Ra = U @ D @ Vt
    ta = np.einsum("tji,tj->ti", Re[keep], (tg - te)[keep]).mean(0)  # mean Re^T (tg-te)
    A = np.eye(4)
    A[:3, :3], A[:3, 3] = Ra, ta
    return A


def robust_align(Test, Tgt, iters=5):
    keep = np.arange(len(Test))
    for _ in range(iters):
        A = align_A(Test, Tgt, keep)
        Ta = Test @ A
        rerr = geodesic_deg(np.einsum("tji,tjk->tik", Tgt[:, :3, :3], Ta[:, :3, :3]))
        nk = np.where(rerr < FLIP_DEG)[0]
        keep = nk if len(nk) >= 10 else np.where(rerr < np.percentile(rerr, 60))[0]
    return A, keep


def autocorr(x, k):
    x = x - x.mean()
    d = np.sum(x * x)
    return float(np.sum(x[k:] * x[:-k]) / d) if d > 1e-12 and len(x) > k else 0.0


def ar_white(x):
    """Identify AR(1)+white from a scalar series. Returns the generative knobs
    (sigma_ar, rho_ar, sigma_white) plus diagnostics (sigma, rho1, jitter)."""
    x = np.asarray(x, float)
    n = len(x)
    sigma = float(np.std(x))
    jit = float(np.std(np.diff(x))) if n > 1 else 0.0
    if n < 6 or sigma < 1e-9:
        return dict(sigma=sigma, rho1=0.0, rho2=0.0, jitter=jit,
                    rho_ar=0.0, sigma_ar=0.0, sigma_white=sigma)
    r1, r2 = autocorr(x, 1), autocorr(x, 2)
    if r1 > 0.05 and 0.02 < r2 < r1:          # AR+white separable via lag1/lag2
        rho_ar = float(np.clip(r2 / r1, 0.0, 0.995))
        f = float(np.clip(r1 / rho_ar if rho_ar > 1e-6 else 0.0, 0.0, 1.0))
    elif r1 > 0.05:                            # measurable lag1 only -> treat as pure AR(1)
        rho_ar, f = float(np.clip(r1, 0.0, 0.995)), 1.0
    else:                                      # no correlation -> pure white
        rho_ar, f = 0.0, 0.0
    return dict(sigma=sigma, rho1=r1, rho2=r2, jitter=jit, rho_ar=rho_ar,
                sigma_ar=sigma * np.sqrt(f), sigma_white=sigma * np.sqrt(1 - f))


def series_stats(x):
    x = np.asarray(x, float)
    d = ar_white(x - x.mean())
    d["bias"] = float(x.mean())
    return {k: round(v, 4) for k, v in d.items()}


def pool_series_stats(series_list):
    """Frame-weighted pool. bias/sigma come from the concatenation (so cross-clip spread
    is captured); rho_ar and the AR fraction f=var_ar/var are frame-weighted; then
    sigma_ar/sigma_white are RE-DERIVED from the pooled sigma and f so that the
    generative pair still reconstructs the pooled sigma (avoids the Jensen gap that
    averaging the per-clip sqrt-sigmas would introduce)."""
    allx = np.concatenate(series_list)
    w = np.array([len(s) for s in series_list], float)
    per = [ar_white(s - s.mean()) for s in series_list]
    def wavg(k):
        return float(np.average([p[k] for p in per], weights=w))
    sigma = float(allx.std())
    f = float(np.average([(p["sigma_ar"] / p["sigma"]) ** 2 if p["sigma"] > 1e-9 else 0.0
                          for p in per], weights=w))
    f = min(max(f, 0.0), 1.0)
    return {"bias": round(float(allx.mean()), 4),
            "sigma": round(sigma, 4),
            "rho_ar": round(wavg("rho_ar"), 4),
            "sigma_ar": round(sigma * np.sqrt(f), 4),
            "sigma_white": round(sigma * np.sqrt(1 - f), 4),
            "rho1": round(wavg("rho1"), 4),
            "jitter": round(wavg("jitter"), 4)}


# ---------------------------------------------------------------------------- objects
def measure_object(clip, arm):
    Test = np.load(f"{RUNS}/hot3d_{clip}_{arm}/stage8_eval/pseudo_gt.npz")["obj_poses"].astype(float)
    Tgt = np.load(f"{rc_dir(clip)}/gt_target.npz")["poses"].astype(float)
    T = min(len(Test), len(Tgt))
    Test, Tgt = Test[:T], Tgt[:T]
    A, keep = robust_align(Test, Tgt)
    Ta = Test @ A
    e = (Ta[:, :3, 3] - Tgt[:, :3, 3]) * 1000.0                # camera-frame error, mm
    rhat = Tgt[:, :3, 3] / np.linalg.norm(Tgt[:, :3, 3], axis=1, keepdims=True)
    r0 = rhat.mean(0); r0 /= np.linalg.norm(r0)                # fixed perp basis from mean ray
    u1 = np.cross(r0, [0, 1.0, 0]); u1 /= np.linalg.norm(u1)
    u2 = np.cross(r0, u1)
    along = np.einsum("ti,ti->t", e, rhat)                     # scalar depth error series
    p1, p2 = e @ u1, e @ u2                                    # 2 perpendicular series
    rerr = geodesic_deg(np.einsum("tji,tjk->tik", Tgt[:, :3, :3], Ta[:, :3, :3]))

    is_flip = rerr > FLIP_DEG
    # flip events = contiguous runs of flipped frames
    durs, mags = [], []
    i = 0
    while i < T:
        if is_flip[i]:
            j = i
            while j < T and is_flip[j]:
                j += 1
            durs.append(j - i); mags.append(float(np.median(rerr[i:j])))
            i = j
        else:
            i += 1
    n_events = len(durs)
    trans_mm = np.linalg.norm(e, axis=1)

    perp_pooled = pool_series_stats([p1, p2])
    return dict(
        n_frames=int(T), arm=arm, kept_align_frames=int(len(keep)),
        align_trans_med_mm=round(float(np.median(trans_mm)), 2),
        align_trans_p90_mm=round(float(np.percentile(trans_mm, 90)), 2),
        trans_along=series_stats(along),
        trans_perp={**{k: round(v, 4) for k, v in perp_pooled.items()}},
        rot_smooth=series_stats(rerr[~is_flip]) if (~is_flip).sum() > 6 else series_stats(rerr),
        rot_all_med_deg=round(float(np.median(rerr)), 2),
        flip=dict(
            event_rate=round(n_events / T, 5),                # onsets per frame
            state_frac=round(float(is_flip.mean()), 4),       # fraction of frames flipped
            mag_deg_mean=round(float(np.mean(mags)), 2) if mags else 0.0,
            mag_deg_std=round(float(np.std(mags)), 2) if mags else 0.0,
            dur_mean=round(float(np.mean(durs)), 2) if durs else 0.0,
            dur_std=round(float(np.std(durs)), 2) if durs else 0.0,
            dur_max=int(max(durs)) if durs else 0,
            durations=[int(d) for d in durs],
        ),
        _series=dict(along=along, p1=p1, p2=p2, rot_smooth=rerr[~is_flip]),
    )


# ------------------------------------------------------------------------------- hands
def measure_hand(clip):
    est = np.load(f"{RUNS}/hot3d_{clip}_icpjgr/hand_reproj_opt/out.npz")
    hv, vis = est["hand_verts"].astype(float), est["visible"]
    gt = np.load(f"{rc_dir(clip)}/gt_hands.npz", allow_pickle=True)["verts"]
    T = min(len(hv), len(gt))
    e, artic = [], []
    valid = []
    for t in range(T):
        hands = gt[t]
        if not vis[t] or hands.ndim != 3 or hands.shape[0] == 0:
            valid.append(False); e.append([0, 0, 0]); continue
        ec = hv[t].mean(0)
        # match nearest GT hand by centroid
        cds = [np.linalg.norm(ec - g.mean(0)) for g in hands]
        g = hands[int(np.argmin(cds))]
        off = ec - g.mean(0)                                  # translation placement error (m)
        e.append(off * 1000.0); valid.append(True)
        # articulation residual: post-offset one-sided chamfer est->gt (subsample for speed)
        est_v = hv[t] - off
        idx = np.random.default_rng(0).choice(len(est_v), min(300, len(est_v)), replace=False)
        d = np.linalg.norm(est_v[idx, None, :] - g[None, :, :], axis=2).min(1)
        artic.append(float(np.median(d) * 1000.0))
    valid = np.array(valid); e = np.array(e)
    if valid.sum() < 6:
        return None
    ev = e[valid]
    # view-ray from matched GT centroids ~ camera forward; use est centroid direction per frame
    cdir = hv[:T][valid].mean(1)
    rhat = cdir / np.linalg.norm(cdir, axis=1, keepdims=True)
    r0 = rhat.mean(0); r0 /= np.linalg.norm(r0)
    u1 = np.cross(r0, [0, 1.0, 0]); u1 /= np.linalg.norm(u1)
    u2 = np.cross(r0, u1)
    along = np.einsum("ti,ti->t", ev, rhat)
    p1, p2 = ev @ u1, ev @ u2
    return dict(
        n_frames=int(valid.sum()),
        offset_med_mm=round(float(np.median(np.linalg.norm(ev, axis=1))), 2),
        trans_along=series_stats(along),
        trans_perp={k: round(v, 4) for k, v in pool_series_stats([p1, p2]).items()},
        articulation_resid_mm=dict(
            median=round(float(np.median(artic)), 2),
            mean=round(float(np.mean(artic)), 2),
            p90=round(float(np.percentile(artic, 90)), 2)),
        _series=dict(along=along, p1=p1, p2=p2),
    )


# --------------------------------------------------------------------- MANO reliability
def measure_mano(n_segs=20):
    import torch
    import smplx
    md = "/workspace/code/hoi_recon/render_and_compare/checkpoints/mano"
    scratch = "/tmp/claude-1002/-workspace-code-hoi-recon/cb99b7a6-8766-4fc9-a273-b7123285115d/scratchpad/mano_models"
    os.makedirs(scratch, exist_ok=True)
    dst = os.path.join(scratch, "MANO_RIGHT.pkl")
    if not os.path.exists(dst):
        os.symlink(os.path.join(md, "MANO_RIGHT_np.pkl"), dst)
    layer = smplx.MANO(dst, is_rhand=True, use_pca=True, num_pca_comps=15,
                       flat_hand_mean=False, batch_size=1)
    TIPS = [744, 320, 443, 554, 671]
    MAP = [16, 17, 18, 19, 20, 0, 14, 15, 1, 2, 3, 4, 5, 6, 10, 11, 12, 7, 8, 9]

    def fk(hm):  # hm [T,21] right hand: theta[15] + global_orient[3] + transl[3]
        T = hm.shape[0]
        out = layer(betas=torch.zeros(T, 10),
                    global_orient=torch.tensor(hm[:, 15:18], dtype=torch.float32),
                    hand_pose=torch.tensor(hm[:, :15], dtype=torch.float32),
                    transl=torch.tensor(hm[:, 18:21], dtype=torch.float32), return_verts=True)
        lm = torch.cat([out.joints, out.vertices[:, TIPS, :]], 1)
        return lm[:, MAP, :].detach().numpy()

    def kabsch_resid(P, Q):
        Pc, Qc = P - P.mean(0), Q - Q.mean(0)
        U, _, Vt = np.linalg.svd(Pc.T @ Qc)
        R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
        return np.linalg.norm((R @ Pc.T).T + Q.mean(0) - Q, axis=1)

    segs = sorted(glob.glob(f"{HOT3D}/hoi_segments/seg_*.npz"))[:n_segs]
    raw, trans, rigid, rot_spread = [], [], [], []
    for sp in segs:
        s = np.load(sp, allow_pickle=True)
        hm, hj, pr = s["hand_mano"], s["hand_joints"], s["hands_present"]
        h = 1  # right hand (only genuine MANO_RIGHT on this box)
        lm = fk(hm[:, h])
        Rs, dr, dt, d0 = [], [], [], []
        for t in range(len(hm)):
            if not pr[t, h]:
                continue
            gt = hj[t, h]
            d0.append(np.linalg.norm(lm[t] - gt, axis=1))
            off = (gt - lm[t]).mean(0)
            dt.append(np.linalg.norm(lm[t] + off - gt, axis=1))
            dr.append(kabsch_resid(lm[t], gt))
        if not dr:
            continue
        raw.append(np.median(np.concatenate(d0)) * 1000)
        trans.append(np.median(np.concatenate(dt)) * 1000)
        rigid.append(np.median(np.concatenate(dr)) * 1000)
    return dict(
        n_segs=len(rigid), hand="right", betas="zero",
        raw_median_mm=round(float(np.median(raw)), 1),
        trans_aligned_median_mm=round(float(np.median(trans)), 1),
        rigid_aligned_median_mm=round(float(np.median(rigid)), 1),
        rigid_aligned_p25_mm=round(float(np.percentile(rigid, 25)), 1),
        rigid_aligned_p75_mm=round(float(np.percentile(rigid, 75)), 1),
    )


# --------------------------------------------------------------------------------- main
def main():
    obj = {c: measure_object(c, a) for c, a in CLIPS.items()}
    hand = {c: measure_hand(c) for c in CLIPS}
    hand = {c: v for c, v in hand.items() if v is not None}

    def pool_obj(clips):
        al = pool_series_stats([obj[c]["_series"]["along"] for c in clips])
        pe = pool_series_stats([np.concatenate([obj[c]["_series"]["p1"], obj[c]["_series"]["p2"]])
                                for c in clips])
        rs = pool_series_stats([obj[c]["_series"]["rot_smooth"] for c in clips])
        durs = sum([obj[c]["flip"]["durations"] for c in clips], [])
        nf = sum(obj[c]["n_frames"] for c in clips)
        nev = sum(len(obj[c]["flip"]["durations"]) for c in clips)
        mags = [obj[c]["flip"]["mag_deg_mean"] for c in clips if obj[c]["flip"]["durations"]]
        state = float(np.average([obj[c]["flip"]["state_frac"] for c in clips],
                                 weights=[obj[c]["n_frames"] for c in clips]))
        return dict(
            trans_along={k: round(v, 4) for k, v in al.items()},
            trans_perp={k: round(v, 4) for k, v in pe.items()},
            rot_smooth={k: round(v, 4) for k, v in rs.items()},
            flip=dict(event_rate=round(nev / nf, 5), state_frac=round(state, 4),
                      mag_deg_mean=round(float(np.mean(mags)), 2) if mags else 0.0,
                      dur_mean=round(float(np.mean(durs)), 2) if durs else 0.0,
                      dur_std=round(float(np.std(durs)), 2) if durs else 0.0,
                      dur_max=int(max(durs)) if durs else 0,
                      durations=[int(d) for d in durs]))

    def pool_hand(clips):
        al = pool_series_stats([hand[c]["_series"]["along"] for c in clips])
        pe = pool_series_stats([np.concatenate([hand[c]["_series"]["p1"], hand[c]["_series"]["p2"]])
                                for c in clips])
        art = [hand[c]["articulation_resid_mm"]["median"] for c in clips]
        return dict(trans_along={k: round(v, 4) for k, v in al.items()},
                    trans_perp={k: round(v, 4) for k, v in pe.items()},
                    articulation_resid_median_mm=round(float(np.median(art)), 2))

    mano = measure_mano(20)

    def strip(d):  # drop private _series before serialising
        return {c: {k: v for k, v in e.items() if not k.startswith("_")} for c, e in d.items()}

    out = dict(
        meta=dict(
            what="per-frame error of coarse HOI stack vs HOT3D mocap GT",
            units="translation mm, rotation deg (camera/pinhole frame)",
            object_arm="fpauto (learned core) on 5 clips; potato_masher uses icpjgr (registration)",
            noise_model="AR(1)+white identified via lag-1/lag-2 autocorr (rho_ar=r2/r1, var_ar/var=r1^2/r2)",
            flip_thresh_deg=FLIP_DEG,
            pooled_default="object/hand 'pooled' = the 5 fpauto clips (the shipped object arm); "
                           "'pooled_all' adds potato_masher (icpjgr, symmetric outlier)",
        ),
        object=dict(per_clip=strip(obj), pooled=pool_obj(FPAUTO), pooled_all=pool_obj(list(CLIPS))),
        hand=dict(per_clip=strip(hand), pooled=pool_hand([c for c in FPAUTO if c in hand])),
        mano_reliability=mano,
    )
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)

    # ---- console summary
    print(f"\nwrote {OUT}\n")
    print("OBJECT (pooled fpauto):")
    p = out["object"]["pooled"]
    print(f"  along-ray: bias={p['trans_along']['bias']:+.1f}mm sigma={p['trans_along']['sigma']:.1f} "
          f"rho_ar={p['trans_along']['rho_ar']:.2f}")
    print(f"  perp     : sigma={p['trans_perp']['sigma']:.1f}mm rho_ar={p['trans_perp']['rho_ar']:.2f}")
    print(f"  rot smooth: bias={p['rot_smooth']['bias']:.1f}deg sigma={p['rot_smooth']['sigma']:.1f} "
          f"rho_ar={p['rot_smooth']['rho_ar']:.2f}")
    print(f"  flips: rate={p['flip']['event_rate']:.4f}/frame state_frac={p['flip']['state_frac']:.2f} "
          f"mag={p['flip']['mag_deg_mean']:.0f}deg dur_mean={p['flip']['dur_mean']:.1f}")
    print("HAND (pooled fpauto):")
    h = out["hand"]["pooled"]
    print(f"  along-ray: bias={h['trans_along']['bias']:+.1f}mm sigma={h['trans_along']['sigma']:.1f} "
          f"rho_ar={h['trans_along']['rho_ar']:.2f}")
    print(f"  perp     : sigma={h['trans_perp']['sigma']:.1f}mm ; articulation resid "
          f"med={h['articulation_resid_median_mm']:.1f}mm")
    print(f"MANO reliability: raw={mano['raw_median_mm']}mm trans-aligned={mano['trans_aligned_median_mm']}mm "
          f"rigid-aligned={mano['rigid_aligned_median_mm']}mm -> "
          f"{'USABLE' if mano['rigid_aligned_median_mm'] < 10 else 'NOT usable'} (articulation)")
    print("\nper-clip object align sanity (median trans mm):")
    for c in CLIPS:
        print(f"  {c:24s} {obj[c]['align_trans_med_mm']:6.1f}  rot_med={obj[c]['rot_all_med_deg']:5.1f} "
              f"flip_state={obj[c]['flip']['state_frac']:.2f}")


if __name__ == "__main__":
    main()
