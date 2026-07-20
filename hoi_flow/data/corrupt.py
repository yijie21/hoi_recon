"""Turn a CLEAN GT trajectory into a realistic "coarse-like" one, using the error
statistics measured in `calibration.json` (see measure_calibration.py). This lets the
flow-matching HOI refiner train on the SAME error distribution its inputs will have at
inference, instead of hand-guessed noise.

The corruption model, per calibrated knob:
  * translation  -> AR(1)+white, applied SEPARATELY along the camera view-ray (depth,
    weakly constrained, large sigma) and in the two perpendicular image-plane axes
    (well constrained, small sigma). Reproduces bias, sigma, rho and jitter.
  * object rotation -> a smooth AR-correlated wobble (calibrated magnitude/rho about a
    slowly-drifting axis) PLUS occasional FLIP events (Bernoulli onset at the calibrated
    rate, duration resampled from the calibrated distribution, ~180 deg about a random
    axis) — the symmetric-object failure mode.
  * hand -> the same along/perp translation model on the MANO wrist translation (or a
    rigid shift of the joints), a small AR wrist-rotation wobble, and small per-theta
    noise. Hand depth (along-ray) dominates because the shipped optimizer pins the 2D
    reprojection.

AR(1)+white is generated as  s_t = rho_ar*s_{t-1} + N(0, sigma_ar*sqrt(1-rho_ar^2))
plus independent N(0, sigma_white); this matches how the stats were identified
(rho_ar = r2/r1, var_ar/var = r1^2/r2). Every knob is overridable via `overrides`.

numpy in, numpy out, fully seeded (pass an np.random.Generator).
Self-test:  python -m hoi_flow.data.corrupt
"""
import copy
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation

CALIB = os.path.join(os.path.dirname(__file__), "calibration.json")


# ------------------------------------------------------------------------ loading
def load_stats(path=CALIB):
    with open(path) as f:
        return json.load(f)


def object_profile(stats, name="pooled", overrides=None):
    """Pick an object-error profile: 'pooled' (5 fpauto clips, the shipped arm, DEFAULT),
    'pooled_all' (adds the potato_masher icpjgr outlier), or a clip name."""
    s = stats["object"]["per_clip"].get(name) or stats["object"][name]
    return _apply_overrides(copy.deepcopy(s), overrides)


def hand_profile(stats, name="pooled", overrides=None):
    s = stats["hand"]["per_clip"].get(name) or stats["hand"][name]
    return _apply_overrides(copy.deepcopy(s), overrides)


def _apply_overrides(d, overrides):
    if overrides:
        for k, v in overrides.items():
            if isinstance(v, dict) and isinstance(d.get(k), dict):
                d[k].update(v)
            else:
                d[k] = v
    return d


# ------------------------------------------------------------------ noise primitives
def _ar1_white(T, sigma_ar, rho_ar, sigma_white, rng):
    """A scalar AR(1) process + independent white noise. Stationary std ~=
    sqrt(sigma_ar^2 + sigma_white^2); lag-1 autocorr ~= rho_ar*sigma_ar^2/var."""
    s = np.zeros(T)
    if sigma_ar > 1e-9:
        innov = sigma_ar * np.sqrt(max(1.0 - rho_ar ** 2, 1e-6))
        s[0] = rng.normal(0, sigma_ar)
        for t in range(1, T):
            s[t] = rho_ar * s[t - 1] + rng.normal(0, innov)
    if sigma_white > 1e-9:
        s = s + rng.normal(0, sigma_white, T)
    return s


def _perp_basis(r0):
    r0 = r0 / (np.linalg.norm(r0) + 1e-12)
    up = np.array([0.0, 1.0, 0.0]) if abs(r0[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u1 = np.cross(r0, up); u1 /= np.linalg.norm(u1) + 1e-12
    u2 = np.cross(r0, u1)
    return r0, u1, u2


def _view_dirs(t, view_dirs):
    """Unit camera->point direction per frame; from translation if not given."""
    if view_dirs is not None:
        v = np.asarray(view_dirs, float)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    return t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-12)


def _trans_error_m(T, along, perp, rhat, rng):
    """Camera-frame translation error [T,3] in METRES from along/perp AR stats (mm)."""
    r0, u1, u2 = _perp_basis(rhat.mean(0))
    a = along["bias"] + _ar1_white(T, along["sigma_ar"], along["rho_ar"], along["sigma_white"], rng)
    p1 = perp["bias"] + _ar1_white(T, perp["sigma_ar"], perp["rho_ar"], perp["sigma_white"], rng)
    p2 = _ar1_white(T, perp["sigma_ar"], perp["rho_ar"], perp["sigma_white"], rng)
    e = a[:, None] * rhat + p1[:, None] * u1 + p2[:, None] * u2   # mm
    return e / 1000.0


def _smooth_axis(T, rng, rho=0.9):
    """A slowly-drifting unit-axis sequence (AR on each component, then normalised)."""
    ax = np.zeros((T, 3))
    ax[0] = rng.normal(size=3)
    for t in range(1, T):
        ax[t] = rho * ax[t - 1] + np.sqrt(1 - rho ** 2) * rng.normal(size=3)
    return ax / (np.linalg.norm(ax, axis=1, keepdims=True) + 1e-12)


def _smooth_rot(T, rot_stats, rng):
    """Smooth AR-correlated rotation-error rotvecs [T,3] (radians), calibrated magnitude."""
    mag = rot_stats["bias"] + _ar1_white(T, rot_stats["sigma_ar"], rot_stats["rho_ar"],
                                         rot_stats["sigma_white"], rng)
    mag = np.clip(mag, 0.0, 180.0)
    return _smooth_axis(T, rng) * np.deg2rad(mag)[:, None]


def _flip_rot(T, flip, rng):
    """Per-frame FLIP rotvecs [T,3] (radians): Bernoulli onsets at flip['event_rate'],
    duration resampled from flip['durations'], magnitude ~N(mag_mean, mag_std) about a
    random axis, held for the duration."""
    rv = np.zeros((T, 3))
    durs = flip.get("durations") or [int(round(flip.get("dur_mean", 3)))]
    rate = flip.get("event_rate", 0.0)
    mag_m, mag_s = flip.get("mag_deg_mean", 0.0), flip.get("mag_deg_std", 20.0)
    if rate <= 0 or mag_m <= 0:
        return rv
    t = 0
    while t < T:
        if rng.random() < rate:
            d = int(rng.choice(durs))
            axis = rng.normal(size=3); axis /= np.linalg.norm(axis) + 1e-12
            mag = np.deg2rad(np.clip(rng.normal(mag_m, mag_s), 30.0, 180.0))
            rv[t:min(t + d, T)] = axis * mag
            t += max(d, 1)
        else:
            t += 1
    return rv


# --------------------------------------------------------------------------- objects
def corrupt_object(poses_gt, stats, rng, view_dirs=None, overrides=None):
    """Corrupt clean object poses [T,4,4] (object->camera) into coarse-like poses.

    poses_gt : [T,4,4] float
    stats    : an object profile dict (from object_profile(...)) OR the full calibration
               dict (then 'pooled' is used). Keys: trans_along, trans_perp, rot_smooth, flip.
    rng      : np.random.Generator
    view_dirs: optional [T,3] camera->object unit dirs; else taken from translation.
    returns  : [T,4,4] corrupted poses.
    """
    if "trans_along" not in stats:                    # a full calibration dict was passed
        stats = stats["object"]["pooled"]
    stats = _apply_overrides(copy.deepcopy(stats), overrides)
    P = np.asarray(poses_gt, float).copy()
    T = len(P)
    t_gt = P[:, :3, 3]
    rhat = _view_dirs(t_gt, view_dirs)

    e = _trans_error_m(T, stats["trans_along"], stats["trans_perp"], rhat, rng)
    smooth = _smooth_rot(T, stats["rot_smooth"], rng)
    flip = _flip_rot(T, stats["flip"], rng)
    R_noise = (Rotation.from_rotvec(flip) * Rotation.from_rotvec(smooth)).as_matrix()

    out = P.copy()
    out[:, :3, :3] = R_noise @ P[:, :3, :3]           # rotate orientation in place
    out[:, :3, 3] = t_gt + e                          # translation error (about object centre)
    return out


# ----------------------------------------------------------------------------- hands
def corrupt_hand(hand, stats, rng, view_dirs=None, overrides=None,
                 theta_sigma=0.05, wrist_rot_deg=4.0):
    """Corrupt a clean hand into a coarse-like one.

    hand : either
        - dict {'wrist_xform':[T,6] (axis-angle[3]+transl[3], metres), 'theta':[T,K]} -> MANO params
        - ndarray [T,J,3] joints/verts (metres)                                       -> joints fallback
    stats: a hand profile dict (from hand_profile(...)) OR the full calibration dict.
    theta_sigma  : per-theta white noise std (PCA units) — not calibrated; documented knob.
    wrist_rot_deg: sigma of the small AR wrist-rotation wobble (deg) — documented knob.
    returns: same type/shape as `hand`, corrupted.
    """
    if "trans_along" not in stats:
        stats = stats["hand"]["pooled"]
    stats = _apply_overrides(copy.deepcopy(stats), overrides)
    al, pe = stats["trans_along"], stats["trans_perp"]

    if isinstance(hand, dict):                        # ---- MANO params
        wrist = np.asarray(hand["wrist_xform"], float).copy()
        theta = np.asarray(hand["theta"], float).copy()
        T = len(wrist)
        transl = wrist[:, 3:6]
        rhat = _view_dirs(transl, view_dirs)
        wrist[:, 3:6] = transl + _trans_error_m(T, al, pe, rhat, rng)             # along-ray AR offset
        wob = _smooth_rot(T, dict(bias=0.0, sigma_ar=wrist_rot_deg,               # deg; small AR wobble
                                  rho_ar=0.9, sigma_white=0.0), rng)
        R_new = Rotation.from_rotvec(wob) * Rotation.from_rotvec(wrist[:, 0:3])
        wrist[:, 0:3] = R_new.as_rotvec()
        theta = theta + rng.normal(0, theta_sigma, theta.shape)                  # per-theta noise
        return {"wrist_xform": wrist, "theta": theta}

    J = np.asarray(hand, float).copy()                # ---- joints/verts fallback: rigid shift
    T = len(J)
    centroid = J.mean(1)
    rhat = _view_dirs(centroid, view_dirs)
    e = _trans_error_m(T, al, pe, rhat, rng)          # [T,3] metres
    return J + e[:, None, :]


# ------------------------------------------------------------------------- self-test
def _autocorr(x, k):
    x = x - x.mean(); d = np.sum(x * x)
    return float(np.sum(x[k:] * x[:-k]) / d) if d > 1e-12 else 0.0


def _measure(poses_c, poses_gt):
    """Re-derive along/perp translation stats and flip rate the same way calibration did."""
    e = (poses_c[:, :3, 3] - poses_gt[:, :3, 3]) * 1000.0
    rhat = poses_gt[:, :3, 3] / np.linalg.norm(poses_gt[:, :3, 3], axis=1, keepdims=True)
    r0, u1, u2 = _perp_basis(rhat.mean(0))
    along = np.einsum("ti,ti->t", e, rhat)
    perp = np.concatenate([e @ u1, e @ u2])
    Rrel = np.einsum("tji,tjk->tik", poses_gt[:, :3, :3], poses_c[:, :3, :3])
    ang = np.degrees(np.arccos(np.clip((np.trace(Rrel, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    return dict(along_bias=along.mean(), along_sigma=along.std(), along_rho1=_autocorr(along, 1),
                perp_sigma=perp.std(), flip_rate=np.mean(ang > 60) )


def _self_test():
    import glob
    stats = load_stats()
    rng = np.random.default_rng(0)
    seg = sorted(glob.glob("/workspace/datasets/hot3d/hoi_segments/seg_*.npz"))[0]
    d = np.load(seg, allow_pickle=True)
    poses_gt = d["obj_pose"].astype(float)            # [T,4,4] object->camera (clean GT)

    prof = object_profile(stats, "pooled")
    # average over several seeds to compare distributions to targets
    m = {k: [] for k in ["along_bias", "along_sigma", "along_rho1", "perp_sigma", "flip_rate"]}
    for s in range(40):
        c = corrupt_object(poses_gt, prof, np.random.default_rng(s))
        r = _measure(c, poses_gt)
        for k in m:
            m[k].append(r[k])
    m = {k: float(np.mean(v)) for k, v in m.items()}
    ta, tp, tf = prof["trans_along"], prof["trans_perp"], prof["flip"]
    print("OBJECT corruption self-test (mean over 40 seeds) vs calibration target:")
    print(f"  along bias   : {m['along_bias']:+7.2f}  target {ta['bias']:+7.2f}")
    print(f"  along sigma  : {m['along_sigma']:7.2f}  target {ta['sigma']:7.2f}")
    print(f"  along rho(1) : {m['along_rho1']:7.2f}  target {ta['rho1']:7.2f}")
    print(f"  perp  sigma  : {m['perp_sigma']:7.2f}  target {tp['sigma']:7.2f}")
    print(f"  flip  rate   : {m['flip_rate']:7.3f}  target(state_frac) {tf['state_frac']:7.3f}")

    # hand joints-space fallback
    hprof = hand_profile(stats, "pooled")
    joints = d["hand_joints"][:, 1].astype(float)     # [T,20,3] right-hand GT landmarks
    mh = {"along_sigma": [], "perp_sigma": [], "along_bias": []}
    for s in range(40):
        jc = corrupt_hand(joints, hprof, np.random.default_rng(s))
        e = (jc.mean(1) - joints.mean(1)) * 1000.0
        rhat = joints.mean(1) / np.linalg.norm(joints.mean(1), axis=1, keepdims=True)
        r0, u1, u2 = _perp_basis(rhat.mean(0))
        mh["along_sigma"].append(np.einsum("ti,ti->t", e, rhat).std())
        mh["along_bias"].append(np.einsum("ti,ti->t", e, rhat).mean())
        mh["perp_sigma"].append(np.concatenate([e @ u1, e @ u2]).std())
    ha = hprof["trans_along"]; hp = hprof["trans_perp"]
    print("\nHAND corruption self-test (joints fallback, 40 seeds) vs target:")
    print(f"  along bias   : {np.mean(mh['along_bias']):+7.2f}  target {ha['bias']:+7.2f}")
    print(f"  along sigma  : {np.mean(mh['along_sigma']):7.2f}  target {ha['sigma']:7.2f}")
    print(f"  perp  sigma  : {np.mean(mh['perp_sigma']):7.2f}  target {hp['sigma']:7.2f}")

    # MANO-params path smoke (shapes only)
    hm = d["hand_mano"][:, 1].astype(float)           # [T,21]
    out = corrupt_hand({"wrist_xform": hm[:, 15:21], "theta": hm[:, :15]}, hprof,
                       np.random.default_rng(1))
    print(f"\nMANO-params path OK: wrist {out['wrist_xform'].shape} theta {out['theta'].shape}")


if __name__ == "__main__":
    _self_test()
