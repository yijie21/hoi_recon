"""Temporal-consistency layer for a per-frame object pose trajectory.

It cleans a sequence of per-frame object poses produced by a *per-frame-independent* estimator
(e.g. Any6D) using two priors the frame-independent estimator lacks:

  1. Symmetry-flip resolution: discover the object's near-symmetry group G
     (rotations that leave the mesh ~invariant), then per frame pick the
     symmetry-equivalent rotation R[t]@S (S in G) temporally closest to the running
     trajectory. R[t] and R[t]@S render the SAME object, so this only fixes the
     estimator's basin choice — it never changes what is actually observed.
  2. Jitter smoothing: a data-anchored acceleration smoother (closed form) that
     removes high-frequency wiggle without moving the trajectory off the estimate.

Item-1 (integrate the learned core inside the pipeline) imports `clean_trajectory`
so BOTH the grasp stages and eval see the cleaned poses. Item-2 (flip-aware SO(3)
rotation smoother + depth-anchored basin selection) extends this module.

Pure numpy + scipy (available in the pipeline env `rc5090`); no torch.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def discover_symmetries(verts, faces, tol_mm=4.0, n_sample=2000, seed=0):
    """Rotations (list of 3x3) that leave the mesh ~invariant.

    Candidate set covers the common HOI cases: the 24 octahedral rotations (boxes
    like the Rubik's cube) and a fine azimuthal sweep about each principal axis
    (bottles / cans / cups are revolution-symmetric). A candidate is kept if the
    95th-percentile surface-to-surface deviation of the rotated sample vs the
    original mesh is below tol_mm (verts are in metres; tol scaled accordingly)."""
    rng = np.random.default_rng(seed)
    V = np.asarray(verts, dtype=np.float64)
    V = V - V.mean(0)
    idx = rng.choice(len(V), min(n_sample, len(V)), replace=False)
    P = V[idx]
    tree = cKDTree(V)
    _, _, Vt = np.linalg.svd(P - P.mean(0), full_matrices=False)
    A = Vt  # rows = principal axes
    tol = tol_mm / 1000.0

    cands = [np.eye(3)]
    cands += list(Rotation.create_group("O").as_matrix())          # octahedral (boxes)
    for ax in range(3):                                            # revolution sweeps
        axis = A[ax]
        for k in range(1, 24):
            cands.append(Rotation.from_rotvec(axis * (2 * np.pi * k / 24)).as_matrix())

    kept, seen = [], []
    for R in cands:
        if any(np.degrees(Rotation.from_matrix(R @ K.T).magnitude()) < 5 for K in seen):
            continue  # dedup near-identical candidates
        d, _ = tree.query(P @ R.T)
        if np.quantile(d, 0.95) < tol:
            kept.append(R)
            seen.append(R)
    return kept


def _geod_deg(A, B):
    return np.degrees(Rotation.from_matrix(A @ B.T).magnitude())


def resolve_flips(R, sym):
    """Greedy temporal symmetry resolution with a backward refinement pass.

    R[t] (3x3) per frame; sym = list of symmetry rotations. Choose S[t] in sym to
    minimise geodesic(R[t]@S[t], ref) where ref is the previous cleaned frame,
    anchored at the frame whose raw pose is most agreed-with by its neighbours."""
    R = np.asarray(R, dtype=np.float64)
    T = len(R)
    if len(sym) <= 1 or T < 2:
        return R.copy()
    Rs = np.stack(sym)                                            # [S,3,3]
    nb = np.array([_geod_deg(R[t], R[t - 1]) + _geod_deg(R[t], R[(t + 1) % T])
                   for t in range(T)])
    a = int(np.argmin(nb))                                        # most stable region
    out = R.copy()
    for _ in range(2):                                            # forward then backward, twice
        for t in list(range(a + 1, T)) + list(range(a - 1, -1, -1)):
            ref = out[t - 1] if t > a else out[t + 1]
            cands = np.einsum("ab,sbc->sac", R[t], Rs)           # R[t]@S for all S
            j = int(np.argmin([_geod_deg(c, ref) for c in cands]))
            out[t] = cands[j]
    return out


def _accel_smooth(X, lam):
    """Data-anchored acceleration smoother, per column, closed form:
    argmin_y ||y - X||^2 + lam * ||D2 y||^2  =>  (I + lam D2^T D2) y = X.
    Strong data term keeps y AT the estimate; lam penalises only the second
    difference (acceleration = jitter). Endpoints are naturally free."""
    X = np.asarray(X, dtype=np.float64)
    T = len(X)
    if lam <= 0 or T < 3:
        return X.copy()
    D2 = np.zeros((T - 2, T))
    for i in range(T - 2):
        D2[i, i], D2[i, i + 1], D2[i, i + 2] = 1.0, -2.0, 1.0
    A = np.eye(T) + lam * (D2.T @ D2)
    return np.linalg.solve(A, X)


def smooth_traj(R, t, lam_rot=0.0, lam_trans=0.0):
    """Jitter-only smoothing that does not move the trajectory off the estimate.

    Translations: acceleration-smoothed directly. Rotations: quaternions
    (hemisphere-consistent, so no sign flips), each component acceleration-smoothed,
    then renormalised. The flip-resolved basin is preserved because neighbours are
    already aligned. NOTE (item-2): scalar rotation smoothing averages across any
    residual flips, so lam_rot stays 0 until the flip-aware smoother lands."""
    t_s = _accel_smooth(t, lam_trans) if lam_trans > 0 else np.asarray(t, np.float64).copy()
    if lam_rot <= 0:
        return np.asarray(R, np.float64).copy(), t_s
    q = np.stack([Rotation.from_matrix(r).as_quat() for r in R])   # xyzw
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    q_s = _accel_smooth(q, lam_rot)
    q_s /= np.linalg.norm(q_s, axis=1, keepdims=True)
    R_s = np.stack([Rotation.from_quat(qq).as_matrix() for qq in q_s])
    return R_s, t_s


def frame_jumps(R):
    """Diagnostic: (median, p90, #>90deg) of consecutive-frame rotation jumps."""
    R = np.asarray(R, dtype=np.float64)
    if len(R) < 2:
        return (0.0, 0.0, 0)
    d = np.array([_geod_deg(R[t + 1], R[t]) for t in range(len(R) - 1)])
    return float(np.median(d)), float(np.percentile(d, 90)), int((d > 90).sum())


def clean_trajectory(poses, verts, faces, lam_rot=0.0, lam_trans=3.0,
                     sym=None, tol_mm=4.0):
    """One-call temporal cleanup for the pipeline.

    poses: [T,4,4] object->camera. verts/faces: the (already metric-scaled) mesh.
    Returns (poses_out[T,4,4], info) where info records the discovered symmetry
    count and the raw/cleaned frame-jump stats. Pure post-hoc on the poses — the
    mesh is untouched, so downstream stages and eval stay consistent."""
    poses = np.asarray(poses, dtype=np.float64).copy()
    R = poses[:, :3, :3]
    t = poses[:, :3, 3]
    if sym is None:
        sym = discover_symmetries(verts, faces, tol_mm=tol_mm)
    R_res = resolve_flips(R, sym)
    R_out, t_out = smooth_traj(R_res, t, lam_rot=lam_rot, lam_trans=lam_trans)
    poses[:, :3, :3] = R_out
    poses[:, :3, 3] = t_out
    info = {"n_sym": len(sym),
            "jumps_raw": frame_jumps(R),
            "jumps_cleaned": frame_jumps(R_out)}
    return poses, info
