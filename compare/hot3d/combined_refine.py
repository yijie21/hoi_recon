"""Combined method prototype: learned per-frame RGB-D pose (Any6D / FoundationPose)
+ icpjgr's temporal-consistency layer as a POST-PROCESSOR.

The bake-off (T4_RESULTS.md) showed learned per-frame estimators beat icpjgr's
registration on accuracy but lack temporal robustness: Any6D snaps a minority of
frames to a ~180deg symmetry-equivalent pose (rot p90 153-173deg) and FP `track`
drifts. This wraps the SAVED per-frame poses in the two priors icpjgr contributes:

  1. Symmetry-flip resolution (the big one): discover the object's near-symmetry
     group G (rotations that leave the mesh ~invariant), then per frame pick the
     symmetry-equivalent rotation R[t]@S (S in G) that is temporally closest to the
     running trajectory. R[t] and R[t]@S render the SAME object, so this only fixes
     the estimator's basin choice — it never changes what's actually observed.
  2. Robust SO(3) trajectory smoothing (second-difference), to remove residual
     per-frame jitter without washing out real motion.

Operates on the saved pseudo_gt.npz of a learned run; writes a cleaned run so it
can be scored by gt_pose_eval_hot3d.py exactly like any arm.

Usage: combined_refine.py <learned_run_dir> <out_run_dir> [--w_smooth 40]
"""
import argparse
import os

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def discover_symmetries(verts, faces, tol_mm=4.0, n_sample=2000, seed=0):
    """Return the rotations (list of 3x3) that leave the mesh ~invariant.

    Candidate set covers the common HOI cases: the 24 octahedral rotations (boxes
    like the cube) and a fine azimuthal sweep about each principal axis (bottles /
    cans / cups are revolution-symmetric). A candidate is kept if the max
    surface-to-surface deviation of the rotated points vs the originals is below
    tol_mm (metres in, so tol scaled)."""
    rng = np.random.default_rng(seed)
    V = verts - verts.mean(0)
    idx = rng.choice(len(V), min(n_sample, len(V)), replace=False)
    P = V[idx]
    tree = cKDTree(V)
    # PCA axes to define the sweep frame
    _, _, Vt = np.linalg.svd(P - P.mean(0), full_matrices=False)
    A = Vt  # rows = principal axes
    tol = tol_mm / 1000.0

    cands = [np.eye(3)]
    cands += list(Rotation.create_group("O").as_matrix())          # octahedral (boxes)
    for ax in range(3):                                            # revolution sweeps
        axis = A[ax]
        for k in range(1, 24):
            cands.append(Rotation.from_rotvec(axis * (2 * np.pi * k / 24)).as_matrix())

    kept = []
    seen = []
    for R in cands:
        # dedup near-identical candidates
        if any(np.degrees(Rotation.from_matrix(R @ K.T).magnitude()) < 5 for K in seen):
            continue
        d, _ = tree.query(P @ R.T)
        if np.quantile(d, 0.95) < tol:
            kept.append(R)
            seen.append(R)
    return kept


def resolve_flips(R, sym):
    """Greedy temporal symmetry resolution + a backward refinement pass.

    R[t] (3x3) per frame; sym = list of symmetry rotations. Choose S[t] in sym to
    minimise geodesic(R[t]@S[t], ref) where ref is the previous cleaned frame,
    anchored at the frame whose raw pose is most agreed-with by its neighbours."""
    T = len(R)
    if len(sym) <= 1:
        return R.copy()
    Rs = np.stack(sym)                                            # [S,3,3]

    def geod(A, B):
        return np.degrees(Rotation.from_matrix(A @ B.T).magnitude())

    # anchor = frame with smallest summed neighbour jump (most stable region)
    nb = np.array([geod(R[t], R[t - 1]) + geod(R[t], R[(t + 1) % T]) for t in range(T)])
    a = int(np.argmin(nb))
    out = R.copy()
    # forward from anchor, then backward, twice (settles the anchor region)
    for _ in range(2):
        for t in list(range(a + 1, T)) + list(range(a - 1, -1, -1)):
            ref = out[t - 1] if t > a else out[t + 1]
            cands = np.einsum("ab,sbc->sac", R[t], Rs)           # R[t]@S for all S
            j = int(np.argmin([geod(c, ref) for c in cands]))
            out[t] = cands[j]
    return out


def _accel_smooth(X, lam):
    """Data-anchored acceleration smoother, per column, closed form:
    argmin_y ||y - X||^2 + lam * ||D2 y||^2  =>  (I + lam D2^T D2) y = X.
    Strong data term (weight 1) keeps y AT the estimate; lam only penalises the
    second difference (acceleration = jitter). Small lam -> y ~= X (no harm),
    removing only the high-frequency wiggle. Endpoints are naturally free."""
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

    Translations: acceleration-smoothed directly (mm space).
    Rotations: quaternions (hemisphere-consistent, so no sign flips), each of the
    4 components acceleration-smoothed, then renormalised. The flip-resolved basin
    is preserved because neighbours are already aligned. `lam_*` are the ONLY knobs
    — swept to the largest value that leaves chamfer/rot_traj within noise."""
    t_s = _accel_smooth(t, lam_trans) if lam_trans > 0 else t
    if lam_rot <= 0:
        return R.copy(), t_s
    q = np.stack([Rotation.from_matrix(r).as_quat() for r in R])   # xyzw
    for i in range(1, len(q)):                                     # hemisphere align
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    q_s = _accel_smooth(q, lam_rot)
    q_s /= np.linalg.norm(q_s, axis=1, keepdims=True)
    R_s = np.stack([Rotation.from_quat(qq).as_matrix() for qq in q_s])
    return R_s, t_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("learned_run")
    ap.add_argument("out_run")
    # Data-anchored acceleration smoothing (jitter removal). Defaults are the
    # swept "no-harm" values; 0 disables. See the sweep in T4_RESULTS.md.
    # Swept on the bench (T4_RESULTS.md): TRANSLATION acceleration-smoothing is a
    # universal free win — jitter p90 drops 5-10x (66->7mm) at <1mm chamfer cost,
    # rotation untouched. ROTATION smoothing is OFF by default: naively averaging
    # the learned estimator's rotation flips DISTORTS the trajectory (masher
    # chamfer 12->21, vase rot median 8.5->14). It needs a flip-aware SO(3)
    # smoother, not this scalar knob — enable --lam_rot only with that in place.
    ap.add_argument("--lam_rot", type=float, default=0.0)
    ap.add_argument("--lam_trans", type=float, default=3.0)
    a = ap.parse_args()

    z = np.load(f"{a.learned_run}/stage8_eval/pseudo_gt.npz")
    poses = z["obj_poses"].copy()
    verts, faces = z["obj_verts"], z["obj_faces"]
    R = poses[:, :3, :3]

    sym = discover_symmetries(verts, faces)
    print(f"discovered {len(sym)} near-symmetries")
    R_res = resolve_flips(R, sym)
    R_out, t_out = smooth_traj(R_res, poses[:, :3, 3],
                               lam_rot=a.lam_rot, lam_trans=a.lam_trans)

    poses[:, :3, :3] = R_out
    poses[:, :3, 3] = t_out
    # (translations are already smooth from the learned estimator; keep them)
    out_dir = f"{a.out_run}/stage8_eval"
    os.makedirs(out_dir, exist_ok=True)
    np.savez(f"{out_dir}/pseudo_gt.npz", obj_verts=verts, obj_faces=faces,
             obj_poses=poses)

    # report the flip fix
    def jumps(Rm):
        d = [np.degrees(Rotation.from_matrix(Rm[t + 1] @ Rm[t].T).magnitude())
             for t in range(len(Rm) - 1)]
        return np.median(d), np.percentile(d, 90), int((np.array(d) > 90).sum())
    print(f"frame-jumps raw    : med/p90/#>90 = {tuple(round(x,1) for x in jumps(R))}")
    print(f"frame-jumps cleaned: med/p90/#>90 = {tuple(round(x,1) for x in jumps(R_out))}")
    print(f"wrote {out_dir}/pseudo_gt.npz")


if __name__ == "__main__":
    main()
