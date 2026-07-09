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


def smooth_so3(R, w=40.0, iters=200, lr=0.2):
    """Light second-difference smoothing on the rotation trajectory (tangent-space
    gradient descent). Keeps the flip-resolved basin; removes jitter."""
    T = len(R)
    logs = np.stack([Rotation.from_matrix(r).as_rotvec() for r in R])
    y = logs.copy()
    for _ in range(iters):
        d2 = np.zeros_like(y)
        d2[1:-1] = y[2:] - 2 * y[1:-1] + y[:-2]
        grad = (y - logs) + w * (np.roll(d2, 1, 0) - 2 * d2 + np.roll(d2, -1, 0)) * 0
        # simpler: pull toward data + Laplacian smoothing
        lap = np.zeros_like(y)
        lap[1:-1] = y[2:] - 2 * y[1:-1] + y[:-2]
        y = y + lr * (-(y - logs) / max(w, 1e-3) + lap)
    return np.stack([Rotation.from_rotvec(v).as_matrix() for v in y])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("learned_run")
    ap.add_argument("out_run")
    ap.add_argument("--w_smooth", type=float, default=40.0)
    # Flip-resolution is the proven value (bottle rot p90 171->17, chamfer kept
    # exactly since a symmetry op can't move the surface). The second-difference
    # smoother distorted placement on non-flip clips (masher chamfer 12->25), so
    # it is OFF by default; enable with --smooth only after reformulating it.
    ap.add_argument("--smooth", action="store_true")
    a = ap.parse_args()
    a.no_smooth = not a.smooth

    z = np.load(f"{a.learned_run}/stage8_eval/pseudo_gt.npz")
    poses = z["obj_poses"].copy()
    verts, faces = z["obj_verts"], z["obj_faces"]
    R = poses[:, :3, :3]

    sym = discover_symmetries(verts, faces)
    print(f"discovered {len(sym)} near-symmetries")
    R_res = resolve_flips(R, sym)
    R_out = R_res if a.no_smooth else smooth_so3(R_res, w=a.w_smooth)

    poses[:, :3, :3] = R_out
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
