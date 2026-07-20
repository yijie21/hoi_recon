"""Solve the constant MANO-root <-> UmeTrack-wrist SE3 offset C (per hand side).

WHY
---
The 57-D flow state stores each hand's wrist as the UmeTrack `T_cam_from_wrist`
(license-free mocap frame). Our coarse hand estimator (joint_opt.py, via
run_hand_reproj.py) instead produces MANO params — a global-orient rotation + a root
translation. MANO's root joint and UmeTrack's wrist are the SAME physical wrist but use
DIFFERENT local origin/axis conventions, related by a CONSTANT rigid offset C (a property
of the two hand models, independent of camera frame and of finger articulation). This
script measures C from GT so sample.py can map coarse MANO -> the state's wrist convention.

METHOD (reusing measure_calibration.py's MANO-vs-UmeTrack FK machinery)
----------------------------------------------------------------------
Over N random segments x all present+finite frames, per side:
  * FK the GT `hand_mano` (HOT3D-native: thetas = PCA-15, flat_hand_mean=False) with smplx
    to get the root joint position, and take global_orient (axis-angle) as the root rotation:
        T_world_from_manoroot = [[R(global_orient), FK.joints[0]], [0,1]]
  * HOT3D's `hand_mano` wrist_xform is stored in the WORLD frame, so map it into the
    pinhole CAMERA frame with the per-frame world<-pinhole transform stored in the segment:
        M := T_cam_from_manoroot = inv(T_world_pinhole) @ T_world_from_manoroot
    (verified empirically: the world interp gives a ~5 mm/3.5 deg constant-offset residual;
     treating wrist_xform as camera-direct gives ~700 mm — it is world.)
  * W := hand_wrist  (UmeTrack T_cam_from_wrist, camera frame, the state target)
  * Solve the constant C in  W ~= M @ C   (C on the RIGHT = a local offset in the manoroot
    frame => frame-independent; the LEFT form W ~= C @ M is a camera-frame transform and is
    NOT constant across segments — reported for contrast). Closed form:
        C_t = M_t^{-1} @ W_t ;  C = ( SO3-mean of R(C_t), mean of t(C_t) ).

EQUATION saved to wrist_offset.json (this is exactly what sample.py applies):
    T_cam_from_wrist[side]  ~=  T_cam_from_manoroot  @  C[side]
  where, for the COARSE hand, T_cam_from_manoroot is built directly from the camera-frame
  MANO params joint_opt emits (mano_global_aa + FK root; NO T_world_pinhole — those params
  are already camera-frame). Same C.

VALIDATION: C is fit on half the frames and the median trans (mm) / rot (deg) error of
`M @ C` vs the GT wrist is reported on the held-out half. Bar: ~10-15 mm / ~5 deg.

Run (env rc5090, CPU):  python -m hoi_flow.data.solve_wrist_offset [--n_segs 30] [--seed 0]
"""
import argparse
import glob
import json
import os
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

warnings.filterwarnings("ignore")

SEG_DIR = "/workspace/datasets/hot3d/hoi_segments"
MANO_DIR = "/workspace/code/hoi_recon/render_and_compare/checkpoints/mano"
OUT = os.path.join(os.path.dirname(__file__), "wrist_offset.json")
SIDE_NAME = {0: "left", 1: "right"}


def _mano_layer(scratch, is_rhand):
    """smplx.MANO with the HOT3D-native parameterization (PCA-15, flat_hand_mean=False),
    matching measure_calibration.py. Symlinks the chumpy-free pkl to the name smplx expects."""
    import smplx
    os.makedirs(scratch, exist_ok=True)
    src = f"{MANO_DIR}/MANO_{'RIGHT' if is_rhand else 'LEFT'}_np.pkl"
    dst = os.path.join(scratch, f"MANO_{'RIGHT' if is_rhand else 'LEFT'}.pkl")
    if not os.path.exists(dst):
        os.symlink(src, dst)
    return smplx.MANO(dst, is_rhand=is_rhand, use_pca=True, num_pca_comps=15,
                      flat_hand_mean=False, batch_size=1)


def _so3_mean(Rs):
    """Chordal L2 mean of rotations: SVD-orthogonalize the arithmetic mean matrix."""
    U, _, Vt = np.linalg.svd(Rs.mean(0))
    return U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def _geodesic_deg(Ra, Rb):
    R = np.einsum("tji,tjk->tik", Ra, Rb)  # Ra^T Rb
    tr = np.trace(R, axis1=1, axis2=2)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def _manoroot_cam(layer, hm, T_world_pinhole):
    """[K,21] HOT3D hand_mano (thetas15 + global3 + transl3, WORLD frame) + [K,4,4]
    T_world_pinhole  ->  M = T_cam_from_manoroot [K,4,4] (camera frame)."""
    import torch
    K = len(hm)
    out = layer(betas=torch.zeros(K, 10),
                global_orient=torch.tensor(hm[:, 15:18], dtype=torch.float32),
                hand_pose=torch.tensor(hm[:, :15], dtype=torch.float32),
                transl=torch.tensor(hm[:, 18:21], dtype=torch.float32), return_verts=True)
    root = out.joints[:, 0].detach().numpy()                    # world-frame root joint
    R_root = Rotation.from_rotvec(hm[:, 15:18]).as_matrix()     # world-frame root rotation
    Mworld = np.tile(np.eye(4), (K, 1, 1))
    Mworld[:, :3, :3] = R_root
    Mworld[:, :3, 3] = root
    T_pw = np.linalg.inv(T_world_pinhole)                       # pinhole <- world (per frame)
    return T_pw @ Mworld


def _fit_C(M, W):
    """C = SE3-mean of M_t^{-1} @ W_t  (so W ~= M @ C)."""
    Craw = np.linalg.inv(M) @ W
    C = np.eye(4)
    C[:3, :3] = _so3_mean(Craw[:, :3, :3])
    C[:3, 3] = Craw[:, :3, 3].mean(0)
    return C


def _resid(M, W, C, right_form=True):
    Wp = (M @ C) if right_form else (C @ M)
    terr = np.linalg.norm(Wp[:, :3, 3] - W[:, :3, 3], axis=1) * 1000.0
    rerr = _geodesic_deg(Wp[:, :3, :3], W[:, :3, :3])
    return terr, rerr


def solve(n_segs=30, seed=0, seg_dir=SEG_DIR, out=OUT):
    scratch = ("/tmp/claude-1002/-workspace-code-hoi-recon/"
               "cb99b7a6-8766-4fc9-a273-b7123285115d/scratchpad/mano_models")
    layers = {1: _mano_layer(scratch, True), 0: _mano_layer(scratch, False)}

    segs = sorted(glob.glob(os.path.join(seg_dir, "seg_*.npz")))
    rng = np.random.default_rng(seed)
    segs = [segs[i] for i in rng.choice(len(segs), min(n_segs, len(segs)), replace=False)]

    # accumulate per-side M (camera-frame manoroot) and W (UmeTrack wrist)
    acc = {0: {"M": [], "W": []}, 1: {"M": [], "W": []}}
    for sp in segs:
        z = np.load(sp, allow_pickle=True)
        hm, hw, pr, Twp = z["hand_mano"], z["hand_wrist"], z["hands_present"], z["T_world_pinhole"]
        for h in (0, 1):
            ok = (pr[:, h] & np.isfinite(hm[:, h]).all(1)
                  & np.isfinite(hw[:, h]).reshape(len(hm), -1).all(1)
                  & np.isfinite(Twp).reshape(len(hm), -1).all(1))
            if not ok.any():
                continue
            idx = np.where(ok)[0]
            M = _manoroot_cam(layers[h], hm[idx, h], Twp[idx])
            acc[h]["M"].append(M)
            acc[h]["W"].append(hw[idx, h])

    result = {
        "_equation": "T_cam_from_wrist[side]  ~=  T_cam_from_manoroot  @  C[side]",
        "_convention": (
            "T_cam_from_manoroot = [[R(global_orient), FK.root_joint],[0,1]] in the pinhole "
            "camera frame. For GT hand_mano (world frame) that is inv(T_world_pinhole) @ world "
            "manoroot; for the COARSE joint_opt MANO (mano_global_aa + mano_transl, already "
            "camera frame) it is built directly with NO T_world_pinhole. MANO FK: smplx.MANO, "
            "PCA-15 thetas, flat_hand_mean=False, betas=0, right=MANO_RIGHT_np / left=MANO_LEFT_np "
            "(fabricated mirror). C is applied on the RIGHT (a constant offset in the manoroot "
            "local frame). Units: C translation in metres."),
        "n_segments": len(segs), "seed": seed,
        "residual_mm": {}, "residual_deg": {}, "n_frames": {},
    }
    print(f"solving wrist offset over {len(segs)} segments (seed {seed})\n")
    print(f"{'side':6s} {'N':>6s} {'fit_mm':>8s} {'held_mm':>9s} {'held_deg':>9s} "
          f"{'C@M_held_mm':>12s}")
    for h in (0, 1):
        if not acc[h]["M"]:
            print(f"{SIDE_NAME[h]:6s}  (no frames)")
            result[SIDE_NAME[h]] = np.eye(4).tolist()
            continue
        M = np.concatenate(acc[h]["M"]); W = np.concatenate(acc[h]["W"])
        n = len(M)
        # deterministic 50/50 fit/held split (by frame parity of the pooled order)
        fit = np.arange(0, n, 2); held = np.arange(1, n, 2)
        C = _fit_C(M[fit], W[fit])
        tf, _ = _resid(M[fit], W[fit], C)
        th, rh = _resid(M[held], W[held], C)
        # contrast: the LEFT-multiply form (not frame-constant) on held-out
        Cl = np.eye(4)
        Craw = W[fit] @ np.linalg.inv(M[fit])
        Cl[:3, :3] = _so3_mean(Craw[:, :3, :3]); Cl[:3, 3] = Craw[:, :3, 3].mean(0)
        thl, _ = _resid(M[held], W[held], Cl, right_form=False)
        print(f"{SIDE_NAME[h]:6s} {n:6d} {np.median(tf):8.1f} {np.median(th):9.1f} "
              f"{np.median(rh):9.2f} {np.median(thl):12.1f}")
        # refit on ALL frames for the shipped C (more data); held-out numbers above are the honest report
        C_all = _fit_C(M, W)
        result[SIDE_NAME[h]] = C_all.tolist()
        result["residual_mm"][SIDE_NAME[h]] = round(float(np.median(th)), 2)
        result["residual_deg"][SIDE_NAME[h]] = round(float(np.median(rh)), 3)
        result["n_frames"][SIDE_NAME[h]] = int(n)

    # top-level pooled residual (both sides, held-out) for a single headline number
    rm = [v for v in result["residual_mm"].values()]
    rd = [v for v in result["residual_deg"].values()]
    result["residual_mm"]["pooled"] = round(float(np.mean(rm)), 2) if rm else None
    result["residual_deg"]["pooled"] = round(float(np.mean(rd)), 3) if rd else None

    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {out}")
    print(f"  right C residual: {result['residual_mm'].get('right')} mm / "
          f"{result['residual_deg'].get('right')} deg (held-out)")
    print(f"  left  C residual: {result['residual_mm'].get('left')} mm / "
          f"{result['residual_deg'].get('left')} deg (held-out)")
    return result


def load_offsets(path=OUT):
    """-> {'left': (4,4) np, 'right': (4,4) np}. Used by sample.py."""
    with open(path) as f:
        d = json.load(f)
    return {"left": np.array(d["left"], np.float64), "right": np.array(d["right"], np.float64)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_segs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seg_dir", default=SEG_DIR)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    solve(n_segs=a.n_segs, seed=a.seed, seg_dir=a.seg_dir, out=a.out)
