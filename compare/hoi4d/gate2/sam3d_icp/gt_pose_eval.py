"""THE non-circular verdict: score every arm against HOI4D's GT object pose
annotations + CAD model (never seen by any pipeline stage).

Setup facts (verified in-session, 2026-07-08): kettle_N15 is HOI4D sequence
ZY20210800002/H2/C12/N15/S196/s04/T2, our 75 frames = original frames 0-74
(pixel-matched, offset 0); objpose euler convention is INTRINSIC "XYZ"
(picked by chamfer against the depth cloud: 6-16 mm vs 11-21 mm for "xyz" —
that 6-16 mm is also the GT annotation's own noise floor vs the aligned
depth, so sub-cm differences between arms are below what GT can decide).

Per arm and frame:
- chamfer_mm: symmetric median surface-to-surface distance between the posed
  estimated mesh and the posed GT CAD (canonical-frame-free placement error).
- centroid_cm: distance between the two posed surfaces' centroids.
- rot_deg: geodesic( R_est @ G , R_gt ) where G is the constant
  SAM-3D-canonical -> CAD-canonical rotation solved by multi-start rigid ICP
  (24 octahedral starts). This is ABSOLUTE orientation error, including any
  constant flip inherited from the stage-3 tracker — the azimuth-flip test.

Usage: gt_pose_eval.py [run_suffix ...]  (default: icp2 icp4 icp5 icpj3)
"""
import json
import os
import sys

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

RC = "/workspace/code/hoi_recon/render_and_compare/runs"
DS = "/workspace/datasets/hoi4d"
SEQ = "ZY20210800002/H2/C12/N15/S196/s04/T2"
CAD = f"{DS}/HOI4D_CAD_Model_for_release/rigid/Kettle/015.obj"
OUT = os.path.dirname(os.path.abspath(__file__))
N, SEED, T = 15000, 0, 75


def gt_poses():
    Rs, ts = [], []
    for t in range(T):
        d = json.load(open(f"{DS}/HOI4D_annotations/{SEQ}/objpose/{t}.json"))["dataList"][0]
        eul = [d["rotation"]["x"], d["rotation"]["y"], d["rotation"]["z"]]
        Rs.append(Rotation.from_euler("XYZ", eul).as_matrix())
        ts.append([d["center"]["x"], d["center"]["y"], d["center"]["z"]])
    return np.array(Rs), np.array(ts)


def align_canonicals(src, dst):
    """Constant rotation G (+ translation) mapping the SAM-3D canonical mesh
    onto the CAD canonical: rigid multi-start ICP, 24 octahedral inits."""
    best = (np.inf, None, None)
    tree = cKDTree(dst)
    for g0 in Rotation.create_group("O").as_matrix():
        R, t = g0, dst.mean(0) - g0 @ src.mean(0)
        for _ in range(40):
            d, j = tree.query(src @ R.T + t, workers=-1)
            keep = d <= np.quantile(d, 0.9)
            mu_s, mu_d = src[keep].mean(0), dst[j[keep]].mean(0)
            cov = (dst[j[keep]] - mu_d).T @ (src[keep] - mu_s)
            U, _, Vt = np.linalg.svd(cov)
            S = np.eye(3)
            if np.linalg.det(U) * np.linalg.det(Vt) < 0:
                S[2, 2] = -1
            R = U @ S @ Vt
            t = mu_d - R @ mu_s
        res = float(np.median(cKDTree(dst).query(src @ R.T + t, workers=-1)[0]))
        if res < best[0]:
            best = (res, R, t)
    return best


def main():
    names = sys.argv[1:] or ["icp2", "icp4", "icp5", "icpj3"]
    cad = trimesh.load(CAD, force="mesh")
    Vc = np.asarray(trimesh.sample.sample_surface(cad, N, seed=SEED)[0])
    Rg, tg = gt_poses()

    res = {}
    for n in names:
        z = np.load(f"{RC}/kettle_gt_{n}/stage8_eval/pseudo_gt.npz")
        mesh = trimesh.Trimesh(z["obj_verts"], z["obj_faces"], process=False)
        Ve = np.asarray(trimesh.sample.sample_surface(mesh, N, seed=SEED)[0])
        icp_res, G, _ = align_canonicals(Ve, Vc)
        # trajectory-optimal constant alignment (chordal mean): removes ANY
        # constant canonical/azimuth offset, leaving pure per-frame
        # orientation-tracking error. rot (shape-G) minus rot_traj isolates
        # the constant flip — but shape-G itself can land in a wrong azimuth
        # basin on this near-revolution kettle, so treat rot_traj as the
        # reliable cross-arm rotation ranking.
        A = sum(z["obj_poses"][t][:3, :3].T @ Rg[t] for t in range(T))
        U, _, Vt = np.linalg.svd(A)
        S = np.eye(3)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            S[2, 2] = -1
        G_traj = U @ S @ Vt
        cham, cent, rot, rot_tr = [], [], [], []
        for t in range(T):
            Me = z["obj_poses"][t]
            Xe = Ve @ Me[:3, :3].T + Me[:3, 3]
            Xg = Vc @ Rg[t].T + tg[t]
            d1 = cKDTree(Xg).query(Xe[::3], workers=-1)[0]
            d2 = cKDTree(Xe).query(Xg[::3], workers=-1)[0]
            cham.append((np.median(d1) + np.median(d2)) / 2 * 1000)
            cent.append(np.linalg.norm(Xe.mean(0) - Xg.mean(0)) * 100)
            rot.append(np.degrees(Rotation.from_matrix(
                (Me[:3, :3] @ G).T @ Rg[t]).magnitude()))
            rot_tr.append(np.degrees(Rotation.from_matrix(
                (Me[:3, :3] @ G_traj).T @ Rg[t]).magnitude()))
        res[n] = {"canonical_icp_mm": icp_res * 1000,
                  "rot_traj_deg_med": float(np.median(rot_tr)),
                  "rot_traj_deg_p90": float(np.percentile(rot_tr, 90)),
                  "rot_traj_deg": np.round(rot_tr, 1).tolist(),
                  "chamfer_mm_med": float(np.median(cham)),
                  "chamfer_mm_p90": float(np.percentile(cham, 90)),
                  "centroid_cm_med": float(np.median(cent)),
                  "rot_deg_med": float(np.median(rot)),
                  "rot_deg_p90": float(np.percentile(rot, 90)),
                  "chamfer_mm": np.round(cham, 2).tolist(),
                  "centroid_cm": np.round(cent, 2).tolist(),
                  "rot_deg": np.round(rot, 1).tolist()}
        print(f"{n:6s} chamfer {res[n]['chamfer_mm_med']:5.1f}/{res[n]['chamfer_mm_p90']:5.1f} mm"
              f"  centroid {res[n]['centroid_cm_med']:5.2f} cm"
              f"  rot {res[n]['rot_deg_med']:6.1f}/{res[n]['rot_deg_p90']:6.1f} deg"
              f"  rot_traj {res[n]['rot_traj_deg_med']:5.1f}/{res[n]['rot_traj_deg_p90']:5.1f} deg"
              f"  (canon ICP {res[n]['canonical_icp_mm']:.1f} mm)")
    with open(os.path.join(OUT, f"gt_pose_ab_{'_'.join(names)}.json"), "w") as fp:
        json.dump(res, fp, indent=1)


if __name__ == "__main__":
    main()
