"""Score a render_and_compare run on a HOT3D clip against the mocap GT pose
trajectory + scanned eval model (never seen by the pipeline).

Same metrics as the HOI4D version (compare/hoi4d/gate2/sam3d_icp/
gt_pose_eval.py): posed-surface symmetric chamfer, centroid offset, absolute
rotation via a shape-ICP canonical alignment G, and rot_traj via the
trajectory-optimal chordal-mean alignment. HOT3D GT is mocap-grade
(pixel-tight overlays), so unlike HOI4D there is no ~6 mm annotation noise
floor excusing residuals.

Usage: gt_pose_eval_hot3d.py <rc_input_dir> <run_dir> [more_run_dirs...]
"""
import json
import os
import sys

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

DS = "/workspace/datasets/hot3d"
OUT = os.path.dirname(os.path.abspath(__file__))
N, SEED = 15000, 0


def align_canonicals(src, dst):
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
    inp = sys.argv[1].rstrip("/")
    runs = [r.rstrip("/") for r in sys.argv[2:]]
    gt = np.load(f"{inp}/gt_target.npz")
    poses_gt, uid = gt["poses"], int(gt["uid"])
    g = trimesh.load(f"{DS}/object_models_eval/obj_{uid:06d}.glb")
    cad = (trimesh.util.concatenate(list(g.geometry.values()))
           if isinstance(g, trimesh.Scene) else g)
    Vc = np.asarray(trimesh.sample.sample_surface(cad, N, seed=SEED)[0])

    res = {}
    for run in runs:
        n = os.path.basename(run)
        z = np.load(f"{run}/stage8_eval/pseudo_gt.npz")
        mesh = trimesh.Trimesh(z["obj_verts"], z["obj_faces"], process=False)
        Ve = np.asarray(trimesh.sample.sample_surface(mesh, N, seed=SEED)[0])
        T = min(len(poses_gt), len(z["obj_poses"]))
        icp_res, G, _ = align_canonicals(Ve, Vc)
        Rg = poses_gt[:T, :3, :3]
        A = sum(z["obj_poses"][t][:3, :3].T @ Rg[t] for t in range(T))
        U, _, Vt = np.linalg.svd(A)
        S = np.eye(3)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            S[2, 2] = -1
        G_traj = U @ S @ Vt
        cham, cent, rot, rot_tr = [], [], [], []
        for t in range(T):
            Me, Mg = z["obj_poses"][t], poses_gt[t]
            Xe = Ve @ Me[:3, :3].T + Me[:3, 3]
            Xg = Vc @ Mg[:3, :3].T + Mg[:3, 3]
            d1 = cKDTree(Xg).query(Xe[::3], workers=-1)[0]
            d2 = cKDTree(Xe).query(Xg[::3], workers=-1)[0]
            cham.append((np.median(d1) + np.median(d2)) / 2 * 1000)
            cent.append(np.linalg.norm(Xe.mean(0) - Xg.mean(0)) * 100)
            rot.append(np.degrees(Rotation.from_matrix(
                (Me[:3, :3] @ G).T @ Rg[t]).magnitude()))
            rot_tr.append(np.degrees(Rotation.from_matrix(
                (Me[:3, :3] @ G_traj).T @ Rg[t]).magnitude()))
        res[n] = {"canonical_icp_mm": icp_res * 1000,
                  "chamfer_mm_med": float(np.median(cham)),
                  "chamfer_mm_p90": float(np.percentile(cham, 90)),
                  "centroid_cm_med": float(np.median(cent)),
                  "rot_deg_med": float(np.median(rot)),
                  "rot_traj_deg_med": float(np.median(rot_tr)),
                  "rot_traj_deg_p90": float(np.percentile(rot_tr, 90)),
                  "chamfer_mm": np.round(cham, 2).tolist(),
                  "centroid_cm": np.round(cent, 2).tolist(),
                  "rot_deg": np.round(rot, 1).tolist(),
                  "rot_traj_deg": np.round(rot_tr, 1).tolist()}
        r = res[n]
        print(f"{n}: chamfer {r['chamfer_mm_med']:.1f}/{r['chamfer_mm_p90']:.1f} mm"
              f"  centroid {r['centroid_cm_med']:.2f} cm"
              f"  rot {r['rot_deg_med']:.1f} deg"
              f"  rot_traj {r['rot_traj_deg_med']:.1f}/{r['rot_traj_deg_p90']:.1f} deg"
              f"  (canon ICP {r['canonical_icp_mm']:.1f} mm)")
    with open(os.path.join(OUT, "gt_pose_hot3d.json"), "w") as fp:
        json.dump(res, fp, indent=1)


if __name__ == "__main__":
    main()
