"""Route B — post-hoc shape refinement of the SAM-3D mesh against the fused
multi-frame GT-depth object cloud (kettle_N15).

Fuse: per frame, backproject GT depth in the 5px-eroded object mask and pull
into the canonical object frame via the stage-4 ICP poses of kettle_gt_icp2
(pure registration, before stages 5-7). Deform: optimize per-vertex
displacements + ONE global log-scale under
    L = data  : one-sided point->mesh-face distance (fused cloud -> surface),
                top 5% trimmed (sensor noise / mask bleed)
    + lap     : Laplacian of the DISPLACEMENT field (smooth deformation,
                preserves the generated shape detail)
    + anchor  : |d|^2 (stay near the generative prior)
The back of the object (never observed) is constrained only by lap+anchor —
exactly the "generative prior fills what depth can't see" division of labor.

Runs in the sam3d5090 env (pytorch3d). Outputs refined_verts.npz + stats.
"""
import json
import os

import cv2
import numpy as np
import torch

RC = "/workspace/code/hoi_recon/render_and_compare/runs"
GT = f"{RC}/kettle_gt"
ICP2 = f"{RC}/kettle_gt_icp2"
OUT = os.path.dirname(os.path.abspath(__file__))
ERODE, SEED = 5, 0
N_PER_FRAME, N_FUSED = 3000, 80000
ITERS, LR = 400, 2e-3
W_DATA, W_LAP, W_ANCHOR, TRIM = 1.0, 30.0, 5.0, 0.95


def fused_cloud(poses):
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ERODE + 1,) * 2)
    K = np.load(f"{GT}/stage0_preprocess/arrays.npz")["intrinsics"]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    rng = np.random.default_rng(SEED)
    pts = []
    for t in range(len(poses)):
        g = np.load(f"{GT}/stage0_preprocess/depth/{t:05d}.npy").astype(np.float32)
        m = (cv2.erode(np.load(f"{GT}/stage1_detect_track/masks/{t:05d}.npy")
                       .astype(np.uint8), ker) > 0) & (g > 0.25) & (g < 5.0)
        ys, xs = np.nonzero(m)
        z = g[ys, xs]
        P = np.stack([(xs - cx) / fx * z, (ys - cy) / fy * z, z], 1)
        if len(P) > N_PER_FRAME:
            P = P[rng.choice(len(P), N_PER_FRAME, replace=False)]
        R, tt = poses[t][:3, :3], poses[t][:3, 3]
        pts.append((P - tt) @ R)               # camera -> canonical
    pts = np.concatenate(pts)
    if len(pts) > N_FUSED:
        pts = pts[rng.choice(len(pts), N_FUSED, replace=False)]
    return pts.astype(np.float32)


def uniform_laplacian(V, F):
    """Sparse uniform Laplacian L (V x V): (mean of neighbors) - self."""
    import scipy.sparse as sp
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.unique(np.sort(e, 1), axis=0)
    n = len(V)
    A = sp.coo_matrix((np.ones(len(e) * 2),
                       (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                      shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(1)).ravel()
    Dinv = sp.diags(1.0 / np.maximum(deg, 1))
    L = (Dinv @ A - sp.eye(n)).tocoo()
    idx = torch.tensor(np.vstack([L.row, L.col]), dtype=torch.long)
    val = torch.tensor(L.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, (n, n)).coalesce()


def main():
    from pytorch3d.structures import Meshes, Pointclouds
    from pytorch3d.loss.point_mesh_distance import point_face_distance

    s3 = np.load(f"{ICP2}/stage3_object/arrays.npz")
    s4 = np.load(f"{ICP2}/stage4_align/arrays.npz")
    verts0, faces = s3["verts"].astype(np.float32), s3["faces"].astype(np.int64)
    cloud = fused_cloud(s4["obj_poses"])
    print(f"fused cloud {len(cloud)} pts; mesh {len(verts0)} verts")

    dev = "cuda"
    V0 = torch.tensor(verts0, device=dev)
    Fc = torch.tensor(faces, device=dev)
    P = torch.tensor(cloud, device=dev)
    Lap = uniform_laplacian(verts0, faces).to(dev)
    d = torch.zeros_like(V0, requires_grad=True)
    log_s = torch.zeros(1, device=dev, requires_grad=True)
    opt = torch.optim.Adam([d, log_s], lr=LR)

    pcl = Pointclouds([P])
    points = pcl.points_packed()
    pfi = pcl.cloud_to_packed_first_idx()
    maxp = int(pcl.num_points_per_cloud().max())

    def data_term(V):
        m = Meshes([V], [Fc])
        tris = m.verts_packed()[m.faces_packed()]
        tfi = m.mesh_to_faces_packed_first_idx()
        d2 = point_face_distance(points, pfi, tris, tfi, maxp)
        keep = d2 <= torch.quantile(d2, TRIM)
        return d2[keep].mean(), d2

    with torch.no_grad():
        _, d2_0 = data_term(V0)
    for it in range(ITERS):
        V = torch.exp(log_s) * (V0 + d)
        data, d2 = data_term(V)
        lap = torch.sparse.mm(Lap, d).pow(2).sum(1).mean()
        anch = d.pow(2).sum(1).mean()
        loss = W_DATA * data + W_LAP * lap + W_ANCHOR * anch
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 100 == 0 or it == ITERS - 1:
            print(f"it {it:3d} data {float(data)*1e6:8.2f}um2 "
                  f"cloud->mesh med {float(d2.detach().median().sqrt())*1000:.2f}mm "
                  f"|d|med {float(d.detach().norm(dim=1).median())*1000:.2f}mm "
                  f"scale {float(torch.exp(log_s)):.4f}")

    with torch.no_grad():
        Vr = (torch.exp(log_s) * (V0 + d)).cpu().numpy()
        _, d2_r = data_term(torch.tensor(Vr, device=dev))
    stats = {
        "cloud_to_mesh_med_mm_before": float(np.sqrt(d2_0.cpu().numpy()).mean() and
                                             np.median(np.sqrt(d2_0.cpu().numpy())) * 1000),
        "cloud_to_mesh_med_mm_after": float(np.median(np.sqrt(d2_r.cpu().numpy())) * 1000),
        "cloud_to_mesh_p90_mm_before": float(np.percentile(np.sqrt(d2_0.cpu().numpy()), 90) * 1000),
        "cloud_to_mesh_p90_mm_after": float(np.percentile(np.sqrt(d2_r.cpu().numpy()), 90) * 1000),
        "disp_med_mm": float(np.median(np.linalg.norm(Vr - verts0, axis=1)) * 1000),
        "global_scale": float(np.exp(log_s.detach().cpu().numpy()[0])),
        "n_cloud": int(len(cloud)),
    }
    print(json.dumps(stats, indent=1))
    np.savez(os.path.join(OUT, "refined_verts.npz"), verts=Vr, faces=faces,
             colors=s3["colors"], **{f"stat_{k}": v for k, v in stats.items()})
    with open(os.path.join(OUT, "route_b_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)


if __name__ == "__main__":
    main()
