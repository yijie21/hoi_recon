"""Three-panel 'current best strategy' HOI overlay on the rectified HOT3D clip:
GT (green obj + tan hands) | icpjgr estimate | any6dp estimate. Objects splatted from
canonical mesh @ per-frame pose; HANDS splatted from per-frame posed meshes in a distinct
tan colour so both the object and the grasping hand(s) are visible.

Hand sources (see extract_gt_hands.py / T5_NOTES):
  - GT hands  = UmeTrack (mocap-grade, `gt_hands.npz` in the rc_input; NOT MANO).
  - est hands = the pipeline's HaMeR/HaWoR MANO hand, grasp-optimized in stage 7
    (`hand_verts` in pseudo_gt + `hand_faces` in stage7). Left-hand clips are unreliable
    (fabricated MANO_LEFT); GT UmeTrack is correct either way.

Usage: make_best_overlay.py <cat> <num>  [out.mp4]     # e.g. mug_white 001970
2D overlay caveat: it shows silhouette agreement, not the full 3D verdict.
"""
import os
import sys

import cv2
import numpy as np
import trimesh

DS = "/workspace/datasets/hot3d"
RC = "/workspace/code/hoi_recon/render_and_compare"
B, N = 2, 200000
HAND_COL = (150, 180, 235)          # tan (BGR), all hands
FONT = cv2.FONT_HERSHEY_SIMPLEX


def sample(mesh):
    P, fidx = trimesh.sample.sample_surface(mesh, N, seed=0)
    return np.asarray(P), np.asarray(mesh.face_normals)[fidx]


def load_run(run):
    z = np.load(f"{run}/stage8_eval/pseudo_gt.npz")
    P, Nrm = sample(trimesh.Trimesh(z["obj_verts"], z["obj_faces"], process=False))
    hv = z["hand_verts"] if "hand_verts" in z.files else None
    hf = None
    if hv is not None:
        s7 = f"{run}/stage7_contact_optim/arrays.npz"
        if os.path.exists(s7):
            zz = np.load(s7)
            hf = zz["hand_faces"] if "hand_faces" in zz.files else None
    return P, Nrm, z["obj_poses"], (hv if hf is not None else None), hf


def bary(faces, n=40000, seed=1):
    rng = np.random.default_rng(seed)
    fi = rng.integers(0, len(faces), n)
    b = rng.random((n, 2))
    flip = b.sum(1) > 1
    b[flip] = 1 - b[flip]
    return fi, np.column_stack([1 - b.sum(1), b])   # (n,), (n,3) barycentric


def posed(verts, faces, fi, w):
    tri = verts[faces[fi]]                            # (n,3,3)
    pts = (tri * w[:, :, None]).sum(1)
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.clip(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9, None)
    return pts, nrm


def main():
    cat, num = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else f"overlays/best/hoi3_{cat}_{num}.mp4"
    inp = f"{DS}/rc_input_{num}_{cat}"
    icp_run = f"{RC}/runs/hot3d_{cat}_{num}_icpjgr"
    a6_run = f"{RC}/runs/hot3d_{cat}_{num}_any6dp"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    gt = np.load(f"{inp}/gt_target.npz")
    K, poses_gt, uid = gt["K"], gt["poses"], int(gt["uid"])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    g = trimesh.load(f"{DS}/object_models_eval/obj_{uid:06d}.glb")
    cad = (trimesh.util.concatenate(list(g.geometry.values()))
           if isinstance(g, trimesh.Scene) else g)
    Pc, Ncn = sample(cad)
    Pi, Ni, poses_i, hvi, hfi = load_run(icp_run)
    Pa, Na, poses_a, hva, hfa = load_run(a6_run)

    # GT UmeTrack hands (per-frame, already in camera frame)
    gh = None
    if os.path.exists(f"{inp}/gt_hands.npz"):
        z = np.load(f"{inp}/gt_hands.npz", allow_pickle=True)
        gh = (z["verts"], z["faces"])
    # precompute barycentric sample indices per hand topology
    gh_bary = bary(gh[1]) if gh is not None and len(gh[1]) else None
    hfi_bary = bary(hfi) if hfi is not None else None
    hfa_bary = bary(hfa) if hfa is not None else None

    frames_dir = f"{icp_run}/stage0_preprocess/frames"
    H, W = cv2.imread(f"{frames_dir}/00000.jpg").shape[:2]
    Wd, Hd = W // B, H // B

    def overlay(o, X, Nrm, base):
        z = X[:, 2]
        ok = z > 0.05
        if not ok.any():
            return o
        u = np.clip((X[ok, 0] / z[ok] * fx + cx) / B, 0, Wd - 1).astype(int)
        v = np.clip((X[ok, 1] / z[ok] * fy + cy) / B, 0, Hd - 1).astype(int)
        lam = 0.35 + 0.65 * np.clip(-(Nrm[ok] @ np.array([0.3, -0.5, -0.81])), 0, 1)
        col = (np.asarray(base)[None] * lam[:, None]).astype(np.uint8)
        order = np.argsort(z[ok])[::-1]
        lay = np.zeros((Hd, Wd, 3), np.uint8)
        cov = np.zeros((Hd, Wd), bool)
        lay[v[order], u[order]] = col[order]
        cov[v, u] = True
        cov = cv2.morphologyEx(cov.astype(np.uint8), cv2.MORPH_CLOSE,
                               np.ones((3, 3), np.uint8)) > 0
        hole = cov & (lay.sum(2) == 0)
        lay[hole] = cv2.blur(lay, (3, 3))[hole]
        o = o.copy()
        o[cov] = (0.42 * o[cov] + 0.58 * lay[cov]).astype(np.uint8)
        return o

    def hands_for(kind, t):
        """(pts, normals) list of posed hand splats for panel `kind` at frame t."""
        out = []
        if kind == "gt" and gh is not None:
            for hv in gh[0][t]:                      # (Nv,3) camera-frame verts
                out.append(posed(hv, gh[1], *gh_bary))
        elif kind == "icp" and hvi is not None:
            out.append(posed(hvi[t], hfi, *hfi_bary))
        elif kind == "a6" and hva is not None:
            out.append(posed(hva[t], hfa, *hfa_bary))
        return out

    T = min(len(poses_gt), len(poses_i), len(poses_a))
    raw = out + ".raw.mp4"
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), 30, (Wd * 3, Hd))
    panels = [(Pc, Ncn, poses_gt, (80, 220, 80), "gt", "GT (mocap): obj + UmeTrack hands"),
              (Pi, Ni, poses_i, (80, 160, 255), "icp", "icpjgr (rotation-robust)"),
              (Pa, Na, poses_a, (255, 170, 80), "a6", "any6dp (placement-optimal)")]
    for t in range(T):
        img = cv2.resize(cv2.imread(f"{frames_dir}/{t:05d}.jpg"), (Wd, Hd))
        cells = []
        for P, Nrm, poses, base, kind, txt in panels:
            M = poses[t]
            im = overlay(img, P @ M[:3, :3].T + M[:3, 3], Nrm @ M[:3, :3].T, base)
            for hp, hn in hands_for(kind, t):        # tan hand(s) on top
                im = overlay(im, hp, hn, HAND_COL)
            cv2.putText(im, txt, (10, 26), FONT, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(im, txt, (10, 26), FONT, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(im, f"frame {t}", (10, Hd - 12), FONT, 0.55,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cells.append(im)
        vw.write(np.hstack(cells))
    vw.release()
    os.system(f"ffmpeg -y -loglevel error -i {raw} -c:v libx264 -crf 20 "
              f"-pix_fmt yuv420p {out} && rm {raw}")
    print("wrote", out)


if __name__ == "__main__":
    main()
