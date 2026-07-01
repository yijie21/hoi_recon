"""Backproject each method's 3D HOI reconstruction onto the frames it consumed, so the pixel
alignment (is the reconstruction where the real hand/object is?) can be judged directly.

Each method uses its OWN camera + image, exactly as it produced the reconstruction — no
cross-method re-warping — so what you see is that method's true reprojection error:

  render_and_compare : full-frame 720x1280 original frames, K=[1158.4;cx360;cy640],
                       geometry in OpenCV camera frame (stage7 arrays.npz). Hand mesh + object mesh.
  forehoi            : its own 518x518 processed frames + per-frame K + object pose (frames.npz)
                       + object_mesh.glb. Hand mesh + object mesh.
  hort               : HORT only runs on a 224 crop and never saved the crop box, so its native
                       reprojection lives on that crop; it already writes per-frame overlay JPGs
                       (mesh+object over the flipped crop). We un-flip them (left hand) into a video.

Meshes are rasterized painter's-algorithm (depth-sorted alpha-blended triangles) with cv2 — no GL,
so it runs headless. Hand = green, object = blue.

Usage:
  PY=/workspace/miniconda3/envs/forehoi/bin/python
  $PY compare/backproject.py rc     compare/backproj/rc
  $PY compare/backproject.py forehoi compare/backproj/forehoi
  $PY compare/backproject.py hort   compare/backproj/hort
"""
import sys, os, glob, json
import numpy as np
import cv2

HAND = (60, 200, 70)      # BGR green
OBJ  = (230, 130, 70)     # BGR blue


def project(K, X):
    """X:[N,3] camera-frame metres -> u,v pixels [N,2] and depth z [N]."""
    z = np.clip(X[:, 2], 1e-4, None)
    u = X[:, 0] / z * K[0, 0] + K[0, 2]
    v = X[:, 1] / z * K[1, 1] + K[1, 2]
    return np.stack([u, v], 1), z


def draw_mesh(img, uv, z, faces, color, alpha=0.45, wire=True):
    """Depth-sorted filled triangles + optional wireframe, alpha-blended onto img in place."""
    ov = img.copy()
    order = np.argsort(-z[faces].mean(1))          # far -> near
    for f in faces[order]:
        tri = uv[f].astype(np.int32)
        if not np.isfinite(tri).all():
            continue
        cv2.fillConvexPoly(ov, tri, color, lineType=cv2.LINE_AA)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)
    if wire:
        for f in faces[order][::7]:                # sparse edges for shape read-out
            tri = uv[f].astype(np.int32)
            cv2.polylines(img, [tri], True, tuple(int(c*0.6) for c in color), 1, cv2.LINE_AA)


def draw_points(img, uv, color, r=2):
    for p in uv.astype(np.int32):
        cv2.circle(img, tuple(p), r, color, -1, cv2.LINE_AA)


def load_rc():
    root = "render_and_compare/runs/wild6_real"
    K = np.load(f"{root}/stage0_preprocess/arrays.npz")["intrinsics"].astype(np.float64)
    a = np.load(f"{root}/stage7_contact_optim/arrays.npz", allow_pickle=True)
    hv, hf = a["hand_verts"], a["hand_faces"].astype(np.int32)
    ov, of, op = a["obj_verts"], a["obj_faces"].astype(np.int32), a["obj_poses"]
    frames = sorted(glob.glob(f"{root}/stage0_preprocess/frames/*.jpg"))
    T = len(hv)
    for t in range(T):
        img = cv2.imread(frames[t])
        ow = ov @ op[t][:3, :3].T + op[t][:3, 3]
        uo, zo = project(K, ow); draw_mesh(img, uo, zo, of, OBJ)
        uh, zh = project(K, hv[t]); draw_mesh(img, uh, zh, hf, HAND)
        yield img, f"render_and_compare  frame {t+1}/{T}"


def load_forehoi():
    z = np.load("forehoi/output_4d/wild6/frames.npz", allow_pickle=True)
    import trimesh
    m = trimesh.load("forehoi/output_4d/wild6/object_mesh.glb", force="mesh")
    ov, of = np.asarray(m.vertices), np.asarray(m.faces).astype(np.int32)
    rgb, K, poses = z["rgb"], z["K"], z["poses"]
    hv, hf = z["hand_verts"], z["hand_faces"].astype(np.int32)
    T = len(rgb)
    for t in range(T):
        img = cv2.cvtColor(rgb[t], cv2.COLOR_RGB2BGR).copy()
        ow = ov @ poses[t][:3, :3].T + poses[t][:3, 3]
        uo, zo = project(K[t], ow); draw_mesh(img, uo, zo, of, OBJ)
        if z["hand_count"][t] > 0:
            uh, zh = project(K[t], hv[t, 0]); draw_mesh(img, uh, zh, hf, HAND)
        yield img, f"forehoi  frame {t+1}/{T}"


def load_hort():
    # HORT's own overlay JPGs are its native reprojection (mesh+object over the flipped 224 crop).
    # Un-flip horizontally so the hand reads as the true left hand; the overlay flips with the image
    # so alignment is preserved.
    jpgs = sorted(glob.glob("hort/out_wild6/*.jpg"))
    T = len(jpgs)
    for t, p in enumerate(jpgs):
        img = cv2.imread(p)[:, ::-1]                # un-flip
        img = cv2.resize(img, (448, 448), interpolation=cv2.INTER_NEAREST)
        yield img.copy(), f"HORT  frame {t+1}/{T}  (model 224 crop, un-flipped)"


LOADERS = {"rc": load_rc, "forehoi": load_forehoi, "hort": load_hort}


def main(method, outdir):
    os.makedirs(outdir, exist_ok=True)
    frames, labels = [], []
    for img, label in LOADERS[method]():
        cv2.putText(img, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        frames.append(img)
    H, W = frames[0].shape[:2]
    vw = cv2.VideoWriter(f"{outdir}/overlay.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 7.5, (W, H))
    for f in frames:
        vw.write(f)
    vw.release()
    # contact sheet of ~6 evenly spaced frames for quick static inspection
    idx = np.linspace(0, len(frames) - 1, min(6, len(frames))).round().astype(int)
    sheet = np.hstack([cv2.resize(frames[i], (W * 320 // H, 320)) for i in idx])
    cv2.imwrite(f"{outdir}/contact_sheet.png", sheet)
    print(f"{method}: {len(frames)} frames -> {outdir}/overlay.mp4  +  contact_sheet.png  ({W}x{H})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
