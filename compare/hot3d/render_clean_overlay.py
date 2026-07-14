"""Validation overlays for clean-clip windows (gen_clean_clips.py). For each sampled
window, splat the TARGET object (green) + OTHER present objects (gray, dim) + both
hands (skin) over the egocentric RGB, across the window's frames. Renders a montage
PNG (4 frames: start..end) so we can eyeball that ONE object is manipulated and the
others stay put. Reuses gt_overlay_hot3d geometry.

Usage: render_clean_overlay.py <manifest.json> --n 20 --out_dir overlays/clean
"""
import argparse, glob, json, os
import cv2, numpy as np, trimesh
from hand_tracking_toolkit import camera as htt_camera
from hand_tracking_toolkit import dataset as htt_dataset
from hand_tracking_toolkit.hand_models.umetrack_hand_model import forward_kinematics as umetrack_fk
from scipy.spatial.transform import Rotation

DS = "/workspace/datasets/hot3d"; STREAM = "214-1"; B = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
TARGET_RGB = np.array([80, 230, 90])     # green (BGR) — the manipulated object
OTHER_RGB  = np.array([150, 150, 150])   # gray — other present objects
HAND_RGB   = np.array([140, 170, 240])   # skin


def T_from(d):
    R = Rotation.from_quat(np.roll(d["quaternion_wxyz"], -1)).as_matrix()
    T = np.eye(4); T[:3, :3], T[:3, 3] = R, d["translation_xyz"]; return T


_C = {}
def obj_pn(uid):
    if uid not in _C:
        g = trimesh.load(f"{DS}/object_models_eval/obj_{int(uid):06d}.glb")
        m = trimesh.util.concatenate(list(g.geometry.values())) if isinstance(g, trimesh.Scene) else g
        P, fi = trimesh.sample.sample_surface(m, 40000, seed=0)
        _C[uid] = (np.asarray(P), np.asarray(m.face_normals)[fi])
    return _C[uid]


def splat(sm, X, C, z, cam, W, H, Wd, Hd):
    ang = np.arccos(np.clip(z / np.linalg.norm(X, axis=1), -1, 1))
    ok = (z > 0.05) & (ang < 1.3)
    uv = np.asarray(cam.eye_to_window(X[ok]))
    inb = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    u = (uv[inb, 0] / B).astype(int); v = (uv[inb, 1] / B).astype(int)
    Ci, zi = C[ok][inb], z[ok][inb]
    order = np.argsort(zi)[::-1]
    lay = np.zeros((Hd, Wd, 3), np.uint8); cov = np.zeros((Hd, Wd), bool)
    lay[v[order], u[order]] = Ci[order]; cov[v, u] = True
    cov = cv2.morphologyEx(cov.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    sm[cov] = (0.4 * sm[cov] + 0.6 * lay[cov]).astype(np.uint8)


def render_frame(clip_dir, fi, target_uid, um_shape):
    stem = sorted(glob.glob(f"{clip_dir}/*.objects.json"))[fi][:-len(".objects.json")]
    img = cv2.imread(f"{stem}.image_{STREAM}.jpg"); H, W = img.shape[:2]
    Wd, Hd = W // B, H // B; sm = cv2.resize(img, (Wd, Hd))
    cams = json.load(open(f"{stem}.cameras.json"))[STREAM]
    cam = htt_camera.from_json(cams); T_cw = np.linalg.inv(T_from(cams["T_world_from_camera"]))
    anno = json.load(open(f"{stem}.objects.json"))
    # objects: target green, others gray
    for uid, e in anno.items():
        P, Nrm = obj_pn(uid)
        T_co = T_cw @ T_from(e[0]["T_world_from_object"])
        X = P @ T_co[:3, :3].T + T_co[:3, 3]; Nc = Nrm @ T_co[:3, :3].T
        lam = 0.4 + 0.6 * np.clip(-(Nc @ np.array([0.3, -0.5, -0.81])), 0, 1)
        base = TARGET_RGB if uid == target_uid else OTHER_RGB
        splat(sm, X, (base[None] * lam[:, None]).astype(np.uint8), X[:, 2], cam, W, H, Wd, Hd)
    # hands
    for side, pc in htt_dataset.decode_hand_pose(json.load(open(f"{stem}.hands.json"))).items():
        if pc.umetrack is None: continue
        _, v, f = umetrack_fk(pc.umetrack, um_shape, requires_mesh=True)
        hm = trimesh.Trimesh(v.detach().numpy().astype(np.float64), f.detach().numpy().astype(np.int64), process=False)
        Ph, fih = trimesh.sample.sample_surface(hm, 15000, seed=0)
        X = np.asarray(Ph) @ T_cw[:3, :3].T + T_cw[:3, 3]
        Nc = np.asarray(hm.face_normals)[fih] @ T_cw[:3, :3].T
        lam = 0.4 + 0.6 * np.clip(-(Nc @ np.array([0.3, -0.5, -0.81])), 0, 1)
        splat(sm, X, (HAND_RGB[None] * lam[:, None]).astype(np.uint8), X[:, 2], cam, W, H, Wd, Hd)
    return cv2.rotate(sm, cv2.ROTATE_90_CLOCKWISE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest"); ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out_dir", default="overlays/clean"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    wins = json.load(open(a.manifest))["windows"]
    # diverse sample: spread across distinct objects, seeded
    import random; random.seed(a.seed)
    by_obj = {}
    for w in wins: by_obj.setdefault(w["object_name"], []).append(w)
    picks, pool = [], list(by_obj.values())
    random.shuffle(pool)
    for grp in pool:
        picks.append(random.choice(grp))
        if len(picks) >= a.n: break
    while len(picks) < a.n and len(picks) < len(wins):
        w = random.choice(wins);
        if w not in picks: picks.append(w)

    for i, w in enumerate(picks):
        cd = f"{DS}/clips/{w['source_clip']}"
        shp = json.load(open(f"{cd}/__hand_shapes.json__"))
        um = htt_dataset.from_umetrack_hand_model_json(shp["umetrack"])
        fa, fb = w["frame_start"], w["frame_end"]
        idxs = np.linspace(fa, fb, 4).astype(int)
        panels = [render_frame(cd, fi, w["target_uid"], um) for fi in idxs]
        row = np.concatenate(panels, axis=1)
        lab = (f"{w['source_clip']} | TARGET(green)={w['object_name']} uid{w['target_uid']} | "
               f"f{fa}-{fb} ({w['duration_s']}s) | hands={w['num_hands']} | "
               f"mov t{w['motion']['trans_cum_mm']}mm r{w['motion']['rot_range_deg']}deg | "
               f"vis {w['visibility']['vis214_median']}")
        bar = np.zeros((30, row.shape[1], 3), np.uint8)
        cv2.putText(bar, lab, (6, 20), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        out = np.concatenate([bar, row], axis=0)
        p = f"{a.out_dir}/clean_{i:02d}_{w['object_name']}.png"
        cv2.imwrite(p, out); print("wrote", p, flush=True)
    # contact sheet of all rows
    rows = [cv2.imread(f"{a.out_dir}/clean_{i:02d}_{w['object_name']}.png") for i, w in enumerate(picks)]
    wmax = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 6, 0, wmax - r.shape[1], cv2.BORDER_CONSTANT) for r in rows]
    sheet = np.concatenate(rows, axis=0)
    sp = f"{a.out_dir}/_contact_sheet.png"; cv2.imwrite(sp, sheet)
    print("wrote", sp, sheet.shape)


if __name__ == "__main__":
    main()
