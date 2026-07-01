"""Export the wild6.mp4 real reconstruction to a self-contained viser scene:
  - hand   : real WiLoR MANO mesh (778 verts + 1538 MANO faces), per frame, camera-metric
  - object : bottle pixels back-projected from Depth-Anything depth, scale-aligned to the
             hand's metric depth (monocular relative depth -> approximate; documented).
Writes /tmp/wild6/viser_scene.npz (+ a matplotlib alignment snapshot for verification)."""
import os, glob, json, pickle
import numpy as np

OUT = "/tmp/wild6/perception"
FR = sorted(glob.glob("/tmp/wild6/frames/*.png"))
MANO_F = np.asarray(pickle.load(open(
    "/workspace/miniconda3/envs/forehoi/lib/python3.11/site-packages/wilor_mini/"
    "pretrained_models/MANO_RIGHT.pkl", "rb"), encoding="latin1")["f"]).astype(np.int32)

def load(fi):
    return dict(np.load(os.path.join(OUT, f"f{fi:04d}.npz"), allow_pickle=True))

idx = json.load(open(os.path.join(OUT, "index.json")))
# use frames that have the hand; object points only where bottle is detected
frames = [r["frame"] for r in idx if r["has_hand"]]

hand_V, obj_P = [], []
M_OBJ = 600  # object points per frame (padded/subsampled to fixed size)
for fi in frames:
    d = load(fi)
    V = d["hand_verts3d"].astype(np.float32) + d["hand_cam_t"].astype(np.float32)  # camera-metric
    hand_V.append(V)

    pts = np.full((M_OBJ, 3), np.nan, np.float32)
    if bool(d["has_obj"]):
        H, W = d["img_hw"].astype(int)
        f = float(d["hand_focal"]); cx, cy = W / 2.0, H / 2.0
        z_hand = float(np.median(V[:, 2]))               # grasp depth = hand metric z
        # back-project a grid of bbox pixels to the constant grasp-depth plane.
        # Honest object proxy: the detected bottle's image footprint at the hand's
        # depth — NO 3D object reconstruction (no asset, no tracked 6-DoF/mesh).
        x0, y0, x1, y1 = d["obj_bbox"].astype(float)
        gx, gy = np.meshgrid(np.linspace(x0, x1, 20), np.linspace(y0, y1, 30))
        xs = gx.ravel(); ys = gy.ravel()
        X = (xs - cx) / f * z_hand; Y = (ys - cy) / f * z_hand
        P = np.stack([X, Y, np.full_like(X, z_hand)], 1).astype(np.float32)
        pts[:len(P)] = P[:M_OBJ]
    obj_P.append(pts)

hand_V = np.asarray(hand_V, np.float32)                   # [T,778,3]
obj_P = np.asarray(obj_P, np.float32)                     # [T,M,3] (nan where missing)
# replace nan rows with the frame centroid so viser gets finite points (rendered tight)
for t in range(len(obj_P)):
    m = np.isnan(obj_P[t]).any(1)
    if m.all():
        obj_P[t][:] = hand_V[t].mean(0)
    elif m.any():
        obj_P[t][m] = np.nanmean(obj_P[t][~m], 0)

np.savez("/tmp/wild6/viser_scene.npz",
         hand_verts=hand_V, hand_faces=MANO_F,
         obj_points=obj_P,
         obj_point_colors=np.tile([235, 235, 235], (M_OBJ, 1)).astype(np.uint8),
         source="wild6.mp4 — WiLoR MANO hand + bottle bbox @ grasp-depth plane (no 3D object recon)")
print("wrote /tmp/wild6/viser_scene.npz  hand", hand_V.shape, "obj", obj_P.shape)

# ---- alignment snapshot (matplotlib) ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ti = len(frames) // 2
fig = plt.figure(figsize=(6, 6)); ax = fig.add_subplot(111, projection="3d")
V = hand_V[ti]; ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=MANO_F,
                                color="#e0b89a", alpha=0.6, linewidth=0)
O = obj_P[ti]; ax.scatter(O[:, 0], O[:, 1], O[:, 2], s=4, c="#3070d0")
ax.set_title(f"wild6 viser scene · frame index {ti}"); ax.view_init(elev=-75, azim=-90)
allp = np.vstack([V, O]); c = allp.mean(0); r = np.abs(allp - c).max()
ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
fig.savefig("/workspace/code/hoi_recon/egoaero/viz_output/wild6_viser_check.png", dpi=110, bbox_inches="tight")
print("wrote wild6_viser_check.png")
