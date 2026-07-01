"""Render workbench scene.npz files to a static PNG montage (one panel per method) —
a quick visual comparison without launching the interactive viser server.

  python compare/snapshot.py compare/out.png compare/scenes/hort.npz compare/scenes/forehoi.npz
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def obj_world(d, t):
    if "obj_points" in d and np.asarray(d["obj_points"]).dtype != object:
        return np.asarray(d["obj_points"])[t], None
    P = np.asarray(d["obj_poses"])[t]
    return np.asarray(d["obj_verts"]) @ P[:3, :3].T + P[:3, 3], np.asarray(d["obj_faces"])


def panel(ax, d):
    src = str(d.get("source", "?"))
    hv = np.asarray(d["hand_verts"]); T = hv.shape[0]; t = T // 2
    ow, of = obj_world(d, t)
    # object
    if of is not None and len(of) < 200000:
        ax.add_collection3d(Poly3DCollection(ow[of[::max(1, len(of)//40000)]],
                            alpha=0.12, facecolor="#1f77b4", edgecolor="none"))
    else:
        ax.scatter(ow[:, 0], ow[:, 1], ow[:, 2], s=2, c="#1f77b4", alpha=0.5)
    # hand
    hf = d.get("hand_faces")
    if hf is not None and np.asarray(hf).dtype != object:
        ax.add_collection3d(Poly3DCollection(hv[t][np.asarray(hf)], alpha=0.5,
                            facecolor="#e0a890", edgecolor="none"))
    else:
        ax.scatter(hv[t][:, 0], hv[t][:, 1], hv[t][:, 2], s=2, c="#d62728")
    allp = np.vstack([hv[t], ow]); c = allp.mean(0); r = np.abs(allp - c).max() * 1.1
    ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.view_init(elev=-75, azim=-90)   # OpenCV camera frame: y-down
    ax.set_title(f"{src}\nframe {t}/{T-1}", fontsize=10)


def main(out, paths):
    n = len(paths)
    fig = plt.figure(figsize=(5 * n, 5.2))
    for i, p in enumerate(paths):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        panel(ax, load(p))
    fig.suptitle("HOI methods workbench — hand + object reconstruction (skin=hand, blue=object)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
