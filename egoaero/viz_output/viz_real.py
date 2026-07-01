"""Visualize the real perception pass on wild6.mp4: RGB + MANO-hand/bottle overlay,
the 3D MANO hand (WiLoR), and monocular depth. Builds a keyframe figure + an overlay GIF."""
import os, glob, json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

OUT = "/tmp/wild6/perception"
FR = sorted(glob.glob("/tmp/wild6/frames/*.png"))
VOUT = "/workspace/code/hoi_recon/egoaero/viz_output"
os.makedirs(VOUT, exist_ok=True)

BONES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
         (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(20-1+1-1+1,20)]
BONES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
         (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
FCOL = ["#e41a1c","#ff7f00","#4daf4a","#377eb8","#984ea3"]  # thumb..pinky

def load(fi):
    return dict(np.load(os.path.join(OUT, f"f{fi:04d}.npz"), allow_pickle=True))

def draw_overlay(ax, fi):
    img = cv2.cvtColor(cv2.imread(FR[fi]), cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    d = load(fi)
    if bool(d["has_hand"]):
        kp = d["hand_kp2d"]
        for bi,(a,b) in enumerate(BONES):
            ax.plot([kp[a,0],kp[b,0]],[kp[a,1],kp[b,1]], c=FCOL[bi//4], lw=2)
        ax.scatter(kp[:,0],kp[:,1], s=9, c="yellow", edgecolors="k", linewidths=0.4, zorder=3)
    if bool(d["has_obj"]):
        x0,y0,x1,y1 = d["obj_bbox"]
        ax.add_patch(plt.Rectangle((x0,y0),x1-x0,y1-y0, fill=False, ec="#00e5ff", lw=2.2))
        ax.text(x0, y0-6, "bottle", color="#00e5ff", fontsize=8, weight="bold")
    ax.set_xlim(0, img.shape[1]); ax.set_ylim(img.shape[0], 0)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"frame {fi}", fontsize=9)

def draw_hand3d(ax, fi):
    d = load(fi)
    if not bool(d["has_hand"]):
        ax.text(0.5,0.5,"no hand"); return
    V = d["hand_verts3d"]; J = d["hand_joints3d"]
    ax.scatter(V[:,0],V[:,1],V[:,2], s=1, c="#bbbbbb", alpha=0.3)
    for bi,(a,b) in enumerate(BONES):
        ax.plot([J[a,0],J[b,0]],[J[a,1],J[b,1]],[J[a,2],J[b,2]], c=FCOL[bi//4], lw=1.8)
    ax.scatter(J[:,0],J[:,1],J[:,2], s=10, c="k")
    c=V.mean(0); r=np.abs(V-c).max()*1.1
    ax.set_xlim(c[0]-r,c[0]+r); ax.set_ylim(c[1]-r,c[1]+r); ax.set_zlim(c[2]-r,c[2]+r)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.view_init(elev=-75, azim=-90)   # roughly camera-facing
    ax.set_title(f"MANO hand · frame {fi}", fontsize=9)

idx = json.load(open(os.path.join(OUT, "index.json")))
both = [r["frame"] for r in idx if r["has_hand"] and r["has_obj"]]
ks = [both[int(x)] for x in np.linspace(0, len(both)-1, 4)]

# ---- figure: row1 overlay, row2 3D hand, + one depth panel ----
fig = plt.figure(figsize=(15, 8.2))
gs = gridspec.GridSpec(2, 4, height_ratios=[1,1], hspace=0.12, wspace=0.08)
for i,fi in enumerate(ks):
    draw_overlay(fig.add_subplot(gs[0,i]), fi)
    draw_hand3d(fig.add_subplot(gs[1,i], projection="3d"), fi)
fig.suptitle("Real perception on wild6.mp4 — WiLoR MANO hand + YOLO bottle  (top: RGB overlay · bottom: 3D hand)",
             fontsize=13)
fig.text(0.5, 0.06, "thumb=red index=orange middle=green ring=blue pinky=purple · cyan box = detected bottle",
         ha="center", fontsize=9, color="#555")
fig.savefig(os.path.join(VOUT,"wild6_perception.png"), dpi=115, bbox_inches="tight")
print("wrote wild6_perception.png keyframes", ks)

# ---- depth panel ----
dfi = [r["frame"] for r in idx if r["frame"] % 5 == 0][:4]
figd, axs = plt.subplots(1, 4, figsize=(15, 4.5))
for ax,fi in zip(axs, dfi):
    dd = load(fi)
    if "depth" in dd:
        im=ax.imshow(dd["depth"], cmap="turbo"); ax.set_title(f"depth · frame {fi}", fontsize=9)
    ax.axis("off")
figd.suptitle("Monocular relative depth (Depth-Anything-V2) on wild6.mp4", fontsize=13)
figd.savefig(os.path.join(VOUT,"wild6_depth.png"), dpi=115, bbox_inches="tight")
print("wrote wild6_depth.png frames", dfi)

# ---- overlay GIF over all frames ----
import imageio.v2 as imageio
gif_frames=[]
for fi in range(len(FR)):
    f = plt.figure(figsize=(3.6,6.4)); ax=f.add_subplot(111); draw_overlay(ax, fi)
    f.tight_layout(pad=0); f.canvas.draw()
    buf=np.frombuffer(f.canvas.buffer_rgba(), np.uint8).reshape(f.canvas.get_width_height()[::-1]+(4,))
    gif_frames.append(buf[...,:3].copy()); plt.close(f)
imageio.mimsave(os.path.join(VOUT,"wild6_overlay.gif"), gif_frames, fps=10)
print("wrote wild6_overlay.gif", len(gif_frames), "frames")
