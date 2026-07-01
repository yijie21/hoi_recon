"""Figure-1-style composite: one egocentric clip -> 3D HOI reconstruction (SP1) -> simulated
Shadow-Hand execution (SP2), side by side. Left panels are rendered live from the contract;
right panels are sliced from a pre-rendered MuJoCo sim strip (sim_keyframes.png)."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import gridspec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

MANO_BONES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
              (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]

def load_obj(p):
    V,F=[],[]
    for ln in open(p):
        if ln.startswith("v "): V.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith("f "): F.append([int(q.split("/")[0])-1 for q in ln.split()[1:4]])
    return np.array(V), np.array(F)

def set_equal(ax, pts):
    lo,hi=pts.min(0),pts.max(0); c=(lo+hi)/2; r=(hi-lo).max()/2*1.1
    ax.set_xlim(c[0]-r,c[0]+r); ax.set_ylim(c[1]-r,c[1]+r); ax.set_zlim(c[2]-r,c[2]+r)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])

def recon_panel(ax, verts, joints, contact, oV, oF, pose, t):
    ow = oV @ pose[:3,:3].T + pose[:3,3]
    ax.add_collection3d(Poly3DCollection(ow[oF], alpha=0.16, facecolor="#1f77b4", edgecolor="none"))
    ax.scatter(verts[:,0],verts[:,1],verts[:,2], s=1, c="#cfcfcf", alpha=0.25)
    for a,b in MANO_BONES:
        ax.plot([joints[a,0],joints[b,0]],[joints[a,1],joints[b,1]],[joints[a,2],joints[b,2]],c="#d62728",lw=1.5)
    ax.scatter(joints[:,0],joints[:,1],joints[:,2], s=8, c="#d62728")
    cm=contact.astype(bool)
    if cm.any(): ax.scatter(verts[cm,0],verts[cm,1],verts[cm,2], s=12, c="#2ca02c")
    set_equal(ax, np.vstack([verts, ow])); ax.view_init(elev=18, azim=-70)
    ax.set_title(f"frame {t}", fontsize=8, pad=0)

def main(run_dir, sim_strip, out_png):
    d = os.path.join(run_dir, "contract")
    hm = np.load(os.path.join(d,"hand_mano.npz")); verts,joints = hm["verts"],hm["joints"]
    obj = np.load(os.path.join(d,"object_traj.npz"))["obj_poses_t"]
    contact = np.load(os.path.join(d,"contact.npz"))["contact_mask"]
    oV,oF = load_obj(os.path.join(d,"object_mesh.obj"))
    q = json.load(open(os.path.join(run_dir,"quality.json")))
    T = verts.shape[0]
    rks = [0, T//2, T-1]

    strip = mpimg.imread(sim_strip)            # (H, 6*W, 3)
    n = 6; sw = strip.shape[1]//n
    sub = [strip[:, i*sw:(i+1)*sw] for i in range(n)]
    sks = [0, 3, 5]                            # early / mid / late sim keyframes
    sframe = [0, int(round(strip.shape[1]/sw*0)), 0]  # labels filled below

    fig = plt.figure(figsize=(17, 4.5))
    gs = gridspec.GridSpec(1, 7, width_ratios=[1,1,1,0.55,1,1,1], wspace=0.05,
                           top=0.80, bottom=0.14, left=0.02, right=0.98)

    for i,t in enumerate(rks):
        ax = fig.add_subplot(gs[0,i], projection="3d")
        recon_panel(ax, verts[t], joints[t], contact[t], oV, oF, obj[t], t)

    axar = fig.add_subplot(gs[0,3]); axar.axis("off")
    axar.annotate("", xy=(0.95,0.5), xytext=(0.05,0.5), xycoords="axes fraction",
                  arrowprops=dict(arrowstyle="-|>", lw=3, color="#333"))
    axar.text(0.5, 0.60, "two-stage\nresidual policy", ha="center", va="bottom", fontsize=9, color="#333")
    axar.text(0.5, 0.40, "(SP2)", ha="center", va="top", fontsize=8, color="#777")

    simlab = [0, 6, 11]
    for i,(k,fr) in enumerate(zip(sks, simlab)):
        ax = fig.add_subplot(gs[0,4+i]); ax.imshow(sub[k]); ax.axis("off")
        ax.set_title(f"step {fr}", fontsize=8, pad=0)

    # section headers
    fig.text(0.235, 0.85, "3D HOI reconstruction  (SP1)", ha="center", fontsize=13, weight="bold")
    fig.text(0.235, 0.05, "red = MANO hand · blue = object mesh · green = contact",
             ha="center", fontsize=8.5, color="#555")
    fig.text(0.79, 0.85, "Simulated Shadow Hand  (SP2)", ha="center", fontsize=13, weight="bold")
    fig.text(0.79, 0.05, "grey = dexterous hand (orange fingertips) · sphere = object",
             ha="center", fontsize=8.5, color="#555")

    badge = {"accept":"#2ca02c","repairable_accept":"#ff7f0e","recapture":"#d62728"}.get(q["decision"],"k")
    fig.suptitle(f"EgoAERO — one egocentric clip  →  reconstruction  →  simulated execution"
                 f"      [quality: {q['decision'].upper()}  Q={q['Q']:.2f}]",
                 fontsize=14, y=0.965, color="#111")
    fig.patches.append(plt.Rectangle((0.005,0.005),0.99,0.99, transform=fig.transFigure,
                                     fill=False, ec=badge, lw=4))
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    print("wrote", out_png, "| recon frames", rks, "| sim steps", simlab, "| decision", q["decision"])

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
