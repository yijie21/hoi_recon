"""Visualize an EgoAERO 4D-HOI reconstruction contract (hand MANO + object mesh +
6-DoF object trajectory + contact) as a multi-panel figure and an animated GIF."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

# MANO 21-joint kinematic tree (root + 5 fingers x 4 joints)
MANO_BONES = [
    (0,1),(1,2),(2,3),(3,4),       # thumb
    (0,5),(5,6),(6,7),(7,8),       # index
    (0,9),(9,10),(10,11),(11,12),  # middle
    (0,13),(13,14),(14,15),(15,16),# ring
    (0,17),(17,18),(18,19),(19,20),# little
]

def load_obj(path):
    V, F = [], []
    for ln in open(path):
        if ln.startswith("v "):
            V.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith("f "):
            F.append([int(p.split("/")[0]) - 1 for p in ln.split()[1:4]])
    return np.array(V), np.array(F)

def obj_world(verts, pose):
    return verts @ pose[:3,:3].T + pose[:3,3]

def set_equal(ax, pts):
    lo, hi = pts.min(0), pts.max(0)
    c = (lo+hi)/2; r = (hi-lo).max()/2 * 1.1
    ax.set_xlim(c[0]-r,c[0]+r); ax.set_ylim(c[1]-r,c[1]+r); ax.set_zlim(c[2]-r,c[2]+r)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

def draw_frame(ax, verts_t, joints_t, contact_t, oV, oF, opose, t, title):
    ow = obj_world(oV, opose)
    # object mesh as light triangles (subsample faces for speed)
    tri = ow[oF]
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    pc = Poly3DCollection(tri, alpha=0.18, facecolor="#1f77b4", edgecolor="none")
    ax.add_collection3d(pc)
    # hand vertices (faint) + skeleton
    hv = verts_t
    ax.scatter(hv[:,0],hv[:,1],hv[:,2], s=1, c="#cccccc", alpha=0.25)
    jt = joints_t
    for a,b in MANO_BONES:
        ax.plot([jt[a,0],jt[b,0]],[jt[a,1],jt[b,1]],[jt[a,2],jt[b,2]],
                c="#d62728", lw=1.6)
    ax.scatter(jt[:,0],jt[:,1],jt[:,2], s=10, c="#d62728")
    # contact vertices highlighted
    cm = contact_t.astype(bool)
    if cm.any():
        ax.scatter(hv[cm,0],hv[cm,1],hv[cm,2], s=14, c="#2ca02c", label="contact")
    allp = np.vstack([hv, ow])
    set_equal(ax, allp)
    ax.set_title(title, fontsize=9)
    ax.view_init(elev=18, azim=-70)

def main(run_dir, out_png, out_gif=None):
    d = os.path.join(run_dir, "contract")
    if not os.path.isdir(d):  # dataset seq dir stores files flat
        d = run_dir
    hm = np.load(os.path.join(d, "hand_mano.npz"))
    verts, joints = hm["verts"], hm["joints"]
    obj = np.load(os.path.join(d, "object_traj.npz"))["obj_poses_t"]
    contact = np.load(os.path.join(d, "contact.npz"))["contact_mask"]
    oV, oF = load_obj(os.path.join(d, "object_mesh.obj"))
    T = verts.shape[0]
    qpath = os.path.join(run_dir, "quality.json")
    if not os.path.exists(qpath): qpath = os.path.join(d, "quality.json")
    q = json.load(open(qpath)) if os.path.exists(qpath) else {}

    # ---------- multi-panel summary figure ----------
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 4, height_ratios=[2.4, 1])
    # 4 keyframes
    ks = np.linspace(0, T-1, 4).astype(int)
    for i, t in enumerate(ks):
        ax = fig.add_subplot(gs[0, i], projection="3d")
        draw_frame(ax, verts[t], joints[t], contact[t], oV, oF, obj[t], t,
                   f"frame {t}/{T-1}")

    # contact count over time
    axc = fig.add_subplot(gs[1, 0])
    axc.plot(contact.sum(1), c="#2ca02c"); axc.fill_between(range(T), contact.sum(1), alpha=.3, color="#2ca02c")
    axc.set_title("contact verts / frame", fontsize=9); axc.set_xlabel("frame")

    # object translation trajectory
    axt = fig.add_subplot(gs[1, 1])
    tr = obj[:, :3, 3]
    for j,lab in zip(range(3),"xyz"): axt.plot(tr[:,j], label=lab)
    axt.legend(fontsize=7); axt.set_title("object translation (m)", fontsize=9); axt.set_xlabel("frame")

    # object speed
    axs = fig.add_subplot(gs[1, 2])
    spd = np.r_[0, np.linalg.norm(np.diff(tr,axis=0),axis=1)]
    axs.plot(spd, c="#9467bd"); axs.set_title("object speed (m/frame)", fontsize=9); axs.set_xlabel("frame")

    # quality summary text
    axq = fig.add_subplot(gs[1, 3]); axq.axis("off")
    lines = [f"DECISION: {q.get('decision','?').upper()}",
             f"Q = {q.get('Q',float('nan')):.3f}",
             f"B_repair = {q.get('B_repair',float('nan')):.3f}",
             f"R_after  = {q.get('R_after',float('nan')):.3f}",
             f"U_unres  = {q.get('U_unresolved',float('nan')):.3f}",
             f"frames = {T}"]
    pf = q.get("per_finger", {})
    if pf:
        lines.append("")
        lines.append("per-finger gap before->after (mm):")
        for f, v in pf.items():
            lines.append(f"  {f:6s} {v['gap_before_mm']:6.1f} -> {v['gap_after_mm']:5.1f}")
    col = {"accept":"#2ca02c","repairable_accept":"#ff7f0e","recapture":"#d62728"}.get(q.get("decision"),"k")
    axq.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=8.5,
             transform=axq.transAxes)
    axq.add_patch(plt.Rectangle((0,0),1,1, transform=axq.transAxes, fill=False, ec=col, lw=3))

    fig.suptitle(f"EgoAERO 4D-HOI reconstruction — {os.path.basename(run_dir.rstrip('/'))}", fontsize=13)
    fig.tight_layout(rect=[0,0,1,0.97])
    fig.savefig(out_png, dpi=110)
    print("wrote", out_png)

    # ---------- animated GIF ----------
    if out_gif:
        try:
            from matplotlib.animation import FuncAnimation, PillowWriter
        except Exception as e:
            print("gif skipped:", e); return
        figa = plt.figure(figsize=(6,6)); axa = figa.add_subplot(111, projection="3d")
        def upd(t):
            axa.cla()
            draw_frame(axa, verts[t], joints[t], contact[t], oV, oF, obj[t], t,
                       f"{os.path.basename(run_dir.rstrip('/'))}  frame {t}/{T-1}  "
                       f"[{q.get('decision','?')}]")
        an = FuncAnimation(figa, upd, frames=T, interval=140)
        an.save(out_gif, writer=PillowWriter(fps=7))
        print("wrote", out_gif)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
