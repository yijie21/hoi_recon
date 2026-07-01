"""Figure-1-style composite for the REAL clip: wild6.mp4 frames (input) -> real perception
front-end (WiLoR MANO hand + YOLO bottle + Depth-Anything) -> 3D MANO hand. Parallels the
mock figure1_composite.png but every panel is derived from the actual video."""
import os, glob, json
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

OUT="/tmp/wild6/perception"; FR=sorted(glob.glob("/tmp/wild6/frames/*.png"))
VOUT="/workspace/code/hoi_recon/egoaero/viz_output"
BONES=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
       (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
FCOL=["#e41a1c","#ff7f00","#4daf4a","#377eb8","#984ea3"]
def load(fi): return dict(np.load(os.path.join(OUT,f"f{fi:04d}.npz"),allow_pickle=True))

idx=json.load(open(os.path.join(OUT,"index.json")))
both=[r["frame"] for r in idx if r["has_hand"] and r["has_obj"]]
ks=[both[int(x)] for x in np.linspace(0,len(both)-1,3)]

fig=plt.figure(figsize=(17,5.2))
gs=gridspec.GridSpec(1,7,width_ratios=[1,1,1,0.55,1,1,1],wspace=0.06,
                     top=0.80,bottom=0.12,left=0.02,right=0.98)
for i,fi in enumerate(ks):
    ax=fig.add_subplot(gs[0,i]); img=cv2.cvtColor(cv2.imread(FR[fi]),cv2.COLOR_BGR2RGB)
    ax.imshow(img); d=load(fi)
    if bool(d["has_obj"]):
        x0,y0,x1,y1=d["obj_bbox"]; ax.add_patch(plt.Rectangle((x0,y0),x1-x0,y1-y0,fill=False,ec="#00e5ff",lw=2))
    ax.set_xlim(0,img.shape[1]); ax.set_ylim(img.shape[0],0); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"frame {fi}",fontsize=8,pad=0)

axar=fig.add_subplot(gs[0,3]); axar.axis("off")
axar.annotate("",xy=(0.95,0.5),xytext=(0.05,0.5),xycoords="axes fraction",
              arrowprops=dict(arrowstyle="-|>",lw=3,color="#333"))
axar.text(0.5,0.60,"WiLoR + YOLO\n+ Depth-Anything",ha="center",va="bottom",fontsize=8.5,color="#333")
axar.text(0.5,0.40,"(real perception)",ha="center",va="top",fontsize=8,color="#777")

for i,fi in enumerate(ks):
    ax=fig.add_subplot(gs[0,4+i],projection="3d"); d=load(fi)
    V=d["hand_verts3d"]; J=d["hand_joints3d"]
    ax.scatter(V[:,0],V[:,1],V[:,2],s=1,c="#bbb",alpha=0.3)
    for bi,(a,b) in enumerate(BONES):
        ax.plot([J[a,0],J[b,0]],[J[a,1],J[b,1]],[J[a,2],J[b,2]],c=FCOL[bi//4],lw=1.7)
    ax.scatter(J[:,0],J[:,1],J[:,2],s=8,c="k")
    c=V.mean(0); r=np.abs(V-c).max()*1.1
    ax.set_xlim(c[0]-r,c[0]+r); ax.set_ylim(c[1]-r,c[1]+r); ax.set_zlim(c[2]-r,c[2]+r)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.view_init(elev=-75,azim=-90); ax.set_title(f"frame {fi}",fontsize=8,pad=0)

fig.text(0.235,0.85,"Egocentric input — wild6.mp4",ha="center",fontsize=13,weight="bold")
fig.text(0.235,0.05,"cyan box = detected bottle (manipulated object)",ha="center",fontsize=8.5,color="#555")
fig.text(0.79,0.85,"Real 3D MANO hand  (WiLoR)",ha="center",fontsize=13,weight="bold")
fig.text(0.79,0.05,"thumb=red index=orange middle=green ring=blue pinky=purple",ha="center",fontsize=8.5,color="#555")
fig.suptitle("wild6.mp4  →  real perception front-end  →  3D hand   "
             "[partial: no metric object mesh / 6-DoF track / ego-SLAM — models absent]",
             fontsize=13,y=0.965,color="#111")
fig.patches.append(plt.Rectangle((0.005,0.005),0.99,0.99,transform=fig.transFigure,fill=False,ec="#888",lw=3))
fig.savefig(os.path.join(VOUT,"wild6_figure1.png"),dpi=120,bbox_inches="tight")
print("wrote wild6_figure1.png keyframes",ks)
