"""HORT per-frame outputs (a folder of <name>.obj + <name>.json) -> temporal scene.npz.
HORT is per-frame feed-forward (no temporal smoothing) — each frame is in its own camera frame.
Usage: python compare/adapters/hort_seq_to_scene.py hort/out_wild6 compare/scenes/hort_wild6.npz [n_obj_pts]
"""
import sys, os, glob, json, numpy as np
def read_obj(p):
    V,F=[],[]
    for ln in open(p):
        s=ln.split()
        if s[:1]==["v"]: V.append([float(x) for x in s[1:4]])
        elif s[:1]==["f"]: F.append([int(q.split("/")[0])-1 for q in s[1:4]])
    return np.asarray(V,np.float32), np.asarray(F,np.int32)
def main(folder, out, npts=2000):
    stems=sorted(g[:-4] for g in glob.glob(os.path.join(folder,"*.obj")))
    rng=np.random.default_rng(0)
    HV=[]; OP=[]; faces=None
    for st in stems:
        V,F=read_obj(st+".obj"); faces=F
        j=json.load(open(st+".json"))
        pc=np.asarray(j["pointclouds_up"],np.float32)
        palm=np.asarray(j["handpalm"],np.float32).reshape(3)
        tr=np.asarray(j["objtrans"],np.float32).reshape(3)
        ow=pc+palm+tr
        sel=rng.choice(len(ow), min(npts,len(ow)), replace=False)
        HV.append(V); OP.append(ow[sel])
    HV=np.asarray(HV,np.float32); OP=np.asarray(OP,np.float32)
    os.makedirs(os.path.dirname(out),exist_ok=True)
    np.savez(out, hand_verts=HV, hand_faces=faces, obj_points=OP,
             obj_point_colors=np.tile([90,150,230],(OP.shape[1],1)).astype(np.uint8),
             source=f"HORT (wild6, {len(stems)}f, per-frame)")
    print(f"wrote {out}: hand {HV.shape}, obj_points {OP.shape}")
if __name__=="__main__": main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv)>3 else 2000)
