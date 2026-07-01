"""HORT per-frame outputs (a folder of <name>.obj + <name>.json) -> temporal scene.npz.

HORT is per-frame feed-forward (no temporal smoothing) — each frame is reconstructed in its own
camera frame, and its ABSOLUTE depth is noisy: on wild6 the hand centroid swings ~1.15 m in z
across frames while x/y barely move. Shown raw this looks like the whole HOI teleporting in depth,
and it inflates the scene bbox ~6x so the workbench normalizer shrinks the hand.

That absolute per-frame placement is not something HORT actually estimates reliably, so we
RECENTER each frame on the hand centroid (subtracting the SAME translation from hand and object,
which preserves the grasp / object-relative-to-hand pose). The result is a stable view of what
HORT does estimate: hand shape + articulation + object pose relative to the hand.
Pass --no-recenter to keep the raw absolute frames.

Usage: python compare/adapters/hort_seq_to_scene.py hort/out_wild6 compare/scenes/hort_wild6.npz [n_obj_pts] [--no-recenter]
"""
import sys, os, glob, json, numpy as np
def read_obj(p):
    V,F=[],[]
    for ln in open(p):
        s=ln.split()
        if s[:1]==["v"]: V.append([float(x) for x in s[1:4]])
        elif s[:1]==["f"]: F.append([int(q.split("/")[0])-1 for q in s[1:4]])
    return np.asarray(V,np.float32), np.asarray(F,np.int32)
def main(folder, out, npts=2000, recenter=True):
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
        ow=ow[sel]
        if recenter:
            # remove noisy per-frame absolute placement; keep grasp (same shift on hand+object)
            c=V.mean(0)
            V=V-c; ow=ow-c
        HV.append(V); OP.append(ow)
    HV=np.asarray(HV,np.float32); OP=np.asarray(OP,np.float32)
    os.makedirs(os.path.dirname(out),exist_ok=True)
    tag="per-frame, hand-centred" if recenter else "per-frame, raw depth"
    np.savez(out, hand_verts=HV, hand_faces=faces, obj_points=OP,
             obj_point_colors=np.tile([90,150,230],(OP.shape[1],1)).astype(np.uint8),
             source=f"HORT (wild6, {len(stems)}f, {tag})")
    print(f"wrote {out}: hand {HV.shape}, obj_points {OP.shape}, recenter={recenter}")
if __name__=="__main__":
    args=[a for a in sys.argv[1:] if a!="--no-recenter"]
    recenter="--no-recenter" not in sys.argv
    main(args[0], args[1], int(args[2]) if len(args)>2 else 2000, recenter)
