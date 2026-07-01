"""egoaero contract/ dir -> workbench scene.npz."""
import sys, os, numpy as np
def read_obj(p):
    V,F=[],[]
    for ln in open(p):
        if ln.startswith("v "): V.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith("f "): F.append([int(q.split("/")[0])-1 for q in ln.split()[1:4]])
    return np.asarray(V,np.float32), np.asarray(F,np.int32)
def main(cdir, out):
    d = os.path.join(cdir,"contract") if os.path.isdir(os.path.join(cdir,"contract")) else cdir
    hv = np.load(os.path.join(d,"hand_mano.npz"))["verts"].astype(np.float32)
    poses = np.load(os.path.join(d,"object_traj.npz"))["obj_poses_t"].astype(np.float64)
    cm = np.load(os.path.join(d,"contact.npz"))["contact_mask"]
    ov, of = read_obj(os.path.join(d,"object_mesh.obj"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, hand_verts=hv, obj_verts=ov, obj_faces=of, obj_poses=poses,
             contact_mask=cm, source="egoaero (EgoAERO, mock)")
    print(f"wrote {out}: hand {hv.shape}, obj {ov.shape}, T={hv.shape[0]}")
if __name__ == "__main__": main(sys.argv[1], sys.argv[2])
