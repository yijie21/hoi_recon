"""EasyHOI partial output -> workbench scene.npz.
Hand: HaMeR MANO mesh (real), HaMeR/OpenGL frame -> flip to OpenCV (y,z) to match others.
Object: InstantMesh canonical [-1,1] shape (real) but NOT pose-aligned (opt stage blocked);
        normalized to ~hand scale and placed at the palm as a PROXY. Labeled honestly.
Usage: python compare/adapters/easyhoi_to_scene.py easyhoi/data_run compare/scenes/easyhoi.npz
"""
import sys, os, numpy as np
def read_obj(p):
    V,F=[],[]
    for ln in open(p):
        s=ln.split()
        if not s: continue
        if s[0]=="v": V.append([float(x) for x in s[1:4]])
        elif s[0]=="f": F.append([int(q.split("/")[0])-1 for q in s[1:4]])
    return np.asarray(V,np.float32), np.asarray(F,np.int32)
def main(run, out):
    hv,hf = read_obj(os.path.join(run,"hamer","f0030_0.obj"))
    ov,of = read_obj(os.path.join(run,"obj_recon/results/instantmesh/instant-mesh-large/meshes/f0030/full.obj"))
    flip=np.array([1,-1,-1],np.float32)          # OpenGL(HaMeR) -> OpenCV (match HORT/ForeHOI)
    hv=hv*flip
    # object: normalize canonical mesh to ~9cm and place at hand centroid (PROXY; not aligned)
    ov=ov-ov.mean(0); ov=ov/ (np.abs(ov).max()+1e-9) * 0.045
    ov=ov*flip + hv.mean(0)
    os.makedirs(os.path.dirname(out),exist_ok=True)
    np.savez(out, hand_verts=hv[None], hand_faces=hf,
             obj_verts=ov, obj_faces=of, obj_poses=np.eye(4)[None],
             source="EasyHOI (hand real; object shape real, pose NOT aligned)")
    print(f"wrote {out}: hand {hv.shape}, obj {ov.shape}")
if __name__=="__main__": main(sys.argv[1],sys.argv[2])
