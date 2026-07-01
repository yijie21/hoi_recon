"""render_and_compare pseudo_gt.npz -> workbench scene.npz."""
import sys, os, numpy as np
def main(npz, out):
    z = np.load(npz, allow_pickle=True)
    a = {k: z[k] for k in z.files}
    kw = dict(hand_verts=a["hand_verts"].astype(np.float32),
              obj_verts=a["obj_verts"].astype(np.float32),
              obj_faces=a["obj_faces"].astype(np.int32),
              obj_poses=a["obj_poses"].astype(np.float64),
              source="render_and_compare (CHOIR, mock)")
    if "hand_faces" in a and a["hand_faces"].dtype != object:
        kw["hand_faces"] = a["hand_faces"].astype(np.int32)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, **kw)
    print(f"wrote {out}: hand {kw['hand_verts'].shape}, obj {kw['obj_verts'].shape}")
if __name__ == "__main__": main(sys.argv[1], sys.argv[2])
