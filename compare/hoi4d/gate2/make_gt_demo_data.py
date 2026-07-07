"""Pack GT-sensor-depth versions of the local 4D viewer data — same schema,
same 48 frames, same resolution and masks as the VGGT-Omega packages, so each
clip appears twice in the viewer dropdown (<clip> vs <clip>__gtdepth) for
direct A/B comparison.

GT convention: static camera at the origin (identity extrinsics — HOI4D rigs
are fixed; VGGT-Omega itself measured <2 cm centre motion), true intrinsics
scaled to model resolution, uniform confidence.

Usage: python make_gt_demo_data.py   (reuses rgb/masks from vggt_recon.npz)
"""
import glob, os
import numpy as np
import cv2

CLIPS_ROOT = "/workspace/hoi4d/clips"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_viewer", "data")


def main():
    os.makedirs(OUT, exist_ok=True)
    for recon in sorted(glob.glob(os.path.join(CLIPS_ROOT, "*", "gate2", "vggt_recon.npz"))):
        clip = os.path.dirname(os.path.dirname(recon))
        name = os.path.basename(clip)
        z = np.load(recon)
        idxs, rgb, hand, obj = z["frame_indices"], z["rgb"], z["hand"], z["obj"]
        S, H, W = z["depth"].shape
        g0 = cv2.imread(os.path.join(clip, "depth", f"{int(idxs[0]):06d}.png"),
                        cv2.IMREAD_UNCHANGED)
        Hf, Wf = g0.shape
        K = np.load(os.path.join(clip, "intrin.npy")).astype(np.float64)
        K[0] *= W / Wf
        K[1] *= H / Hf
        gt = []
        for i in idxs:
            g = cv2.imread(os.path.join(clip, "depth", f"{int(i):06d}.png"),
                           cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
            gt.append(cv2.resize(g, (W, H), interpolation=cv2.INTER_NEAREST))
        gt = np.stack(gt).astype(np.float16)
        E = np.tile(np.hstack([np.eye(3), np.zeros((3, 1))])[None], (S, 1, 1)).astype(np.float32)
        out = os.path.join(OUT, f"{name}__gtdepth.npz")
        np.savez_compressed(out, frame_indices=idxs, depth=gt,
                            conf=np.ones_like(gt), rgb=rgb,
                            K=np.tile(K[None], (S, 1, 1)).astype(np.float32), E=E,
                            hand=hand, obj=obj, gt_depth=gt, K_gt=K)
        print(f"packed {name}__gtdepth: {os.path.getsize(out) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
