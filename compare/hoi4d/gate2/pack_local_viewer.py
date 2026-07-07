"""Pack self-contained data files for local_viewer/ from the cached
VGGT-Omega reconstructions.

Copies each clip's vggt_recon.npz content and embeds the GT sensor depth
(HOI4D align_depth) resampled to the model resolution, with intrinsics scaled
to match, so the local viewer needs nothing but the npz.

Usage: python pack_local_viewer.py [--clips a,b]
"""
import argparse, glob, os
import numpy as np
import cv2

CLIPS_ROOT = "/workspace/hoi4d/clips"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_viewer", "data")


def pack(clip):
    name = os.path.basename(clip)
    z = dict(np.load(os.path.join(clip, "gate2", "vggt_recon.npz")))
    S, H, W = z["depth"].shape
    K_full = np.load(os.path.join(clip, "intrin.npy")).astype(np.float64)
    g0 = cv2.imread(os.path.join(clip, "depth", f"{int(z['frame_indices'][0]):06d}.png"),
                    cv2.IMREAD_UNCHANGED)
    Hf, Wf = g0.shape
    K_gt = K_full.copy()
    K_gt[0] *= W / Wf          # fx, cx
    K_gt[1] *= H / Hf          # fy, cy
    gt = []
    for i in z["frame_indices"]:
        g = cv2.imread(os.path.join(clip, "depth", f"{int(i):06d}.png"), cv2.IMREAD_UNCHANGED)
        g = g.astype(np.float32) / 1000.0
        gt.append(cv2.resize(g, (W, H), interpolation=cv2.INTER_NEAREST))
    z["gt_depth"] = np.stack(gt).astype(np.float16)
    z["K_gt"] = K_gt
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"{name}.npz")
    np.savez_compressed(out, **z)
    print(f"packed {name}: {os.path.getsize(out) / 1e6:.0f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="")
    args = ap.parse_args()
    clips = ([os.path.join(CLIPS_ROOT, c) for c in args.clips.split(",") if c]
             or sorted(d for d in glob.glob(os.path.join(CLIPS_ROOT, "*"))
                       if os.path.isdir(d) and
                       os.path.exists(os.path.join(d, "gate2", "vggt_recon.npz"))))
    for c in clips:
        pack(c)


if __name__ == "__main__":
    main()
