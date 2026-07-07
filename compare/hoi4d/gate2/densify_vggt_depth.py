"""Densify VGGT-Omega depth to EVERY frame of each HOI4D clip, saved as 16-bit
mm PNGs for render_and_compare's gt-backend injection (RC_GT_DEPTH_DIR).

One forward pass over the full clip (75-150 frames) so the whole sequence
shares a single scale convention — no windowing, no stitching. ~17-20 GB peak
on the 150-frame clips (fits a 32 GB RTX 5090).

Depth is saved at model resolution (384x688 for these clips); RC's gt backend
resizes to frame size with INTER_NEAREST, and the clip's true intrinsics
(intrin.npy) apply because this aspect ratio is a pure resize (no crop).
VGGT-Omega's own intrinsics estimate is stored in meta.json as a diagnostic.

Usage: python densify_vggt_depth.py [--clips a,b]
Output: <clip>/gate2/vggt_depth_mm/%06d.png + meta.json
"""
import argparse, glob, json, os, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate2_extract import CLIPS_ROOT, VGGT_ROOT, load_vggt


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="")
    args = ap.parse_args()
    clips = ([os.path.join(CLIPS_ROOT, c) for c in args.clips.split(",") if c]
             or sorted(d for d in glob.glob(os.path.join(CLIPS_ROOT, "*"))
                       if os.path.isdir(d)))
    sys.path.insert(0, VGGT_ROOT)
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    model = load_vggt()
    for clip in clips:
        name = os.path.basename(clip)
        outdir = os.path.join(clip, "gate2", "vggt_depth_mm")
        paths = sorted(glob.glob(os.path.join(clip, "rgb", "*.jpg")))
        if os.path.exists(os.path.join(outdir, "meta.json")):
            print(f"[skip] {name} (done)", flush=True)
            continue
        images = load_and_preprocess_images(paths, image_resolution=512).cuda()
        with torch.inference_mode():
            pred = model(images)
        H, W = pred["images"].shape[-2:]
        E, K = encoding_to_camera(pred["pose_enc"], (H, W))
        depth = pred["depth"].squeeze().float().cpu().numpy()
        if depth.ndim == 4:
            depth = depth[..., 0]
        del pred
        torch.cuda.empty_cache()
        os.makedirs(outdir, exist_ok=True)
        for i, d in enumerate(depth):
            mm = np.clip(np.nan_to_num(d, nan=0.0) * 1000.0, 0, 65535).astype(np.uint16)
            cv2.imwrite(os.path.join(outdir, f"{i:06d}.png"), mm)
        with open(os.path.join(outdir, "meta.json"), "w") as f:
            json.dump({"frames": len(paths), "model_res": [int(H), int(W)],
                       "K_vggt_frame0": K.squeeze(0)[0].tolist(),
                       "E_range_t": [float(np.abs(E.squeeze(0)[:, :, 3].cpu().numpy()).max())]},
                      f, indent=1)
        print(f"[densify] {name}: {len(paths)} frames at {H}x{W}", flush=True)


if __name__ == "__main__":
    main()
