# Consistent camera + depth geometry for the hoi_recon pipeline via VGGT.
#
# Runs INSIDE the sam3d-objects conda env (numpy<2 / torch 2.5 match VGGT's pins).
# Given the video frames, VGGT jointly predicts a globally-consistent camera
# trajectory (extrinsics world->cam, OpenCV) + per-frame intrinsics + consistent
# depth maps -- replacing per-frame monocular MoGe depth + the static-camera
# assumption. Output saved to a .npz the pipeline loads.
#
#   python vggt_geom.py --frames_dir F --out geo.npz [--ckpt model.pt] [--max_frames 192]
import os
import sys
import glob
import argparse

import numpy as np
import torch

_VGGT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _VGGT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=None, help="local model.pt (else HF download)")
    ap.add_argument("--max_frames", type=int, default=192,
                    help="uniformly subsample to at most this many frames (memory)")
    ap.add_argument("--mode", default="pad", choices=["pad", "crop"],
                    help="pad preserves all pixels (needed for portrait HOI frames)")
    a = ap.parse_args()

    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    dev = "cuda"
    paths = sorted(glob.glob(os.path.join(a.frames_dir, "*.jpg")))
    T = len(paths)
    idx = (np.unique(np.linspace(0, T - 1, a.max_frames).round().astype(int))
           if T > a.max_frames else np.arange(T))
    sel = [paths[i] for i in idx]
    print(f"[vggt] {len(sel)}/{T} frames")

    model = VGGT()
    if a.ckpt and os.path.exists(a.ckpt):
        sd = torch.load(a.ckpt, map_location="cpu")
        model.load_state_dict(sd)
        print(f"[vggt] loaded local ckpt {a.ckpt}")
    else:
        model = VGGT.from_pretrained("facebook/VGGT-1B")
        print("[vggt] loaded HF checkpoint")
    model = model.to(dev).eval()

    # original frame size + the pad-mode content rectangle (for mapping depth back)
    from PIL import Image
    ow, oh = Image.open(sel[0]).size                        # (W,H)
    target = 518
    if a.mode == "pad":
        if ow >= oh:
            nw = target; nh = round(oh * (nw / ow) / 14) * 14
        else:
            nh = target; nw = round(ow * (nh / oh) / 14) * 14
        pad_left = (target - nw) // 2; pad_top = (target - nh) // 2
    else:
        nw = target; nh = round(oh * (nw / ow) / 14) * 14
        pad_left = 0; pad_top = -((nh - target) // 2 if nh > target else 0)
    content_rect = np.array([pad_left, pad_top, nw, nh])     # x0,y0,w,h in the 518 image

    images = load_and_preprocess_images(sel, mode=a.mode).to(dev)   # (N,3,H,W) processed
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        agg, ps_idx = model.aggregator(images[None])
        pose_enc = model.camera_head(agg)[-1]
        extr, intr = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        depth, dconf = model.depth_head(agg, images[None], ps_idx)

    extr = extr[0].float().cpu().numpy()                    # (N,3,4) world->cam
    intr = intr[0].float().cpu().numpy()                    # (N,3,3) at processed res
    depth = depth[0].float().cpu().numpy()                  # (N,H,W,1)
    if depth.ndim == 4:
        depth = depth[..., 0]
    dconf = dconf[0].float().cpu().numpy()                  # (N,H,W)
    procH, procW = int(images.shape[-2]), int(images.shape[-1])

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, extrinsic=extr, intrinsic=intr, depth=depth.astype(np.float16),
             conf=dconf.astype(np.float16), proc_hw=np.array([procH, procW]),
             content_rect=content_rect, orig_hw=np.array([oh, ow]),
             sel_idx=idx, n_total=T)
    print(f"[vggt] wrote {a.out}: extr{extr.shape} intr{intr.shape} depth{depth.shape} "
          f"proc {procH}x{procW}; depth range {depth[depth>0].min():.3f}-{depth.max():.3f}")


if __name__ == "__main__":
    main()
