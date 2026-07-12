"""Best-method HOI overlay: the best OBJECT track (fpauto) + the best HAND
(hand-reprojection optimizer) splatted together onto the original frames.

3 panels: [ original | object only | object + hand ] so the full 4D HOI reconstruction
is visible backprojected. Object and hand faces are z-sorted together (render_multi), so
they occlude each other correctly.

Usage:
  make_hoi_best_overlay.py <obj_run> <hand_run> <out.mp4>
    obj_run  : run with stage8_eval/pseudo_gt.npz (fpauto object track)
    hand_run : run with hand_reproj_opt/out.npz + stage2_hand + stage0 frames (icpjgr)
"""
import argparse
import glob
import os

import cv2
import numpy as np

from hoi_recon.viz.reproject import render_multi, _label, SKIN
from hoi_recon.bundle import Bundle

OBJ_RGB = np.array([90, 175, 235], np.float32)   # object tint (blue) vs SKIN hand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("obj_run")
    ap.add_argument("hand_run")
    ap.add_argument("out")
    ap.add_argument("--obj_label", default="fpauto")
    ap.add_argument("--fps", type=float, default=24.0)
    a = ap.parse_args()

    hrun = a.hand_run.rstrip("/")
    frames = sorted(glob.glob(os.path.join(hrun, "stage0_preprocess", "frames", "*.jpg")))
    K = Bundle.load(os.path.join(hrun, "stage0_preprocess"))["intrinsics"]

    o = np.load(os.path.join(a.obj_run.rstrip("/"), "stage8_eval", "pseudo_gt.npz"))
    ov, ofc, op = o["obj_verts"], o["obj_faces"].astype(np.int32), o["obj_poses"]
    hv = np.load(os.path.join(hrun, "hand_reproj_opt", "out.npz"))["hand_verts"]
    hfc = np.load(os.path.join(hrun, "stage2_hand", "arrays.npz"))["hand_faces"].astype(np.int32)
    obj_frgb = np.tile(OBJ_RGB, (len(ofc), 1)); hand_frgb = np.tile(SKIN, (len(hfc), 1))
    T = min(len(frames), len(op), len(hv))

    H, W = cv2.imread(frames[0]).shape[:2]
    sw, sh = W // 2, H // 2
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (3 * sw, sh))
    grid, gframes = [], {38, 75, 110, 145}
    for t in range(T):
        fr = cv2.imread(frames[t])
        R, tt = op[t][:3, :3], op[t][:3, 3]
        ow = ov @ R.T + tt
        obj_only = render_multi(fr, [(ow, ofc, obj_frgb)], K, alpha=0.7)
        hoi = render_multi(fr, [(ow, ofc, obj_frgb), (hv[t], hfc, hand_frgb)], K, alpha=0.65)
        row = np.hstack([
            _label(cv2.resize(fr, (sw, sh)), "original"),
            _label(cv2.resize(obj_only, (sw, sh)), f"object ({a.obj_label})  f{t}"),
            _label(cv2.resize(hoi, (sw, sh)), f"object + hand  f{t}")])
        vw.write(row)
        if t in gframes:
            grid.append(row)
    vw.release()
    if grid:
        cv2.imwrite(a.out.replace(".mp4", "_grid.png"), np.vstack(grid))
    print(f"wrote {a.out} ({T} frames)")


if __name__ == "__main__":
    main()
