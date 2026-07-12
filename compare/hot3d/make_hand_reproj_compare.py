"""3-panel hand-reprojection comparison: [original | BEFORE | AFTER].
BEFORE = the run's current pipeline hand (stage7 joint_grasp, no image term).
AFTER  = the image-aligned hand from run_hand_reproj.py (kp2d + hand-sil + soft contact).
Both splatted the same way (viz.reproject.render_multi) so only the hand pose differs.

Usage: make_hand_reproj_compare.py <run_dir> <out.mp4> [--after run/hand_reproj_opt/out.npz]
"""
import argparse
import glob
import os

import cv2
import numpy as np

from hoi_recon.viz.reproject import render_multi, _label, SKIN
from hoi_recon.bundle import Bundle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("out")
    ap.add_argument("--after", default=None, help="out.npz from run_hand_reproj (default run/hand_reproj_opt/out.npz)")
    ap.add_argument("--fps", type=float, default=24.0)
    a = ap.parse_args()

    run = a.run_dir.rstrip("/")
    after_npz = a.after or os.path.join(run, "hand_reproj_opt", "out.npz")
    frames = sorted(glob.glob(os.path.join(run, "stage0_preprocess", "frames", "*.jpg")))
    K = Bundle.load(os.path.join(run, "stage0_preprocess"))["intrinsics"]

    b7 = Bundle.load(os.path.join(run, "stage7_contact_optim"))
    hv_before = b7["hand_verts"]
    hfc = b7["hand_faces"].astype(np.int32)
    hv_after = np.load(after_npz)["hand_verts"]
    frgb = np.tile(SKIN, (len(hfc), 1))
    T = min(len(frames), len(hv_before), len(hv_after))

    H, W = cv2.imread(frames[0]).shape[:2]
    sw, sh = W // 2, H // 2
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (3 * sw, sh))
    grid, gframes = [], {38, 75, 110, 150}
    for t in range(T):
        fr = cv2.imread(frames[t])
        before = render_multi(fr, [(hv_before[t], hfc, frgb)], K, alpha=0.6)
        after = render_multi(fr, [(hv_after[t], hfc, frgb)], K, alpha=0.6)
        row = np.hstack([
            _label(cv2.resize(fr, (sw, sh)), "original"),
            _label(cv2.resize(before, (sw, sh)), f"BEFORE joint_grasp  f{t}"),
            _label(cv2.resize(after, (sw, sh)), f"AFTER kp2d-aligned  f{t}")])
        vw.write(row)
        if t in gframes:
            grid.append(row)
    vw.release()
    if grid:
        cv2.imwrite(a.out.replace(".mp4", "_grid.png"), np.vstack(grid))
    print(f"wrote {a.out} ({T} frames)")


if __name__ == "__main__":
    main()
