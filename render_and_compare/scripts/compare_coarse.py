"""Side-by-side coarse-HOI comparison: CHOIR's coarse vs this repo's coarse.

Renders the hand+object reprojection of a chosen stage from two runs onto the same
input frames and stacks them horizontally (left = run A, right = run B), into a
video + a montage. Use it to eyeball CHOIR's coarse init against ours.

  python scripts/compare_coarse.py --a runs/grab --b runs/grab_choir \
      --stage stage5_coarse_fit --labels "ours" "CHOIR" --out runs/compare_coarse
"""
from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from hoi_recon.bundle import Bundle                       # noqa: E402
from hoi_recon.viz.reproject import render_multi, _label, SKIN  # noqa: E402


def load_stage(run, stage):
    b = Bundle.load(os.path.join(run, stage))
    ov, ofc = b["obj_verts"], b["obj_faces"].astype(np.int32)
    ocol = (b["obj_colors"].astype(np.float32) if b.get("obj_colors") is not None
            else np.full((len(ov), 3), 180, np.float32))
    hfc = b.get("hand_faces")
    return {"ov": ov, "ofc": ofc, "ofrgb": ocol[ofc].mean(1), "op": b["obj_poses"],
            "hv": b["hand_verts"], "hfc": None if hfc is None else hfc.astype(np.int32)}


def overlay(frame, d, t, K):
    R, tt = d["op"][t][:3, :3], d["op"][t][:3, 3]
    ow = d["ov"] @ R.T + tt
    parts = [(ow, d["ofc"], d["ofrgb"])]
    if d["hfc"] is not None:
        parts.append((d["hv"][t], d["hfc"], np.tile(SKIN, (len(d["hfc"]), 1))))
    return render_multi(frame, parts, K, alpha=0.65)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="run dir A (left)")
    ap.add_argument("--b", required=True, help="run dir B (right)")
    ap.add_argument("--stage", default="stage5_coarse_fit")
    ap.add_argument("--labels", nargs=2, default=["A", "B"])
    ap.add_argument("--out", default="runs/compare_coarse")
    ap.add_argument("--fps", type=float, default=24.0)
    a = ap.parse_args()

    frames = sorted(glob.glob(os.path.join(a.a, "stage0_preprocess", "frames", "*.jpg")))
    K = Bundle.load(os.path.join(a.a, "stage0_preprocess"))["intrinsics"]
    da, db = load_stage(a.a, a.stage), load_stage(a.b, a.stage)
    T = min(len(frames), len(da["op"]), len(db["op"]))
    H, W = cv2.imread(frames[0]).shape[:2]
    sw, sh = W // 2, H // 2

    os.makedirs(a.out, exist_ok=True)
    vid = os.path.join(a.out, "coarse_compare.mp4")
    writer = cv2.VideoWriter(vid, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (2 * sw, sh))
    grid_frames = {38, 75, 110, 150, 185}
    grid = []
    for t in range(T):
        frame = cv2.imread(frames[t])
        la = _label(cv2.resize(overlay(frame, da, t, K), (sw, sh)), f"{a.labels[0]} coarse  f{t}")
        lb = _label(cv2.resize(overlay(frame, db, t, K), (sw, sh)), f"{a.labels[1]} coarse  f{t}")
        row = np.hstack([la, lb])
        writer.write(row)
        if t in grid_frames:
            grid.append(row)
    writer.release()
    if grid:
        cv2.imwrite(os.path.join(a.out, "coarse_compare_grid.png"), np.vstack(grid))
    print(f"wrote {vid}")
    print(f"wrote {os.path.join(a.out, 'coarse_compare_grid.png')}")


if __name__ == "__main__":
    main()
