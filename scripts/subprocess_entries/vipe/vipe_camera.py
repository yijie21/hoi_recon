# VIPE camera trajectory for the hoi_recon CHOIR coarse path.
#
# Runs INSIDE the `vipe` conda env (cu128 + lietorch custom ops). Given a folder of
# frames, runs VIPE inference and converts its per-frame camera output into the
# pipeline's convention: intrinsics K (3x3, original frame res) and extrinsics
# (T,4,4) world->cam (OpenCV). VIPE outputs c2w OpenCV matrices, so extrinsics =
# inv(c2w). Saved as cam.npz for the hoi_recon env to load.
#
#   python vipe_camera.py --frames_dir F --out cam.npz
import os
import sys
import glob
import argparse
import subprocess

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pipeline", default="default")
    a = ap.parse_args()

    frames = sorted(glob.glob(os.path.join(a.frames_dir, "*.jpg")))
    T = len(frames)
    work = os.path.join(os.path.dirname(os.path.abspath(a.out)), "vipe_raw")
    os.makedirs(work, exist_ok=True)

    # VIPE CLI: infer on a directory of frames.
    cmd = ["vipe", "infer", "--image-dir", os.path.abspath(a.frames_dir),
           "--output", work, "--pipeline", a.pipeline]
    print(f"[vipe_camera] {' '.join(cmd)}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"[vipe_camera] vipe infer failed (exit {r.returncode})")

    # Locate the pose + intrinsics npz (one sequence).
    pose_f = sorted(glob.glob(os.path.join(work, "pose", "*.npz")))
    intr_f = sorted(glob.glob(os.path.join(work, "intrinsics", "*.npz")))
    if not pose_f or not intr_f:
        sys.exit("[vipe_camera] VIPE produced no pose/intrinsics output")
    pg = np.load(pose_f[0]); ig = np.load(intr_f[0])
    c2w = pg["data"].astype(np.float64)             # (N,4,4) OpenCV cam->world
    pinds = pg["inds"].astype(int)
    intr = ig["data"].astype(np.float64)            # (N,4+D) [fx,fy,cx,cy,...]

    # intrinsics: constant across the sequence (median over frames)
    fx, fy, cx, cy = np.median(intr[:, :4], axis=0)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])

    # extrinsics world->cam per frame; fill by index, hold through gaps
    extr = np.tile(np.eye(4), (T, 1, 1))
    last = None
    filled = {int(i): np.linalg.inv(c2w[k]) for k, i in enumerate(pinds)}
    for t in range(T):
        if t in filled:
            last = filled[t]
        if last is not None:
            extr[t] = last

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, intrinsics=K, extrinsics=extr)
    print(f"[vipe_camera] wrote {a.out}: K=({fx:.0f},{fy:.0f}), {len(pinds)}/{T} posed frames")


if __name__ == "__main__":
    main()
