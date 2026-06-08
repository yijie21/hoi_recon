# Dyn-HaMR 4D hand stabilization for the hoi_recon CHOIR coarse path.
#
# Runs INSIDE the `dynhamr` conda env (torch 1.13/cu117). Drives Dyn-HaMR's
# run_opt.py on a folder of frames in is_static mode (camera frame, no SLAM), then
# loads the optimized per-frame MANO parameters and runs MANO forward to produce
# camera-frame hand vertices (T,778,3) and joints (T,21,3) for the pipeline.
#
#   python dynhamr_track.py --frames_dir F --K K.npy --out hand.npz --is_static
#
# NOTE: this assumes Dyn-HaMR's _DATA tree (HaMeR/ViTPose/HMP weights + MANO) is
# populated and its submodules (DROID-SLAM/HaMeR/ViTPose) are built — see
# scripts/setup_choir_envs.sh. Falls back (nonzero exit) if the optimization did
# not produce a results npz, so the caller degrades to HaMeR + the isolated fit.
import os
import sys
import glob
import argparse
import subprocess

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--K", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--is_static", action="store_true")
    ap.add_argument("--seq", default="hoi_clip")
    a = ap.parse_args()

    frames = sorted(glob.glob(os.path.join(a.frames_dir, "*.jpg")))
    T = len(frames)

    # Dyn-HaMR expects its own data root layout; stage frames into it and run the
    # multi-stage optimizer (is_static => camera frame, skip SLAM/world-scale).
    root = os.path.join(_HERE, "_run", a.seq)
    img_dir = os.path.join(root, "images", a.seq)
    os.makedirs(img_dir, exist_ok=True)
    for i, f in enumerate(frames):
        dst = os.path.join(img_dir, f"{i:06d}.jpg")
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(f), dst)

    cmd = [sys.executable, os.path.join(_HERE, "dyn-hamr", "run_opt.py"),
           "data=video", "run_opt=True", f"data.seq={a.seq}",
           f"data.root={root}", f"is_static={'True' if a.is_static else 'False'}"]
    print(f"[dynhamr_track] {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=_HERE)
    if r.returncode != 0:
        sys.exit(f"[dynhamr_track] run_opt failed (exit {r.returncode})")

    # Find the latest results npz (world or prior); is_static => values are camera-frame.
    cand = sorted(glob.glob(os.path.join(_HERE, "outputs", "**", "*_results.npz"),
                            recursive=True), key=os.path.getmtime)
    if not cand:
        sys.exit("[dynhamr_track] no results npz produced")
    d = np.load(cand[-1])
    root_orient = d["root_orient"][0]        # (T,3) axis-angle
    pose_body = d["pose_body"][0]            # (T,45) axis-angle (15 joints)
    betas = d["betas"][0]                    # (10,)
    trans = d["trans"][0]                    # (T,3)

    # MANO forward (right-hand layer; left handled upstream by mirroring if needed)
    import torch
    import smplx
    mano_dir = os.path.join(_HERE, "_DATA", "data", "mano")
    mano = smplx.MANOLayer(model_path=mano_dir, is_rhand=True, use_pca=False)
    from pytorch3d.transforms import axis_angle_to_matrix
    Tt = root_orient.shape[0]
    go = axis_angle_to_matrix(torch.tensor(root_orient, dtype=torch.float32)).view(Tt, 1, 3, 3)
    hp = axis_angle_to_matrix(torch.tensor(pose_body, dtype=torch.float32).view(Tt, 15, 3)).view(Tt, 15, 3, 3)
    out = mano(global_orient=go, hand_pose=hp,
               betas=torch.tensor(betas, dtype=torch.float32)[None].expand(Tt, -1))
    verts = out.vertices.detach().numpy() + trans[:, None]
    joints = out.joints.detach().numpy() + trans[:, None]

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, verts=verts.astype(np.float32), joints=joints.astype(np.float32))
    print(f"[dynhamr_track] wrote {a.out}: verts={verts.shape} joints={joints.shape}")


if __name__ == "__main__":
    main()
