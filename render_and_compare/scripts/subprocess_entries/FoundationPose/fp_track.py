# Per-frame 6D object pose tracking for the hoi_recon pipeline via FoundationPose.
#
# Runs INSIDE the sam3d-objects conda env (reused: it already has torch/pytorch3d/
# nvdiffrast/kaolin/warp). Given the SAM-3D object mesh (metric, object frame),
# the RGB frames, MoGe metric depth, the camera intrinsics, and one object mask to
# register on, it registers on an anchor frame and tracks the object both forward
# and backward, writing per-frame object->camera 4x4 poses to a .npy.
#
#   python fp_track.py --mesh mesh.obj --frames_dir F --depth_dir D \
#       --K K.npy --mask mask.npy --register_frame 38 --out poses.npy
import os
import sys
import glob
import argparse

import numpy as np
import cv2
import trimesh

_CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _CODE)
sys.path.insert(0, os.path.join(_CODE, "mycpp", "build"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--depth_dir", required=True)
    ap.add_argument("--K", required=True)
    ap.add_argument("--mask", required=True, help="object mask (.npy) at register_frame")
    ap.add_argument("--register_frame", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--est_iter", type=int, default=5)
    ap.add_argument("--track_iter", type=int, default=2)
    ap.add_argument("--mode", default="track", choices=["track", "register_each"],
                    help="track: register once + track (drifts on noisy depth); "
                         "register_each: re-register every frame with its mask (drift-free)")
    ap.add_argument("--masks_dir", default=None, help="per-frame masks for register_each")
    ap.add_argument("--depth_scale", type=float, default=1.0)
    ap.add_argument("--clean_depth", type=int, default=0,
                    help="clamp depth within the object mask to a robust band (cm-thick) to "
                         "remove MoGe smear; 0=off, else band in mm")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame (testing)")
    a = ap.parse_args()

    import torch
    from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
    import nvdiffrast.torch as dr
    import logging
    logging.getLogger().setLevel(logging.WARNING)

    mesh = trimesh.load(a.mesh, process=False)
    K = np.load(a.K).astype(np.float64)
    frames = sorted(glob.glob(os.path.join(a.frames_dir, "*.jpg")))
    T = len(frames)

    def rgb(i):
        return cv2.cvtColor(cv2.imread(frames[i]), cv2.COLOR_BGR2RGB)

    def maskfile(i):
        return os.path.join(a.masks_dir, f"{i:05d}.npy") if a.masks_dir else None

    def depth(i):
        d = np.load(os.path.join(a.depth_dir, f"{i:05d}.npy")).astype(np.float32)
        d[~np.isfinite(d)] = 0
        d *= a.depth_scale
        if a.clean_depth and a.masks_dir and os.path.exists(maskfile(i)):
            # MoGe depth smears in z near the object's silhouette; clamp the masked
            # depth to a robust band around its near surface so FoundationPose sees a
            # clean object surface instead of a 12cm-deep blob.
            m = np.load(maskfile(i)).astype(bool)
            zin = d[m & (d > 0)]
            if zin.size > 50:
                znear = np.percentile(zin, 20)
                band = a.clean_depth / 1000.0
                mm = m & (d > 0)
                d[mm] = np.clip(d[mm], znear - band, znear + band)
        return d

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                         mesh=mesh, scorer=scorer, refiner=refiner,
                         glctx=glctx, debug=0, debug_dir="/tmp/fp_debug")

    rf = int(a.register_frame)
    poses = np.tile(np.eye(4), (T, 1, 1)).astype(np.float64)
    ok = np.zeros(T, bool)

    if a.mode == "register_each":
        # drift-free: independently register every frame using its own SAM2 mask
        for i in range(0, T, a.stride):
            mf = maskfile(i)
            if mf is None or not os.path.exists(mf):
                continue
            m = np.load(mf).astype(bool)
            if m.sum() < 50:
                continue
            poses[i] = est.register(K=K, rgb=rgb(i), depth=depth(i), ob_mask=m,
                                    iteration=a.est_iter)
            ok[i] = True
        print(f"[fp_track] register_each: {int(ok.sum())} frames")
    else:
        ob_mask = np.load(a.mask).astype(bool)
        poses[rf] = est.register(K=K, rgb=rgb(rf), depth=depth(rf), ob_mask=ob_mask,
                                 iteration=a.est_iter)
        ok[rf] = True
        pl0 = est.pose_last.detach().clone()
        print(f"[fp_track] registered on frame {rf}")
        for i in range(rf + 1, T):                        # forward
            poses[i] = est.track_one(rgb=rgb(i), depth=depth(i), K=K, iteration=a.track_iter)
            ok[i] = True
        est.pose_last = pl0                               # reset, track backward
        for i in range(rf - 1, -1, -1):
            poses[i] = est.track_one(rgb=rgb(i), depth=depth(i), K=K, iteration=a.track_iter)
            ok[i] = True

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out if a.out.endswith(".npz") else a.out + ".npz",
             poses=poses, ok=ok, register_frame=rf)
    print(f"[fp_track] wrote {a.out} ({int(ok.sum())}/{T} frames tracked)")


if __name__ == "__main__":
    main()
