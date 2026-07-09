"""Run Any6D (CVPR'25, FoundationPose-style RGB-D render-and-compare) on a HOT3D
bench clip and write a scorable pseudo_gt.npz.

Clean comparison vs icpjgr: same SAM-3D object mesh (from the icpjgr run's
stage3_object/arrays.npz), same calibrated depth+intrinsics, same per-frame object
masks (stage1_detect_track/masks). Any6D re-estimates its own metric scale + pose
from the anchor RGB-D frame, then does independent per-frame RGB-D registration
(as in the paper's run_ho3d_query.py: est.register per frame, no temporal tracking).

Output: <out_dir>/stage8_eval/pseudo_gt.npz with obj_verts, obj_faces, obj_poses
([T,4,4], object->camera), matching compare/hot3d/gt_pose_eval_hot3d.py.

Usage:
  python run_any6d_hot3d.py <rc_input_dir> <icpjgr_run_dir> <out_dir> [--anchor N] [--max_frames N]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import trimesh

ANY6D = "/workspace/code/hoi_recon/any6d"
sys.path.insert(0, ANY6D)
os.chdir(ANY6D)  # predictors resolve weights relative to their own dir; cwd-safe anyway

import nvdiffrast.torch as dr  # noqa: E402
from estimater import Any6D, ScorePredictor, PoseRefinePredictor  # noqa: E402


def read_rgb_frames(mp4):
    cap = cv2.VideoCapture(mp4)
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rc_input")
    ap.add_argument("run_dir")   # icpjgr run: mesh + masks source
    ap.add_argument("out_dir")   # any6d run
    ap.add_argument("--anchor", type=int, default=None)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--iteration", type=int, default=5)
    args = ap.parse_args()

    rc, run, out = args.rc_input.rstrip("/"), args.run_dir.rstrip("/"), args.out_dir.rstrip("/")
    os.makedirs(f"{out}/stage8_eval", exist_ok=True)

    K = np.load(f"{rc}/intrinsics.npy").astype(np.float64)
    rgb_frames = read_rgb_frames(f"{rc}/rgb.mp4")
    T = len(rgb_frames)
    if args.max_frames:
        T = min(T, args.max_frames)

    # anchor frame: from icpjgr stage3 meta if not given
    anchor = args.anchor
    if anchor is None:
        meta = json.load(open(f"{run}/stage3_object/meta.json"))
        anchor = int(meta.get("anchor_frame", 0))
    print(f"[any6d] T={T} anchor_frame={anchor} K=\n{K}")

    # mesh: exact icpjgr SAM-3D mesh (stage3 arrays.npz verts/faces/colors)
    z3 = np.load(f"{run}/stage3_object/arrays.npz")
    mesh = trimesh.Trimesh(vertices=z3["verts"].astype(np.float64),
                           faces=z3["faces"].astype(np.int64),
                           vertex_colors=z3["colors"] if "colors" in z3.files else None,
                           process=False)
    print(f"[any6d] input mesh verts={len(mesh.vertices)} extent={mesh.extents}")

    def load_depth(i):
        d = cv2.imread(f"{rc}/depth_png/{i:05d}.png", cv2.IMREAD_ANYDEPTH)
        return d.astype(np.float32) / 1000.0  # mm -> m

    def load_mask(i):
        return np.load(f"{run}/stage1_detect_track/masks/{i:05d}.npy").astype(bool)

    glctx = dr.RasterizeCudaContext()
    dbg = f"{out}/stage8_eval/_any6d_debug"
    os.makedirs(dbg, exist_ok=True)
    est = Any6D(mesh=mesh, scorer=ScorePredictor(), refiner=PoseRefinePredictor(),
                glctx=glctx, debug=0, debug_dir=dbg)

    # 1) anchor: joint alignment -> metric-scaled mesh (est.mesh_ori) + anchor pose
    a_rgb, a_depth, a_mask = rgb_frames[anchor], load_depth(anchor), load_mask(anchor)
    print(f"[any6d] anchor register_any6d (mask px={a_mask.sum()}) ...")
    est.register_any6d(K=K, rgb=a_rgb, depth=a_depth, ob_mask=a_mask,
                       iteration=args.iteration, name="anchor")
    scaled = est.mesh_ori  # un-centered, metric-scaled mesh; register() returns pose for THIS
    print(f"[any6d] scaled mesh extent={scaled.extents}")

    # 2) per-frame independent RGB-D registration
    poses = np.zeros((T, 4, 4), dtype=np.float64)
    for i in range(T):
        mask = load_mask(i)
        if mask.sum() < 50:
            poses[i] = poses[i - 1] if i > 0 else np.eye(4)
            print(f"[any6d] frame {i}: mask too small, carry-forward")
            continue
        p = est.register(K=K, rgb=rgb_frames[i], depth=load_depth(i), ob_mask=mask,
                         iteration=args.iteration, name=f"f{i}")
        poses[i] = p
        if i % 10 == 0:
            print(f"[any6d] frame {i}/{T} t={p[:3,3]}")

    np.savez(f"{out}/stage8_eval/pseudo_gt.npz",
             obj_verts=np.asarray(scaled.vertices, dtype=np.float64),
             obj_faces=np.asarray(scaled.faces, dtype=np.int64),
             obj_poses=poses)
    print(f"[any6d] wrote {out}/stage8_eval/pseudo_gt.npz  poses={poses.shape}")


if __name__ == "__main__":
    main()
