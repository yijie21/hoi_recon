"""Run FoundationPose (CVPR'24, NVlabs) per-frame RGB-D pose on a HOT3D bench clip
and write a scorable pseudo_gt.npz — the `fp-p` learned core (open-direction #1).

Why this is a *distinct* arm from `any6dp`, given the same estimater code:
`any6d/estimater.py` exposes ONE class whose plain `register()` (line 218) already
IS FoundationPose's render-and-compare register — so any6dp == "FP register_each with
Any6D's scale". The only real lever left is the two things any6dp does NOT do:

  1. **Uniform metric scale.** Any6D's `register_any6d` rescales the mesh *per-axis*
     (OBB extent ratio, estimater.py:456-458), which distorts shape → worse chamfer on
     revolution objects (T4: bottle Any6D 5.2 vs FP-reg 3.2). We instead feed FP the
     icpjgr **uniform** global-metric mesh (from `{icpjgr_run}/stage8_eval/pseudo_gt.npz`
     obj_verts) — same SAM-3D shape, undistorted. Mesh + scale are both controlled vs
     icpjgr; only the per-frame pose method differs from icpjgr (ICP) and from any6dp
     (per-axis scale).
  2. **Temporal tracking.** `track` mode (register once + track_one fwd/back) keeps
     rotation *continuous* → no per-frame symmetry flips (T4: mug rot 2.7°, vase 3.0°,
     vs any6dp's flip tail). Its risk is translation drift on 2/6 clips.

So we compute three pose sets from one FP pass and cache them:
  - **register_each** : drift-free metric translation, but per-frame-independent rotation
                        (flips like any6dp).
  - **track**         : flip-free continuous rotation, but can drift in translation.
  - **fuse**          : R from `track` + t from `register_each`. Coherent because both
                        come from the SAME estimator/mesh/centered-frame (unlike the
                        failed T5 icpjgr-R/Any6D-t transplant) → aims to win BOTH axes.

Output: <out_dir>/stage8_eval/pseudo_gt.npz (obj_verts, obj_faces, obj_poses[T,4,4],
object->camera) for the selected --mode, matching gt_pose_eval_hot3d.py; plus
<out_dir>/stage8_eval/_fp_all_poses.npz (poses_reg, poses_track, anchor, ok_reg) so
re-scoring a different --mode is instant (no FP rerun).

Runs in the forehoi5090 env (torch 2.8+cu128, FoundationPose CUDA exts built for sm_120).

Usage:
  python run_fp_hot3d.py <rc_input_dir> <icpjgr_run_dir> <out_dir> \
      [--mode register_each|track|fuse] [--anchor N] [--iteration 5] [--track_iter 2]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import trimesh

# any6d (taeyeopl/Any6D clone) at the repo root; derived from this file for clone portability.
# Set up by scripts/setup_third_party.sh (see REPRODUCE.md "Best method"); env forehoi5090.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANY6D = os.environ.get("ANY6D_DIR", os.path.join(_REPO, "any6d"))
if not os.path.isdir(ANY6D):
    raise SystemExit(f"Any6D not found at {ANY6D}. Clone taeyeopl/Any6D there (or set "
                     "$ANY6D_DIR) and build its FoundationPose CUDA exts — see REPRODUCE.md.")
sys.path.insert(0, ANY6D)
os.chdir(ANY6D)  # predictors resolve weights relative to their own dir

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
    ap.add_argument("run_dir", help="icpjgr run: uniform metric mesh + masks + anchor source")
    ap.add_argument("out_dir")
    ap.add_argument("--mode", default="register_each",
                    choices=["register_each", "track", "fuse", "auto"],
                    help="which pose set becomes pseudo_gt.npz (all are cached regardless). "
                         "auto: use track where its translation agrees with the drift-free "
                         "register_each (tracker held), else fall back to register_each.")
    ap.add_argument("--drift_thresh_m", type=float, default=0.05,
                    help="auto mode: median ||track_t - reg_t|| below this => tracker held")
    ap.add_argument("--anchor", type=int, default=None)
    ap.add_argument("--mesh_glb", default=None,
                    help="pose this GLB (e.g. the GT eval model) instead of the icpjgr run's "
                         "stage8 mesh. Poses then come out in the GLB's own canonical frame "
                         "(model-based FP's native setting), directly comparable to that GLB's "
                         "GT poses. Masks/anchor still come from <run_dir>.")
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--iteration", type=int, default=5, help="register refine iters")
    ap.add_argument("--track_iter", type=int, default=2, help="track_one refine iters")
    args = ap.parse_args()

    rc, run, out = (args.rc_input.rstrip("/"), args.run_dir.rstrip("/"),
                    args.out_dir.rstrip("/"))
    os.makedirs(f"{out}/stage8_eval", exist_ok=True)
    cache = f"{out}/stage8_eval/_fp_all_poses.npz"

    K = np.load(f"{rc}/intrinsics.npy").astype(np.float64)
    rgb_frames = read_rgb_frames(f"{rc}/rgb.mp4")
    T = len(rgb_frames)
    if args.max_frames:
        T = min(T, args.max_frames)

    anchor = args.anchor
    if anchor is None:
        try:
            meta = json.load(open(f"{run}/stage3_object/meta.json"))
            anchor = int(meta.get("anchor_frame", 0))
        except (FileNotFoundError, json.JSONDecodeError):
            anchor = T // 2  # no stage3 (e.g. --mesh_glb GT mode): central frame anchors track
    anchor = max(0, min(anchor, T - 1))  # clamp (e.g. when --max_frames caps below anchor)

    # UNIFORM-metric mesh: icpjgr's globally-scaled canonical mesh (obj_verts posed by
    # obj_poses in its eval npz). Same SAM-3D shape as any6dp's raw mesh but scaled by a
    # single global factor (Umeyama), so it is NOT per-axis distorted like register_any6d.
    if args.mesh_glb:
        # GT-mesh mode: pose the eval GLB itself (meters, GT canonical). FoundationPose
        # returns poses wrt the input mesh frame (register() applies get_tf_to_centered_mesh),
        # so obj_poses land in this GLB's canonical -> directly comparable to its GT poses.
        g = trimesh.load(args.mesh_glb)
        gm = (trimesh.util.concatenate(list(g.geometry.values()))
              if isinstance(g, trimesh.Scene) else g)
        verts = np.asarray(gm.vertices, np.float64)
        faces = np.asarray(gm.faces, np.int64)
    else:
        zc = np.load(f"{run}/stage8_eval/pseudo_gt.npz")
        verts = zc["obj_verts"].astype(np.float64)
        faces = zc["obj_faces"].astype(np.int64)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    print(f"[fp] T={T} anchor={anchor} mesh verts={len(verts)} extent={mesh.extents} K=\n{K}")

    def load_depth(i):
        d = cv2.imread(f"{rc}/depth_png/{i:05d}.png", cv2.IMREAD_ANYDEPTH)
        return d.astype(np.float32) / 1000.0  # mm -> m

    def load_mask(i):
        return np.load(f"{run}/stage1_detect_track/masks/{i:05d}.npy").astype(bool)

    if os.path.exists(cache):
        zz = np.load(cache)
        poses_reg, poses_track = zz["poses_reg"], zz["poses_track"]
        ok_reg = zz["ok_reg"]
        print(f"[fp] reusing cached poses {cache}")
    else:
        glctx = dr.RasterizeCudaContext()
        dbg = f"{out}/stage8_eval/_fp_debug"
        os.makedirs(dbg, exist_ok=True)
        est = Any6D(mesh=mesh, scorer=ScorePredictor(), refiner=PoseRefinePredictor(),
                    glctx=glctx, debug=0, debug_dir=dbg)

        # ---- register_each: independent per-frame register (drift-free placement) ----
        poses_reg = np.tile(np.eye(4), (T, 1, 1)).astype(np.float64)
        ok_reg = np.zeros(T, bool)
        for i in range(T):
            m = load_mask(i)
            if m.sum() < 50:
                poses_reg[i] = poses_reg[i - 1] if i > 0 else np.eye(4)
                continue
            poses_reg[i] = est.register(K=K, rgb=rgb_frames[i], depth=load_depth(i),
                                        ob_mask=m, iteration=args.iteration, name=f"reg{i}")
            ok_reg[i] = True
            if i % 20 == 0:
                print(f"[fp] register_each {i}/{T} t={poses_reg[i][:3,3]}")
        print(f"[fp] register_each done ({int(ok_reg.sum())}/{T})")

        # ---- track: register anchor once, track_one forward then backward ----
        poses_track = np.tile(np.eye(4), (T, 1, 1)).astype(np.float64)
        am = load_mask(anchor)
        poses_track[anchor] = est.register(K=K, rgb=rgb_frames[anchor],
                                           depth=load_depth(anchor), ob_mask=am,
                                           iteration=args.iteration, name="trk_anchor")
        pl0 = est.pose_last.detach().clone()  # centered-frame anchor pose
        for i in range(anchor + 1, T):        # forward
            poses_track[i] = est.track_one(rgb=rgb_frames[i], depth=load_depth(i), K=K,
                                           iteration=args.track_iter)
        est.pose_last = pl0                   # reset, track backward
        for i in range(anchor - 1, -1, -1):
            poses_track[i] = est.track_one(rgb=rgb_frames[i], depth=load_depth(i), K=K,
                                           iteration=args.track_iter)
        print(f"[fp] track done (anchor {anchor}, fwd+bwd)")

        np.savez(cache, poses_reg=poses_reg, poses_track=poses_track, ok_reg=ok_reg,
                 anchor=anchor)

    # ---- fuse: continuous rotation from track + drift-free translation from reg ----
    poses_fuse = poses_track.copy()
    poses_fuse[:, :3, 3] = poses_reg[:, :3, 3]

    # ---- auto: register_each ALWAYS supplies the (drift-free) translation; track
    # supplies the (flip-free, temporally continuous) ROTATION only when it did not drift,
    # judged by its translation agreement with register_each. Track drift is blatant
    # (>25cm) vs held (<5cm), so this gate is unambiguous (see fp docstring). Downstream,
    # the temporal jitter-smoother cleans register_each's translation jitter. ----
    drift = float(np.median(np.linalg.norm(
        poses_track[:, :3, 3] - poses_reg[:, :3, 3], axis=1)))
    held = drift < args.drift_thresh_m
    poses_auto = poses_fuse if held else poses_reg  # reg-t always; track-R iff held
    print(f"[fp] auto gate: median||track_t-reg_t||={drift*100:.1f}cm "
          f"thresh={args.drift_thresh_m*100:.0f}cm -> tracker "
          f"{'HELD (fuse: reg-t + track-R)' if held else 'DRIFTED (register_each)'}")

    poses = {"register_each": poses_reg, "track": poses_track,
             "fuse": poses_fuse, "auto": poses_auto}[args.mode]

    np.savez(f"{out}/stage8_eval/pseudo_gt.npz",
             obj_verts=verts, obj_faces=faces, obj_poses=poses.astype(np.float64))
    print(f"[fp] wrote {out}/stage8_eval/pseudo_gt.npz  mode={args.mode} poses={poses.shape}")


if __name__ == "__main__":
    main()
