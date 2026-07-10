"""Stage-4 LEARNED object-pose core: Any6D per-frame RGB-D registration + the
temporal-consistency layer, run as an in-pipeline alternative to the ICP core.

This is item 1 of the roadmap ("integrate the learned core inside the pipeline"):
instead of `object_icp.refine_object_poses`, stage 4 can call this to get the object
trajectory from a learned per-frame RGB-D estimator (Any6D, CVPR'25) that *consumes*
the calibrated depth, then clean it with the temporal layer — all within one pipeline
run, so the grasp stages (5-7, with the object frozen) and eval see the cleaned poses.

Any6D needs its own conda env (`forehoi5090`: torch 2.5.1+cu121, nvdiffrast) that is
incompatible with the pipeline env (`rc5090`), so it runs as a subprocess exactly like
the SAM-3D / VGGT / FoundationPose entries. We reuse the *validated*
`compare/hot3d/run_any6d_hot3d.py` (same script that produced the trusted `combined`
numbers) rather than forking a new entry — it reads the frozen rc_input (rgb.mp4 +
uint16-mm depth_png + intrinsics) and the run's promoted stage-3 mesh + stage-1 masks,
and writes a `pseudo_gt.npz` we read back. The raw Any6D poses are cached; the temporal
layer is re-applied in-process each call so its λ knobs stay tunable without a rerun.

Only valid under identity camera extrinsics (object->camera == object->world); a moving
camera backend would need to compose per-frame extrinsics here.
"""
from __future__ import annotations

import os
import subprocess

import numpy as np

from .logging_utils import log
from .temporal_pose import clean_trajectory

REPO = "/workspace/code/hoi_recon"
RUN_ANY6D = f"{REPO}/compare/hot3d/run_any6d_hot3d.py"


def _lcfg(learned, key, default):
    if learned is None:
        return default
    return learned.get(key, default) if hasattr(learned, "get") else default


def estimate_object_poses_any6d(learned, run_dir, rc_input, s3, out_dir=None):
    """Run Any6D (subprocess, forehoi5090) then the temporal layer, in-process.

    learned : cfg.object_icp.learned mapping (env, iteration, anchor, lam_trans,
              lam_rot). run_dir : absolute path to the current run (has stage1 masks +
              promoted stage3 mesh). rc_input : absolute frozen input dir (rgb.mp4,
              depth_png, intrinsics.npy). s3 : stage3 bundle (for the reused colors).

    Returns dict {obj_verts, obj_faces, obj_poses, obj_colors, obj_radius, stats}.
    obj_verts is metric-scaled (by Any6D) AND recentered to its own centroid, with the
    centering folded into obj_poses so posed geometry is bit-identical (eval-invariant)
    but joint_grasp's `ocen = t0` centroid assumption holds.
    """
    run_dir = os.path.abspath(run_dir)
    rc_input = os.path.abspath(rc_input)
    out_dir = os.path.abspath(out_dir or os.path.join(run_dir, "stage4_any6d"))
    raw = os.path.join(out_dir, "stage8_eval", "pseudo_gt.npz")

    env_name = _lcfg(learned, "env", "forehoi5090")
    iteration = int(_lcfg(learned, "iteration", 5))
    anchor = _lcfg(learned, "anchor", None)
    lam_trans = float(_lcfg(learned, "lam_trans", 3.0))
    lam_rot = float(_lcfg(learned, "lam_rot", 0.0))

    if not os.path.exists(raw):
        conda = os.environ.get("CONDA_EXE", "conda")
        cmd = [conda, "run", "--no-capture-output", "-n", env_name, "python",
               RUN_ANY6D, rc_input, run_dir, out_dir, "--iteration", str(iteration)]
        if anchor is not None:
            cmd += ["--anchor", str(int(anchor))]
        log(f"stage4 LEARNED core: Any6D subprocess (env {env_name}) -> {out_dir}")
        log("  + " + " ".join(cmd))
        r = subprocess.run(cmd, cwd=os.path.dirname(RUN_ANY6D))
        if r.returncode != 0 or not os.path.exists(raw):
            raise RuntimeError(f"Any6D subprocess failed (exit {r.returncode}); no {raw}")
    else:
        log(f"stage4 LEARNED core: reusing cached Any6D poses {raw}")

    z = np.load(raw)
    verts = z["obj_verts"].astype(np.float64)          # metric-scaled (Any6D mesh_ori)
    faces = z["obj_faces"].astype(np.int64)
    poses = z["obj_poses"].astype(np.float64).copy()   # object->camera, [T,4,4]

    # colors: Any6D scales the SAME SAM-3D mesh (identical vertex order), so stage-3
    # colors map 1:1 onto the scaled verts.
    colors = None
    s3_colors = s3.get("colors") if hasattr(s3, "get") else None
    if s3_colors is not None and len(s3_colors) == len(verts):
        colors = s3_colors

    # recenter the (un-centered) Any6D mesh to its centroid; fold the offset into the
    # translations so R@v_c + t' == R@v + t exactly (posed geometry unchanged).
    c = verts.mean(0)
    verts_c = verts - c
    poses[:, :3, 3] += np.einsum("tij,j->ti", poses[:, :3, :3], c)

    # temporal layer: symmetry-flip resolution + translation jitter smoothing (rc5090).
    # NOTE: rotation smoothing (lam_rot) stays 0 — porting rotation priors (grasp-
    # rigidity, depth basin selection, flip smoothing) onto the learned poses was tried
    # and is a net loss (see compare/hot3d/docs/T5_NOTES.md): the learned per-frame
    # rotation is already depth-optimal, so temporal corrections disturb more frames
    # than they fix. any6dp is the placement-optimal arm; icpjgr stays rotation-optimal.
    poses_clean, tinfo = clean_trajectory(poses, verts_c, faces,
                                          lam_rot=lam_rot, lam_trans=lam_trans)
    radius = float(np.linalg.norm(verts_c, axis=1).mean())
    log(f"stage4 LEARNED core: T={len(poses_clean)} n_sym={tinfo['n_sym']} "
        f"radius={radius*100:.1f}cm  frame-jumps raw {tinfo['jumps_raw']} -> "
        f"cleaned {tinfo['jumps_cleaned']}")

    stats = {"pose_core": "learned", "method": "any6d",
             "lam_trans": lam_trans, "lam_rot": lam_rot, **tinfo}
    return {"obj_verts": verts_c, "obj_faces": faces, "obj_poses": poses_clean,
            "obj_colors": colors, "obj_radius": radius, "stats": stats}
