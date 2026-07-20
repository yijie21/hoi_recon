"""Resumable driver — REAL coarse HOI training pairs for the flow-matching refiner.

Runs our real coarse stack (FoundationPose object arm `fpauto` + hand-reprojection
optimizer) on training windows from the clean-clip manifest and writes, per window, a
coarse estimate that is PAIRED with the GT already stored in the segment tensors
(compare/hot3d/precompute_segments.py -> hoi_segments/seg_<window_id>.npz). Same
upright-pinhole camera convention (R_FP, 90 deg FOV) as the segments, so coarse poses
transfer 1:1 to the segment GT regardless of resolution.

Per window -> <out_dir>/coarse_<window_id>.npz:
  obj_pose    f32 [T,4,4]   coarse object->pinhole-camera pose, GT-mesh canonical
  hand_joints f32 [T,21,3]  coarse MANO joints, camera frame (NaN where hand absent)
  hand_verts  f16 [T,778,3] coarse MANO mesh verts (NaN where absent)
  visible     bool[T]       object visible (segmented) this frame
  hand_mano_global f32 [T,3]  coarse MANO global-orient axis-angle (camera frame; NaN if absent)
  hand_mano_pose   f32 [T,45] coarse MANO FULL 15-joint articulation axis-angle (NOT PCA-15)
  hand_mano_transl f32 [T,3]  coarse MANO root translation (m, camera frame; NaN if absent)
  hand_mano_side   f32 [T]    1=right/0=left (left => MANO verts mirrored x->-x); NaN if absent
  hand_mano_betas  f32 [10]   shared MANO shape vector
                            (from joint_opt.py's additive export via run_hand_reproj out.npz;
                             RIGHT-hand MANOLayer, axis-angle. FK: see joint_opt.py header)
  meta        json str      window_id, source_clip, frame_start/end, target_uid,
                            per-stage wall-times, versions/modes used

BACKFILL: coarse_*.npz written before the hand_mano_* export was added lack those keys.
To backfill, DELETE the affected coarse_*.npz and re-run the batch (resume regenerates them
with the new keys). Do NOT delete while a batch is running (the in-memory driver won't see it).

Per-window flow (all pinned to GPU 0):
  1. extract clip tar (cached per source_clip)                          [tar]
  2. make_rc_input.py --start --end --out_res  -> windowed rc_input
  3. hoi_recon.cli --stages 0-2 (env rc5090)   -> frames + object masks + HaMeR MANO/kp2d
  4. run_fp_hot3d.py --mode auto --mesh_glb GT (env forehoi5090) -> FP obj poses (GT canonical)
  5. synthesize <run>/stage6_rectify/arrays.npz -> object track the hand optimizer expects
  6. run_hand_reproj.py (self-spawns env sam3d5090) -> hand fit to observed hand pixels
  7. assemble coarse_<window_id>.npz  (+ optional overlay jpg)

Resumable: windows whose coarse_<window_id>.npz exists are skipped. Per-window failures
are logged and skipped, never fatal. Clip extraction is cached while a clip's windows are
processed, then deleted; per-window temp is deleted on success (unless --keep_temp).

Usage:
  python gen_real_pairs.py --participants P0002,P0015 --limit N [--out_res 512]
  python gen_real_pairs.py --window_ids clip-003284_000017_000 --keep_temp --overlay  # validate one
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np

DS = "/workspace/datasets/hot3d"
HERE = os.path.dirname(os.path.abspath(__file__))
RC = os.path.abspath(os.path.join(HERE, "..", "..", "render_and_compare"))
CONFIG = os.path.join(RC, "configs", "real_forehoi_icp_joint_grasp.yaml")  # ABSOLUTE (trap)
PY_RC = "/workspace/miniconda3/envs/rc5090/bin/python"
PY_FP = "/workspace/miniconda3/envs/forehoi5090/bin/python"
CONDA = "/workspace/miniconda3/bin/conda"
TMP = f"{DS}/_tmp_pairs"
MANIFEST = f"{DS}/hoi_clean_manifest.json"
SEG_DIR = f"{DS}/hoi_segments"
STACK_VERSION = "fpauto(auto)+hand_reproj/image-first"


def run_cmd(cmd, log, cwd=None, env=None):
    """Run a subprocess, appending combined stdout/stderr to `log`. Raise on failure."""
    with open(log, "a") as lf:
        lf.write("\n+ " + " ".join(str(c) for c in cmd) + "\n")
        lf.flush()
        r = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env,
                           stdout=lf, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed (exit {r.returncode}): {' '.join(str(c) for c in cmd[:3])}"
                           f" ... (see {log})")


def gpu_env(**extra):
    return dict(os.environ, CUDA_VISIBLE_DEVICES="0", **extra)


def find_tar(clip):
    for split in ("train_aria", "test_aria"):
        p = f"{DS}/{split}/{clip}.tar"
        if os.path.exists(p):
            return p
    return None


def extract_clip(clip):
    cd = f"{TMP}/clip_{clip}"
    if os.path.isdir(cd) and glob.glob(f"{cd}/*.objects.json"):
        return cd
    tar = find_tar(clip)
    if tar is None:
        raise FileNotFoundError(f"no tar found for {clip} in train_aria/test_aria")
    os.makedirs(cd, exist_ok=True)
    subprocess.run(["tar", "-xf", tar, "-C", cd], check=True)
    return cd


def render_overlay(rc, obj_poses, glb, seg_pose, out_jpg, frame=0):
    """Project GT-mesh verts under the COARSE pose (green) and, if available, the segment
    GT pose (red) onto the rc_input rgb frame. A visual sanity check for the pose canonical."""
    import cv2
    import trimesh
    g = trimesh.load(glb)
    m = (trimesh.util.concatenate(list(g.geometry.values()))
         if isinstance(g, trimesh.Scene) else g)
    V = np.asarray(m.vertices, np.float64)
    if len(V) > 1500:
        V = V[np.random.default_rng(0).choice(len(V), 1500, replace=False)]
    K = np.load(f"{rc}/intrinsics.npy")
    cap = cv2.VideoCapture(f"{rc}/rgb.mp4")
    img = None
    for _ in range(frame + 1):
        ok, img = cap.read()
    cap.release()
    if img is None:
        return
    def draw(P, color):
        if P is None or not np.isfinite(P).all():
            return
        X = V @ P[:3, :3].T + P[:3, 3]
        X = X[X[:, 2] > 1e-3]
        uv = (X @ K.T)
        uv = (uv[:, :2] / uv[:, 2:3]).astype(int)
        for u, v in uv:
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                cv2.circle(img, (u, v), 1, color, -1)
    draw(seg_pose, (0, 0, 255))          # GT (red)
    draw(obj_poses[frame], (0, 255, 0))  # coarse (green)
    cv2.imwrite(out_jpg, img)


def process_window(w, clip_dir, out_dir, out_res, fp_mode, keep_temp, overlay):
    wid = w["window_id"]
    uid = str(w["target_uid"])
    fs, fe = int(w["frame_start"]), int(w["frame_end"])
    glb = f"{DS}/object_models_eval/obj_{int(uid):06d}.glb"
    rc = f"{TMP}/rc_{wid}"
    run = f"{TMP}/run_{wid}"
    fp = f"{TMP}/fp_{wid}"
    hand = f"{TMP}/hand_{wid}"
    os.makedirs(out_dir, exist_ok=True)
    logf = f"{out_dir}/logs/{wid}.log"
    os.makedirs(os.path.dirname(logf), exist_ok=True)
    open(logf, "w").close()
    times = {}

    # --- 2. windowed rc_input (fisheye -> upright pinhole; GT-depth raycast) ---
    t0 = time.time()
    if not os.path.exists(f"{rc}/meta.json"):
        run_cmd([PY_RC, f"{HERE}/make_rc_input.py", clip_dir, uid, rc,
                 "--start", fs, "--end", fe, "--out_res", out_res], logf, env=gpu_env())
    times["rc_input"] = time.time() - t0
    meta = json.load(open(f"{rc}/meta.json"))
    prompt = meta.get("prompt")
    if not prompt:
        raise RuntimeError("target not annotated in window's first frame (no SAM2 prompt)")
    px, py = prompt
    if not (0 <= px < meta["out"] and 0 <= py < meta["out"]):
        raise RuntimeError(f"prompt {prompt} outside frame")

    # --- 3. pipeline stages 0-2: frames + object masks + HaMeR MANO/kp2d (env rc5090) ---
    t0 = time.time()
    if not os.path.exists(f"{run}/stage2_hand/arrays.npz"):
        env = gpu_env(RC_GT_DEPTH_DIR=f"{rc}/depth_png",
                      RC_GT_INTRINSICS=f"{rc}/intrinsics.npy")
        run_cmd([PY_RC, "-m", "hoi_recon.cli", "--video", f"{rc}/rgb.mp4", "--out", run,
                 "--real", "--config", CONFIG, "--depth", "gt", "--object-prompt",
                 f"{px:.1f}", f"{py:.1f}", "--stages", "0-2"], logf, cwd=RC, env=env)
    times["pipeline_0_2"] = time.time() - t0

    # --- 4. FoundationPose object arm (fpauto), GT eval GLB (env forehoi5090) ---
    t0 = time.time()
    if not os.path.exists(f"{fp}/stage8_eval/pseudo_gt.npz"):
        run_cmd([PY_FP, f"{HERE}/run_fp_hot3d.py", rc, run, fp, "--mode", fp_mode,
                 "--mesh_glb", glb], logf, env=gpu_env())
    times["fp_object"] = time.time() - t0
    fpz = np.load(f"{fp}/stage8_eval/pseudo_gt.npz")
    obj_poses = fpz["obj_poses"].astype(np.float32)

    # --- 5. synthesize the stage-6 object track the hand optimizer marshals ---
    s6 = f"{run}/stage6_rectify"
    os.makedirs(s6, exist_ok=True)
    ov = fpz["obj_verts"]
    np.savez(f"{s6}/arrays.npz", obj_verts=ov, obj_faces=fpz["obj_faces"],
             obj_colors=np.full((len(ov), 3), 128, np.uint8), obj_poses=fpz["obj_poses"])

    # --- 6. hand-reprojection optimizer (self-spawns env sam3d5090) ---
    t0 = time.time()
    if not os.path.exists(f"{hand}/out.npz"):
        run_cmd([PY_RC, f"{HERE}/run_hand_reproj.py", run, rc, hand], logf,
                env=gpu_env(CONDA_EXE=CONDA))
    times["hand_reproj"] = time.time() - t0
    ho = np.load(f"{hand}/out.npz")
    hj = ho["hand_joints"].astype(np.float32)
    hv = ho["hand_verts"].astype(np.float16)
    vis = ho["visible"].astype(bool)

    # NaN the hand where it was not actually detected. stage-2 HaMeR fills MANO on every
    # frame (extrapolated where undetected), so its kp2d is not a presence signal; stage-1
    # WiLoR/SAM2 hand_valid is the true per-frame detection mask.
    s1 = np.load(f"{run}/stage1_detect_track/arrays.npz")
    if "hand_valid" in s1.files:
        present = np.asarray(s1["hand_valid"]).any(axis=1)
    else:
        kp = np.load(f"{run}/stage2_hand/arrays.npz")["kp2d"]
        present = np.abs(kp).reshape(len(kp), -1).sum(1) > 0
    T = min(len(obj_poses), len(hj), len(present))
    obj_poses, hj, hv, vis, present = obj_poses[:T], hj[:T], hv[:T], vis[:T], present[:T]
    hj[~present] = np.nan
    hv[~present] = np.nan

    # ADDITIVE: marshal joint_opt's per-frame MANO params (see its header) into the coarse
    # pair, NaN'd on undetected frames like joints/verts. Strictly NEW keys — existing
    # coarse_*.npz keys/shapes are unchanged. Defensive: older out.npz lack these keys.
    mano_extra = {}
    if "mano_global_aa" in ho.files:
        mg = ho["mano_global_aa"].astype(np.float32)[:T]
        mp = ho["mano_pose_aa"].astype(np.float32)[:T]
        mt = ho["mano_transl"].astype(np.float32)[:T]
        ms = ho["mano_side"].astype(np.float32)[:T]
        for arr in (mg, mp, mt):
            arr[~present] = np.nan
        ms[~present] = np.nan
        mano_extra = dict(hand_mano_global=mg, hand_mano_pose=mp, hand_mano_transl=mt,
                          hand_mano_side=ms, hand_mano_betas=ho["mano_betas"].astype(np.float32))

    # --- 7. assemble ---
    metaout = {"window_id": wid, "source_clip": w["source_clip"], "frame_start": fs,
               "frame_end": fe, "target_uid": uid, "object_name": w.get("object_name"),
               "out_res": out_res, "fp_mode": fp_mode, "stack": STACK_VERSION,
               "times_s": {k: round(v, 1) for k, v in times.items()},
               "n_frames": int(T), "n_hand_frames": int(present.sum())}
    outp = f"{out_dir}/coarse_{wid}.npz"
    np.savez(outp, obj_pose=obj_poses, hand_joints=hj, hand_verts=hv, visible=vis,
             meta=json.dumps(metaout), **mano_extra)

    # self-check vs segment GT (if present) + optional overlay
    seg_pose0 = None
    segp = f"{SEG_DIR}/seg_{wid}.npz"
    if os.path.exists(segp):
        seg = np.load(segp, allow_pickle=True)
        gt = seg["obj_pose"][:T]
        good = np.isfinite(gt).all((1, 2)) & np.isfinite(obj_poses).all((1, 2))
        if good.any():
            terr = np.linalg.norm(obj_poses[good, :3, 3] - gt[good, :3, 3], axis=1)
            metaout["obj_t_err_mm_med"] = float(np.median(terr) * 1000)
            print(f"    [{wid}] median obj-translation err = "
                  f"{np.median(terr) * 1000:.1f} mm ({int(good.sum())} frames)", flush=True)
            seg_pose0 = gt[np.where(good)[0][0]]
    if overlay:
        try:
            render_overlay(rc, obj_poses, glb, seg_pose0, f"{out_dir}/coarse_{wid}.jpg")
        except Exception as e:
            print(f"    overlay skipped: {e}", flush=True)

    if not keep_temp:
        for d in (rc, run, fp, hand):
            shutil.rmtree(d, ignore_errors=True)
    return metaout


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--participants", default="P0002,P0015",
                    help="comma list of held-out val participants to process")
    ap.add_argument("--window_ids", default=None,
                    help="comma list of explicit window_ids (overrides --participants; for validation)")
    ap.add_argument("--limit", type=int, default=0, help="cap number of windows (0 = all)")
    ap.add_argument("--out_dir", default=f"{DS}/hoi_coarse_pairs")
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--out_res", type=int, default=512, help="rc_input pinhole size")
    ap.add_argument("--fp_mode", default="auto", choices=["auto", "register_each", "track", "fuse"])
    ap.add_argument("--keep_temp", action="store_true", help="keep per-window temp (debug)")
    ap.add_argument("--overlay", action="store_true", help="write a projection-overlay jpg per window")
    a = ap.parse_args()

    os.makedirs(TMP, exist_ok=True)
    os.makedirs(a.out_dir, exist_ok=True)
    wins = json.load(open(a.manifest))["windows"]
    if a.window_ids:
        want = set(a.window_ids.split(","))
        wins = [w for w in wins if w["window_id"] in want]
    else:
        parts = set(a.participants.split(","))
        wins = [w for w in wins if w.get("participant_id") in parts]
    # resume: drop windows already produced
    todo = [w for w in wins if not os.path.exists(f"{a.out_dir}/coarse_{w['window_id']}.npz")]
    if a.limit:
        todo = todo[:a.limit]
    print(f"windows: {len(wins)} selected | {len(wins) - len(todo)} done | {len(todo)} to do", flush=True)

    by_clip = defaultdict(list)
    for w in todo:
        by_clip[w["source_clip"]].append(w)

    n_ok = n_fail = 0
    for ci, (clip, ws) in enumerate(by_clip.items()):
        try:
            cd = extract_clip(clip)
        except Exception as e:
            print(f"[{ci+1}/{len(by_clip)}] SKIP clip {clip}: {e}", flush=True)
            n_fail += len(ws)
            continue
        for w in ws:
            wid = w["window_id"]
            try:
                t0 = time.time()
                m = process_window(w, cd, a.out_dir, a.out_res, a.fp_mode, a.keep_temp, a.overlay)
                n_ok += 1
                print(f"[{ci+1}/{len(by_clip)}] OK {wid} ({time.time()-t0:.0f}s) "
                      f"times={m['times_s']}", flush=True)
            except Exception as e:
                n_fail += 1
                print(f"[{ci+1}/{len(by_clip)}] FAIL {wid}: {type(e).__name__}: {e}", flush=True)
        # clip's windows done -> free its extraction (keep disk flat)
        if not a.keep_temp:
            shutil.rmtree(cd, ignore_errors=True)

    print(f"\nDONE. ok={n_ok} fail={n_fail} -> {a.out_dir}", flush=True)


if __name__ == "__main__":
    main()
