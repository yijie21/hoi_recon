"""Standalone HAND-reprojection optimizer (frozen object) — ADR-0001.

Marshals an existing run's stage-2 hand (HaMeR MANO + kp2d) and stage-6 object into
joint_opt.py's npz format and runs it with --freeze_object + IMAGE-FIRST weights (env
sam3d5090), so ONLY the hand is optimized against kp2d 2D-keypoint reprojection +
hand-silhouette + SOFT contact to the fixed object track. Fixes the hand not
backprojecting onto the observed hand pixels (the default joint_grasp has no image term).

Fast validation path (no full pipeline): point it at an existing icpjgr run.

Usage:
  python run_hand_reproj.py <run_dir> <rc_input> [<out_dir>]
      [--occluder_dir D]   # per-frame SAM2 hand masks (%05d.npy) -> enables L_hand_sil
      [--w_kp2d 5 --w_hsil 1 --w_contact 1 --w_pen 30 --iters 250]

Writes <out_dir>/out.npz = {hand_verts[T,778,3], hand_joints[T,21,3], obj_poses, visible}.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

# repo root derived from this file (compare/hot3d/run_hand_reproj.py) so the driver is
# portable to any clone location, not just /workspace/code/hoi_recon.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JOINT_OPT = f"{REPO}/render_and_compare/third_party/sam-3d-objects/joint_opt.py"
MANO_ROOT = f"{REPO}/render_and_compare/checkpoints/mano"
# chumpy-free MANO (a direct file path — smplx loads it without needing chumpy in sam3d5090).
MANO_DIR = f"{MANO_ROOT}/MANO_RIGHT_np.pkl"
CONVERT_MANO = f"{REPO}/render_and_compare/scripts/convert_mano_chumpy_free.py"


def _ensure_np_mano():
    """Reproducibility self-heal: create the chumpy-free MANO_RIGHT_np.pkl from the placed
    MANO_RIGHT.pkl if it is missing (this driver runs in an env with chumpy, e.g. rc5090)."""
    if os.path.exists(MANO_DIR):
        return
    orig = f"{MANO_ROOT}/MANO_RIGHT.pkl"
    if not os.path.exists(orig):
        raise SystemExit(f"MANO not found: place the license-gated MANO_RIGHT.pkl at {orig} "
                         "(https://mano.is.tue.mpg.de), then re-run.")
    print(f"[hand_reproj] converting MANO -> chumpy-free ({CONVERT_MANO})")
    import subprocess as _sp
    _sp.run([__import__("sys").executable, CONVERT_MANO, MANO_ROOT], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="existing run (stage2 hand + stage6 object + frames + masks)")
    ap.add_argument("rc_input", help="frozen rc_input dir (intrinsics.npy)")
    ap.add_argument("out_dir", nargs="?", default=None)
    ap.add_argument("--env", default="sam3d5090")
    ap.add_argument("--occluder_dir", default=None,
                    help="per-frame SAM2 hand masks (%05d.npy); enables the hand-silhouette term")
    # image-first defaults (ADR-0001 Q4): raise kp2d, keep pen firm, soften contact.
    ap.add_argument("--w_kp2d", type=float, default=5.0)
    ap.add_argument("--w_hsil", type=float, default=1.0)
    ap.add_argument("--w_contact", type=float, default=1.0)
    ap.add_argument("--w_pen", type=float, default=30.0)
    ap.add_argument("--iters", type=int, default=250)
    a = ap.parse_args()

    run = os.path.abspath(a.run_dir.rstrip("/"))
    rc = os.path.abspath(a.rc_input.rstrip("/"))
    out_dir = os.path.abspath(a.out_dir or os.path.join(run, "hand_reproj_opt"))
    os.makedirs(out_dir, exist_ok=True)
    _ensure_np_mano()

    s2 = np.load(f"{run}/stage2_hand/arrays.npz")
    s6 = np.load(f"{run}/stage6_rectify/arrays.npz")
    T = len(s2["verts"])

    hnpz = f"{out_dir}/hand.npz"
    np.savez(hnpz, mano_global=s2["mano_global"], mano_pose=s2["mano_pose"],
             mano_betas=s2["mano_betas"], verts=s2["verts"], joints=s2["joints"],
             contact_idx=s2["contact_idx"], hand_faces=s2["hand_faces"],
             hand_side=s2["hand_side"] if "hand_side" in s2.files else np.ones(T, np.int64),
             kp2d=s2["kp2d"])
    onpz = f"{out_dir}/obj.npz"
    np.savez(onpz, verts=s6["obj_verts"], faces=s6["obj_faces"],
             vertex_colors=s6["obj_colors"], poses=s6["obj_poses"])
    Kp = f"{out_dir}/K.npy"
    np.save(Kp, np.load(f"{rc}/intrinsics.npy"))
    out_npz = f"{out_dir}/out.npz"

    conda = os.environ.get("CONDA_EXE", "conda")
    cmd = [conda, "run", "--no-capture-output", "-n", a.env, "python", JOINT_OPT,
           "--hand", hnpz, "--obj", onpz,
           "--frames_dir", f"{run}/stage0_preprocess/frames",
           "--masks_dir", f"{run}/stage1_detect_track/masks",
           "--K", Kp, "--mano_dir", MANO_DIR, "--out", out_npz, "--freeze_object",
           "--w_kp2d", str(a.w_kp2d), "--w_hsil", str(a.w_hsil),
           "--w_contact", str(a.w_contact), "--w_pen", str(a.w_pen), "--iters", str(a.iters)]
    if a.occluder_dir:
        cmd += ["--occluder_dir", os.path.abspath(a.occluder_dir)]
    print("[hand_reproj] freeze_object image-first (kp2d=%.1f hsil=%.1f contact=%.1f pen=%.1f)"
          % (a.w_kp2d, a.w_hsil, a.w_contact, a.w_pen))
    print("+ " + " ".join(cmd)); sys.stdout.flush()
    r = subprocess.run(cmd, cwd=os.path.dirname(JOINT_OPT))
    if r.returncode != 0 or not os.path.exists(out_npz):
        raise SystemExit(f"joint_opt failed (exit {r.returncode})")
    print(f"[hand_reproj] wrote {out_npz}")


if __name__ == "__main__":
    main()
