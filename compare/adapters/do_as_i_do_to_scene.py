"""do-as-i-do reconstruction output -> workbench scene.npz.

Inputs (produced by do-as-i-do/reconstruction/run_pipeline_headless.sh on wild6):
  layout_camera_frame_optimized.json  per-frame object 6-DoF pose in OpenCV camera frame
                                       (local_to_scene.{translation,quat_wxyz}_camera_frame)
                                       + translation_scale_optimization.mesh_scale (hand-anchored)
  white_bottle.obj                     canonical object mesh (SAM-3D)
  all_hand_meshes.npz                  HaWoR per-frame hand (left_vertices [T,778,3], camera frame)

Placement replicates the pipeline's own visualize_3d.py (camera_frame branch):
  verts_cam[t] = (mesh_verts * mesh_scale) @ R(quat_wxyz_camera_frame[t]).T + translation_camera_frame[t]

Usage: python compare/adapters/do_as_i_do_to_scene.py <recon/wild6> compare/scenes/do_as_i_do_wild6.npz
"""
import sys, os, json, numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import median_filter, uniform_filter1d


def smooth_translations(T, med=9, avg=7):
    """Temporally smooth a per-frame [N,3] translation track: median (kills spikes) then
    moving-average (removes high-freq wobble), keeping low-freq real motion. Returns the
    smoothed track AND the per-frame delta (smoothed - raw) so the SAME correction can be
    applied to the hand -> the hand-object grasp is preserved exactly (rigid per-frame shift).

    Fixes the apparent-size flicker: the object depth is anchored to the HaWoR hand, whose
    per-frame depth wobbles ~8x more than it truly moves; that wobble (not the depth model)
    is what makes the backprojected object flicker larger/smaller."""
    Ts = np.empty_like(T)
    for c in range(3):
        Ts[:, c] = uniform_filter1d(median_filter(T[:, c], size=med, mode="nearest"),
                                    size=avg, mode="nearest")
    return Ts, (Ts - T)


def read_obj(path):
    V, F = [], []
    for ln in open(path):
        s = ln.split()
        if s[:1] == ["v"]: V.append([float(x) for x in s[1:4]])
        elif s[:1] == ["f"]: F.append([int(q.split("/")[0]) - 1 for q in s[1:4]])
    return np.asarray(V, np.float32), np.asarray(F, np.int32)


def main(recon_dir, out, anchor="left", smooth=True):
    obj = "white_bottle"
    cv = f"{recon_dir}/obj_tracking_out/{obj}/combined_visualization"
    layout = json.load(open(f"{cv}/layout_camera_frame_optimized.json"))
    mesh_scale = float(layout["translation_scale_optimization"]["mesh_scale"])
    mverts, mfaces = read_obj(f"{recon_dir}/video_segmentation/masks/frame_000025_masks/{obj}/{obj}.obj")
    mverts = mverts * mesh_scale
    hand = np.load(f"{recon_dir}/{os.path.basename(recon_dir)}/all_hand_meshes.npz", allow_pickle=True)
    hv_all, hf = hand[f"{anchor}_vertices"], hand[f"{anchor}_faces"].astype(np.int32)

    poses, hverts = [], []
    for o in layout["objects"]:
        p = o["local_to_scene"]
        qw = p["quat_wxyz_camera_frame"]                 # (w,x,y,z)
        Rm = R.from_quat([qw[1], qw[2], qw[3], qw[0]]).as_matrix()
        t = np.asarray(p["translation_camera_frame"], np.float64)
        M = np.eye(4); M[:3, :3] = Rm; M[:3, 3] = t
        poses.append(M)
        fidx = min(int(o["frame_idx"]), len(hv_all) - 1)
        hverts.append(hv_all[fidx])

    poses = np.asarray(poses, np.float32)
    hverts = np.asarray(hverts, np.float32)

    if smooth:
        # de-wobble the shared depth: smooth the object translation, then shift the hand by
        # the SAME per-frame delta so the grasp is preserved exactly (both de-wobble together).
        tr = poses[:, :3, 3].astype(np.float64)
        tr_s, delta = smooth_translations(tr)
        poses[:, :3, 3] = tr_s.astype(np.float32)
        hverts = (hverts + delta[:, None, :]).astype(np.float32)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out,
             hand_verts=hverts, hand_faces=hf,
             obj_verts=mverts, obj_faces=mfaces, obj_poses=poses,
             source=f"do-as-i-do (wild6, {len(poses)}f, SAM3+SAM-3D+DA3-metric+HaWoR+guided-diffusion"
                    f"{', depth-smoothed' if smooth else ''})")
    print(f"wrote {out}: hand {hverts.shape}, obj mesh {mverts.shape} x poses {poses.shape}, "
          f"mesh_scale={mesh_scale:.4f}, smooth={smooth}")


if __name__ == "__main__":
    # usage: do_as_i_do_to_scene.py <recon/wild6> <out.npz> [--raw]
    smooth = "--raw" not in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0], args[1], smooth=smooth)
