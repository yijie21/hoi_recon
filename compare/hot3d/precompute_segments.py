"""Precompute training-ready per-segment tensors from a clean-clip manifest, for a
fast decode-free PyTorch dataloader (flow-matching HOI generation, pixels + poses).

Per manifest window -> seg_<window_id>.npz:
  rgb        uint8 [T, R, R, 3]   # fisheye -> upright PINHOLE (make_rc_input convention), resized to R
  K          f32   [3, 3]         # pinhole intrinsics at resolution R
  obj_pose   f32   [T, 4, 4]      # GT target object pose in the pinhole CAMERA frame (projects via K)
  obj_verts  f32   [Nv, 3]        # target canonical mesh verts (conditioning; meters)
  obj_uid    int
  hand_mano  f32   [T, 2, 21]     # GT MANO (thetas[15]+wrist_xform[6]); rows [left,right]; NaN if absent
  hand_wrist f32   [T, 2, 4, 4]   # UmeTrack T_cam_from_wrist (license-free); NaN if absent
  hands_present bool [T, 2]
  T_world_pinhole f32 [T, 4, 4]   # T_wP per frame (head motion; enables world/gravity canonicalization)
  depth_mm   u16   [T, R, R]      # --with_depth: GT z-depth raycast of ALL objects + hands, mm, 0 = no return
  seg_mask   u8    [T, R, R]      # --with_masks: nearest-hit instance class, MASK_* below
  meta: window_id, source_clip, sequence_id, participant_id, frame_start, frame_end, fps, fov_deg

Reuses make_rc_input.py's fisheye->pinhole rectification (R_FP, T_Pw) so rgb/K/obj_pose are consistent;
depth/masks reuse its raycast recipe (posed eval GLBs + UmeTrack hand meshes, open3d).
GT is harvested from annotations (no reconstruction).

Usage: precompute_segments.py <manifest.json> --out_dir DIR [--res 256] [--fov 90] [--nverts 2000]
       [--with_depth] [--with_masks] [--limit N]
"""
import argparse, glob, json, os
import numpy as np, cv2, trimesh
from collections import defaultdict
from hand_tracking_toolkit import camera as htt_camera
from hand_tracking_toolkit import dataset as htt_dataset
from hand_tracking_toolkit.hand_models.umetrack_hand_model import forward_kinematics as umetrack_fk
from scipy.spatial.transform import Rotation

DS = "/workspace/datasets/hot3d"; STREAM = "214-1"
R_FP = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])  # pinhole<-fisheye (upright)
COMPRESS = False  # set by --compress; default uncompressed for true mmap / fastest load
MASK_BG, MASK_TARGET, MASK_OTHER_OBJ, MASK_HAND_L, MASK_HAND_R = 0, 1, 2, 3, 4


def T_from(d):
    R = Rotation.from_quat(np.roll(d["quaternion_wxyz"], -1)).as_matrix()
    T = np.eye(4); T[:3, :3], T[:3, 3] = R, d["translation_xyz"]; return T


def rect_maps(cams0, res, fov):
    f = 0.5 * res / np.tan(np.radians(fov / 2))
    K = np.array([[f, 0, res/2 - 0.5], [0, f, res/2 - 0.5], [0, 0, 1.0]])
    fish = htt_camera.from_json(cams0)
    uu, vv = np.meshgrid(np.arange(res), np.arange(res))
    d_P = np.stack([(uu - K[0,2]) / K[0,0], (vv - K[1,2]) / K[1,1], np.ones_like(uu, float)], -1).reshape(-1, 3)
    uv = np.asarray(fish.eye_to_window(d_P @ R_FP.T))
    dirs = (d_P / np.linalg.norm(d_P, axis=1, keepdims=True)).astype(np.float32)  # pinhole rays
    return K, uv[:, 0].reshape(res, res).astype(np.float32), uv[:, 1].reshape(res, res).astype(np.float32), dirs


def _o3d():
    global o3d
    if "o3d" not in globals():
        import open3d
        globals()["o3d"] = open3d
    return o3d


def _obj_o3d(uid, cache):
    """Canonical eval-GLB as an open3d tensor mesh (lazy per-clip cache); None if unloadable."""
    if uid not in cache:
        try:
            g = trimesh.load(f"{DS}/object_models_eval/obj_{int(uid):06d}.glb")
            m = trimesh.util.concatenate(list(g.geometry.values())) if isinstance(g, trimesh.Scene) else g
            cache[uid] = _o3d().t.geometry.TriangleMesh(
                o3d.core.Tensor(np.asarray(m.vertices, np.float32)),
                o3d.core.Tensor(np.asarray(m.faces, np.uint32)))
        except Exception:
            cache[uid] = None
    return cache[uid]


def obj_canon_verts(uid, nverts):
    g = trimesh.load(f"{DS}/object_models_eval/obj_{int(uid):06d}.glb")
    m = trimesh.util.concatenate(list(g.geometry.values())) if isinstance(g, trimesh.Scene) else g
    P, _ = trimesh.sample.sample_surface(m, nverts, seed=0)
    return np.asarray(P, np.float32)


def process_clip(clip_dir, wins, out_dir, res, fov, nverts, with_depth=False, with_masks=False):
    ofiles = sorted(glob.glob(f"{clip_dir}/*.objects.json"))
    cams0 = json.load(open(ofiles[0][:-len(".objects.json")] + ".cameras.json"))[STREAM]
    K, mx, my, dirs = rect_maps(cams0, res, fov)
    shp = json.load(open(f"{clip_dir}/__hand_shapes.json__"))
    um_shape = htt_dataset.from_umetrack_hand_model_json(shp["umetrack"])
    raycast = with_depth or with_masks
    mesh_cache = {}
    if raycast:
        _o3d()
        rays = np.concatenate([np.zeros_like(dirs), dirs], 1).astype(np.float32)
        dz = dirs[:, 2].reshape(res, res)
    written = []
    for w in wins:
        uid = w["target_uid"]; fa, fb = w["frame_start"], w["frame_end"]; T = fb - fa + 1
        rgb = np.zeros((T, res, res, 3), np.uint8)
        obj_pose = np.full((T, 4, 4), np.nan, np.float32)
        hand_mano = np.full((T, 2, 21), np.nan, np.float32)     # [thetas15 + wrist6]
        hand_wrist = np.full((T, 2, 4, 4), np.nan, np.float32)
        hand_joints = np.full((T, 2, 20, 3), np.nan, np.float32)  # UmeTrack 20 landmarks, camera frame
        present = np.zeros((T, 2), bool)
        T_world_pinhole = np.full((T, 4, 4), np.nan, np.float32)
        depth_mm = np.zeros((T, res, res), np.uint16) if with_depth else None
        seg_mask = np.zeros((T, res, res), np.uint8) if with_masks else None
        SIDE = {"left": 0, "right": 1}
        for ti, fi in enumerate(range(fa, fb + 1)):
            stem = ofiles[fi][:-len(".objects.json")]
            rgb[ti] = cv2.remap(cv2.imread(f"{stem}.image_{STREAM}.jpg"), mx, my, cv2.INTER_LINEAR)
            cams = json.load(open(f"{stem}.cameras.json"))[STREAM]
            T_wF = T_from(cams["T_world_from_camera"]); T_wP = T_wF.copy()
            T_wP[:3, :3] = T_wF[:3, :3] @ R_FP; T_Pw = np.linalg.inv(T_wP)
            T_world_pinhole[ti] = T_wP.astype(np.float32)
            anno = json.load(open(f"{stem}.objects.json"))
            if uid in anno:
                obj_pose[ti] = (T_Pw @ T_from(anno[uid][0]["T_world_from_object"])).astype(np.float32)
            scene = gid2cls = None
            if raycast:
                scene = o3d.t.geometry.RaycastingScene(); gid2cls = {}
                for uid2, d in anno.items():
                    mesh = _obj_o3d(uid2, mesh_cache)
                    if mesh is None:
                        continue
                    mm_ = mesh.clone()
                    mm_.transform(o3d.core.Tensor((T_Pw @ T_from(d[0]["T_world_from_object"])).astype(np.float32)))
                    gid2cls[scene.add_triangles(mm_)] = MASK_TARGET if uid2 == uid else MASK_OTHER_OBJ
            hjs = json.load(open(f"{stem}.hands.json"))
            for side, pc in htt_dataset.decode_hand_pose(hjs).items():
                s = SIDE.get(getattr(side, "name", str(side)).lower())
                if s is None or pc.umetrack is None:
                    continue
                present[ti, s] = True
                mp = hjs.get(getattr(side, "name", str(side)).lower(), {}).get("mano_pose")
                if mp:
                    hand_mano[ti, s] = np.array(mp["thetas"] + mp["wrist_xform"], np.float32)
                Tww = T_from(hjs[getattr(side, "name", str(side)).lower()]["umetrack_pose"]["T_world_from_wrist"])
                hand_wrist[ti, s] = (T_Pw @ Tww).astype(np.float32)
                if raycast:
                    lm, hverts, hfaces = umetrack_fk(pc.umetrack, um_shape, requires_mesh=True)
                    hv = hverts.detach().numpy() @ T_Pw[:3, :3].T + T_Pw[:3, 3]
                    gid = scene.add_triangles(o3d.t.geometry.TriangleMesh(
                        o3d.core.Tensor(hv.astype(np.float32)),
                        o3d.core.Tensor(hfaces.detach().numpy().astype(np.uint32))))
                    gid2cls[gid] = MASK_HAND_L if s == 0 else MASK_HAND_R
                else:
                    lm = umetrack_fk(pc.umetrack, um_shape, requires_mesh=False)[0]
                lm = lm.detach().numpy()                                                      # (20,3) world
                hand_joints[ti, s] = (lm @ T_Pw[:3, :3].T + T_Pw[:3, 3]).astype(np.float32)   # -> camera
            if raycast:
                hit = scene.cast_rays(o3d.core.Tensor(rays))
                if with_depth:
                    z = hit["t_hit"].numpy().reshape(res, res) * dz   # range -> z-depth
                    z[~np.isfinite(z)] = 0
                    depth_mm[ti] = np.clip(z * 1000, 0, 65535).astype(np.uint16)
                if with_masks:
                    gids = hit["geometry_ids"].numpy().reshape(res, res)
                    m = np.zeros((res, res), np.uint8)
                    for gid, cls in gid2cls.items():
                        m[gids == gid] = cls
                    seg_mask[ti] = m
        p = f"{out_dir}/seg_{w['window_id']}.npz"
        arrs = dict(
            rgb=rgb, K=K.astype(np.float32), obj_pose=obj_pose,
            obj_verts=obj_canon_verts(uid, nverts), obj_uid=int(uid),
            hand_mano=hand_mano, hand_wrist=hand_wrist, hand_joints=hand_joints, hands_present=present,
            T_world_pinhole=T_world_pinhole,
            meta=json.dumps({k: w[k] for k in ("window_id", "source_clip", "sequence_id",
                             "participant_id", "frame_start", "frame_end", "object_name")}
                            | {"fps": 30, "fov_deg": fov, "res": res}))
        if with_depth:
            arrs["depth_mm"] = depth_mm
        if with_masks:
            arrs["seg_mask"] = seg_mask
        saver = np.savez_compressed if COMPRESS else np.savez  # uncompressed => true mmap, fastest load
        saver(p + ".tmp.npz", **arrs)          # atomic: never leave a half-written seg on interrupt
        os.replace(p + ".tmp.npz", p)
        written.append(p)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest"); ap.add_argument("--out_dir", default="/workspace/datasets/hot3d/hoi_segments")
    ap.add_argument("--res", type=int, default=256); ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--nverts", type=int, default=2000); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--with_depth", action="store_true"); ap.add_argument("--with_masks", action="store_true")
    ap.add_argument("--compress", action="store_true", help="zlib npz (smaller disk, but NOT mmap-able => slower load)")
    a = ap.parse_args()
    global COMPRESS; COMPRESS = a.compress
    os.makedirs(a.out_dir, exist_ok=True)
    wins = json.load(open(a.manifest))["windows"]
    if a.limit:
        wins = wins[:a.limit]
    by_clip = defaultdict(list)
    for w in wins:
        by_clip[w["source_clip"]].append(w)
    done = 0
    for ci, (clip, ws) in enumerate(by_clip.items()):
        cd = f"{DS}/clips/{clip}"
        if not os.path.isdir(cd):
            print(f"  skip (not extracted) {clip}"); continue
        for p in process_clip(cd, ws, a.out_dir, a.res, a.fov, a.nverts, a.with_depth, a.with_masks):
            done += 1
        print(f"[{ci+1}/{len(by_clip)}] {clip}: {len(ws)} segs (total {done})", flush=True)
    sz = sum(os.path.getsize(f) for f in glob.glob(f"{a.out_dir}/seg_*.npz"))
    print(f"\nwrote {done} segments -> {a.out_dir}  ({sz/1e6:.0f} MB, {sz/max(done,1)/1e6:.2f} MB/seg)")


if __name__ == "__main__":
    main()
