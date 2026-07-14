"""Generate a manifest of CLEAN single-object hand-object interaction windows from
HOT3D Aria clips (design agreed via /grill-me). One window = a contiguous stretch
in which exactly ONE object is manipulated by the hand(s).

Per-frame primitives (per object O):
  near(O)   = min 3D dist(UmeTrack hand verts -> posed object surface) < PROX
  moving(O) = per-frame translation delta > MOVE_T  OR  rotation delta > MOVE_R
  manipulated(O) = near ∧ moving          (identifies the target + strict-single)

Window formation (per clip, per candidate target O with >=1 manipulated frame):
  - continuity = near(O) run, gaps <= GAP frames bridged (regrips/brief releases)
  - strict single-object: split the run at any frame where a *different* object is
    manipulated (a static 2nd object merely near a hand does NOT cut)
  - keep only if: duration >= MIN_LEN, motion gate (cumulative translation OR
    rotation range over the window above a floor), visibility pre-filter
    (>=1 frame vis214>VIS_HI  and  median vis214>VIS_MED)

Output: a manifest JSON (one record/window) + printed distribution stats.
No frame data is copied; a window is (source_clip, frame_range, target_uid).

Usage:
  gen_clean_clips.py <subset.json|clipdir...> --out manifest.json [--thresholds...]
"""
import argparse, glob, json, os, sys
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from hand_tracking_toolkit import dataset as htt_dataset
from hand_tracking_toolkit.hand_models.umetrack_hand_model import forward_kinematics as umetrack_fk

DS = "/workspace/datasets/hot3d"
STREAM = "214-1"
FPS = 30.0
N_OBJ_PTS = 1500
NAMES = json.load(open(f"{DS}/object_models_eval/models_info.json"))


def T_from(d):
    R = Rotation.from_quat(np.roll(d["quaternion_wxyz"], -1)).as_matrix()
    T = np.eye(4); T[:3, :3], T[:3, 3] = R, d["translation_xyz"]
    return T


def rot_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


_OBJ_CACHE = {}
def obj_points(uid):
    if uid not in _OBJ_CACHE:
        g = trimesh.load(f"{DS}/object_models_eval/obj_{int(uid):06d}.glb")
        m = trimesh.util.concatenate(list(g.geometry.values())) if isinstance(g, trimesh.Scene) else g
        P, _ = trimesh.sample.sample_surface(m, N_OBJ_PTS, seed=0)
        _OBJ_CACHE[uid] = np.asarray(P)
    return _OBJ_CACHE[uid]


def bridge_gaps(mask, gap):
    """Fill False runs of length <= gap that are flanked by True (morph close)."""
    m = mask.copy()
    n = len(m); i = 0
    while i < n:
        if not m[i]:
            j = i
            while j < n and not m[j]:
                j += 1
            if i > 0 and j < n and m[i-1] and m[j] and (j - i) <= gap:
                m[i:j] = True
            i = j
        else:
            i += 1
    return m


def runs(mask):
    """Yield (start, end_inclusive) contiguous True runs."""
    n = len(mask); i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            yield i, j - 1
            i = j
        else:
            i += 1


def process_clip(clip_dir, TH):
    name = os.path.basename(clip_dir)
    ofiles = sorted(glob.glob(f"{clip_dir}/*.objects.json"))
    if not ofiles:
        return [], None
    F = len(ofiles)
    info = json.load(open(ofiles[0].replace(".objects.json", ".info.json")))
    seq = info.get("sequence_id"); part = seq.split("_")[0] if seq else None

    shp = json.load(open(f"{clip_dir}/__hand_shapes.json__"))
    um_shape = htt_dataset.from_umetrack_hand_model_json(shp["umetrack"])

    uids = list(json.load(open(ofiles[0])).keys())
    # per-frame arrays
    Tpose = {u: [None]*F for u in uids}          # T_world_from_object
    vis = {u: np.full(F, np.nan) for u in uids}
    near = {u: np.zeros(F, bool) for u in uids}
    hands_near = {u: [set() for _ in range(F)] for u in uids}

    for fi, of in enumerate(ofiles):
        stem = of[:-len(".objects.json")]
        anno = json.load(open(of))
        # hand verts (world) per side
        hjs = json.load(open(f"{stem}.hands.json"))
        hverts = {}
        for side, pc in htt_dataset.decode_hand_pose(hjs).items():
            if pc.umetrack is None:
                continue
            _, v, _ = umetrack_fk(pc.umetrack, um_shape, requires_mesh=True)
            hverts[side] = v.detach().numpy().astype(np.float64)
        for u in uids:
            e = anno.get(u)
            if not e:
                continue
            e = e[0]
            Tpose[u][fi] = T_from(e["T_world_from_object"])
            vis[u][fi] = e.get("visibilities_modeled", {}).get(STREAM, np.nan)
            if hverts:
                Pw = obj_points(u) @ Tpose[u][fi][:3, :3].T + Tpose[u][fi][:3, 3]
                tree = cKDTree(Pw)
                for side, hv in hverts.items():
                    d, _ = tree.query(hv, k=1)
                    if d.min() < TH["prox"]:
                        near[u][fi] = True
                        hands_near[u][fi].add(getattr(side, "name", str(side)).lower())

    # per-frame motion + moving + manipulated
    mt = {u: np.zeros(F) for u in uids}; mr = {u: np.zeros(F) for u in uids}
    for u in uids:
        for fi in range(1, F):
            A, B = Tpose[u][fi-1], Tpose[u][fi]
            if A is None or B is None:
                continue
            mt[u][fi] = np.linalg.norm(B[:3, 3] - A[:3, 3]) * 1000.0
            mr[u][fi] = rot_deg(A[:3, :3], B[:3, :3])
    moving = {u: (mt[u] > TH["move_t"]) | (mr[u] > TH["move_r"]) for u in uids}
    manip = {u: near[u] & moving[u] for u in uids}
    manip_any = np.zeros(F, bool)
    for u in uids:
        manip_any |= manip[u]

    windows = []
    for u in uids:
        if not manip[u].any():
            continue
        inhand = bridge_gaps(near[u], TH["gap"])
        # frames where a DIFFERENT object is manipulated (these cut the run)
        other_manip = np.zeros(F, bool)
        for v in uids:
            if v != u:
                other_manip |= manip[v]
        for a, b in runs(inhand):
            # split [a,b] at contaminated frames (a different object manipulated)
            span_ok = ~other_manip[a:b+1]
            for sa, sb in runs(span_ok):
                fa, fb = a + sa, a + sb
                L = fb - fa + 1
                if L < TH["min_len"]:
                    continue
                # motion gate over the window (either translation or rotation)
                trans_cum = float(mt[u][fa:fb+1].sum())
                Rs = [Tpose[u][f][:3, :3] for f in range(fa, fb+1) if Tpose[u][f] is not None]
                rot_range = max((rot_deg(Rs[0], R) for R in Rs), default=0.0) if Rs else 0.0
                if not (trans_cum > TH["win_trans"] or rot_range > TH["win_rot"]):
                    continue
                # visibility pre-filter
                vv = vis[u][fa:fb+1]; vv = vv[~np.isnan(vv)]
                if vv.size == 0 or np.nanmax(vv) <= TH["vis_hi"] or np.nanmedian(vv) <= TH["vis_med"]:
                    continue
                hset = set()
                for f in range(fa, fb+1):
                    hset |= hands_near[u][f]
                windows.append({
                    "window_id": f"{name}_{int(u):06d}_{fa:03d}",
                    "source_clip": name, "sequence_id": seq, "participant_id": part,
                    "target_uid": u, "object_name": NAMES.get(u, {}).get("name"),
                    "frame_start": int(fa), "frame_end": int(fb),
                    "num_frames": int(L), "duration_s": round(L / FPS, 2),
                    "num_hands": len(hset), "hands": sorted(hset),
                    "motion": {"trans_cum_mm": round(trans_cum, 1), "rot_range_deg": round(rot_range, 1)},
                    "visibility": {"vis214_median": round(float(np.nanmedian(vv)), 2),
                                   "vis214_max": round(float(np.nanmax(vv)), 2)},
                })
    stats = {"clip": name, "F": F, "n_objects": len(uids),
             "n_manip_objects": int(sum(manip[u].any() for u in uids)),
             "n_windows": len(windows)}
    return windows, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="subset.json (list of clip ids) or clip dirs")
    ap.add_argument("--out", default="clean_clips_subset.json")
    ap.add_argument("--prox", type=float, default=0.02)      # m, hand->object near
    ap.add_argument("--move_t", type=float, default=2.0)     # mm/frame, moving
    ap.add_argument("--move_r", type=float, default=1.0)     # deg/frame, moving
    ap.add_argument("--gap", type=int, default=12)           # frames (~0.4s) bridge
    ap.add_argument("--min_len", type=int, default=30)       # frames (~1s)
    ap.add_argument("--win_trans", type=float, default=30.0) # mm cumulative
    ap.add_argument("--win_rot", type=float, default=15.0)   # deg range
    ap.add_argument("--vis_hi", type=float, default=0.7)     # >=1 frame anchor
    ap.add_argument("--vis_med", type=float, default=0.25)   # median trackable
    a = ap.parse_args()
    TH = dict(prox=a.prox, move_t=a.move_t, move_r=a.move_r, gap=a.gap, min_len=a.min_len,
              win_trans=a.win_trans, win_rot=a.win_rot, vis_hi=a.vis_hi, vis_med=a.vis_med)

    # resolve inputs -> clip dirs
    clip_dirs = []
    for inp in a.inputs:
        if inp.endswith(".json") and os.path.isfile(inp):
            for cid in json.load(open(inp)):
                clip_dirs.append(f"{DS}/clips/{cid}")
        else:
            clip_dirs.append(inp.rstrip("/"))

    all_win, all_stats = [], []
    for i, cd in enumerate(clip_dirs):
        if not os.path.isdir(cd):
            print(f"  skip (missing) {cd}"); continue
        w, s = process_clip(cd, TH)
        if s: all_stats.append(s)
        all_win += w
        print(f"[{i+1}/{len(clip_dirs)}] {os.path.basename(cd)}: "
              f"{s['n_manip_objects']}/{s['n_objects']} objs manipulated -> {len(w)} windows", flush=True)

    json.dump({"thresholds": TH, "windows": all_win}, open(a.out, "w"), indent=1)
    # distribution stats
    import statistics as st
    nclips = len(all_stats); nwin = len(all_win)
    durs = [w["num_frames"] for w in all_win]
    wpc = [s["n_windows"] for s in all_stats]
    print("\n==== DISTRIBUTIONS ====")
    print(f"clips processed: {nclips} | total windows: {nwin} | windows/clip mean {nwin/max(nclips,1):.2f}")
    print(f"clips with >=1 window: {sum(x>0 for x in wpc)} ({100*sum(x>0 for x in wpc)/max(nclips,1):.0f}%)")
    if durs:
        print(f"window duration frames: min {min(durs)} med {int(st.median(durs))} max {max(durs)} "
              f"(={min(durs)/FPS:.1f}-{max(durs)/FPS:.1f}s)")
        print(f"windows/clip: 0->{wpc.count(0)}  1->{wpc.count(1)}  2->{wpc.count(2)}  3+->{sum(x>=3 for x in wpc)}")
        from collections import Counter
        objc = Counter(w["object_name"] for w in all_win)
        print("top objects:", dict(objc.most_common(10)))
        print("bimanual windows:", sum(w["num_hands"] >= 2 for w in all_win))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
