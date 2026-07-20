"""Full-scale build of the clean single-object HOI dataset over ALL Aria clips.
One pass per clip: extract its tar -> temp, detect clean windows (gen_clean_clips),
blocklist Aria devices, precompute per-segment tensors (precompute_segments, with
UmeTrack joints), delete temp. Keeps disk flat (one clip extracted at a time),
resumable (skips clips already recorded), writes the manifest incrementally.

Refresh mode (--from_manifest): re-precompute the SEGMENT TENSORS ONLY for the windows an
existing manifest already lists (no window re-detection), e.g. to add depth/masks. Resumable
per segment (skips seg files that already carry the requested keys); shardable for parallel
runs: --shard i/n splits the clip list, run one process per shard.

Usage: build_hoi_dataset.py [--seg_dir ...] [--manifest ...] [--res 256] [--limit N]
       [--with_depth] [--with_masks] [--from_manifest] [--shard i/n]
"""
import argparse, glob, json, os, shutil, subprocess, time
import numpy as np
import gen_clean_clips as G
import precompute_segments as P

DS = "/workspace/datasets/hot3d"
BENCH_SEQS = {"P0002_65085bfc", "P0001_f6cc0cc8", "P0002_016222d1", "P0009_9e03121a", "P0010_6d15aa57"}
ARIA_BLOCK = lambda name: (name or "").startswith("aria")   # aria_small / aria_large / ...
TH = dict(prox=0.02, move_t=2.0, move_r=1.0, gap=12, min_len=30,
          win_trans=30.0, win_rot=15.0, vis_hi=0.7, vis_med=0.25)   # keep borderline (no tightening)


def clip_list():
    cd = json.load(open(f"{DS}/clip_definitions.json"))
    tars = {}
    for split in ("train_aria", "test_aria"):
        for p in glob.glob(f"{DS}/{split}/*.tar"):
            tars[os.path.basename(p)[:-4]] = p            # clip-XXXXXX -> tar path
    out = []
    for k, v in cd.items():
        if v.get("device") != "Aria":
            continue
        cid = f"clip-{int(k):06d}"
        if cid in tars and v.get("sequence_id") not in BENCH_SEQS:
            out.append((cid, tars[cid]))
    return sorted(out)


def _seg_has_keys(path, keys):
    try:
        with np.load(path, allow_pickle=True) as z:
            return all(k in z.files for k in keys)
    except Exception:
        return False


def refresh_from_manifest(a):
    """Re-precompute seg tensors for the windows the existing manifest lists (adds new keys)."""
    wins = json.load(open(a.manifest))["windows"]
    need = (["depth_mm"] if a.with_depth else []) + (["seg_mask"] if a.with_masks else []) + ["T_world_pinhole"]
    by_clip = {}
    for w in wins:
        by_clip.setdefault(w["source_clip"], []).append(w)
    clips = sorted(by_clip)
    if a.shard:
        i, n = map(int, a.shard.split("/"))
        clips = clips[i::n]
    tars = {os.path.basename(p)[:-4]: p for split in ("train_aria", "test_aria")
            for p in glob.glob(f"{DS}/{split}/*.tar")}
    tmp_root = f"{DS}/_tmp_refresh{('_' + a.shard.replace('/', '_')) if a.shard else ''}"
    os.makedirs(tmp_root, exist_ok=True)
    todo = [c for c in clips
            if not all(_seg_has_keys(f"{a.seg_dir}/seg_{w['window_id']}.npz", need) for w in by_clip[c])]
    print(f"refresh: {len(clips)} clips in shard, {len(todo)} to do (keys {need})", flush=True)
    t0 = time.time()
    for i, cid in enumerate(todo):
        tdir = f"{tmp_root}/{cid}"
        try:
            os.makedirs(tdir, exist_ok=True)
            subprocess.run(["tar", "-xf", tars[cid], "-C", tdir], check=True)
            P.process_clip(tdir, by_clip[cid], a.seg_dir, a.res, a.fov, a.nverts,
                           a.with_depth, a.with_masks)
        except Exception as e:
            print(f"  FAIL {cid}: {type(e).__name__}: {e}", flush=True)
        finally:
            shutil.rmtree(tdir, ignore_errors=True)
        if (i + 1) % 10 == 0 or i + 1 == len(todo):
            rate = (i + 1) / max(time.time() - t0, 1e-9) * 3600
            print(f"[{i+1}/{len(todo)}] {cid} | {rate:.0f} clips/h | eta {(len(todo)-i-1)/max(rate,1e-9):.1f} h", flush=True)
    shutil.rmtree(tmp_root, ignore_errors=True)
    print("refresh done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg_dir", default=f"{DS}/hoi_segments")
    ap.add_argument("--manifest", default=f"{DS}/hoi_clean_manifest.json")
    ap.add_argument("--res", type=int, default=256); ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--nverts", type=int, default=2000); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--with_depth", action="store_true"); ap.add_argument("--with_masks", action="store_true")
    ap.add_argument("--from_manifest", action="store_true",
                    help="refresh seg tensors for the existing manifest's windows (no re-detection)")
    ap.add_argument("--shard", default="", help="i/n: process clips[i::n] (refresh mode)")
    a = ap.parse_args()
    os.makedirs(a.seg_dir, exist_ok=True)
    if a.from_manifest:
        return refresh_from_manifest(a)
    tmp_root = f"{DS}/_tmp_build"; os.makedirs(tmp_root, exist_ok=True)
    prog_path = a.manifest + ".progress.json"
    prog = json.load(open(prog_path)) if os.path.exists(prog_path) else {"done_clips": [], "windows": []}
    done = set(prog["done_clips"]); windows = prog["windows"]

    clips = clip_list()
    if a.limit:
        clips = clips[:a.limit]
    todo = [(c, t) for c, t in clips if c not in done]
    print(f"Aria clips (minus benchmark seqs): {len(clips)} | already done: {len(done)} | to do: {len(todo)}", flush=True)

    for i, (cid, tar) in enumerate(todo):
        tdir = f"{tmp_root}/{cid}"
        try:
            os.makedirs(tdir, exist_ok=True)
            subprocess.run(["tar", "-xf", tar, "-C", tdir], check=True)
            wins, _ = G.process_clip(tdir, TH)
            wins = [w for w in wins if not ARIA_BLOCK(w.get("object_name"))]
            if wins:
                P.process_clip(tdir, wins, a.seg_dir, a.res, a.fov, a.nverts,
                               a.with_depth, a.with_masks)
                windows += wins
        except Exception as e:
            print(f"  FAIL {cid}: {type(e).__name__}: {e}", flush=True)
        finally:
            shutil.rmtree(tdir, ignore_errors=True)
        done.add(cid)
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            json.dump({"done_clips": sorted(done), "windows": windows}, open(prog_path, "w"))
            segs = len(glob.glob(f"{a.seg_dir}/seg_*.npz"))
            sz = sum(os.path.getsize(f) for f in glob.glob(f"{a.seg_dir}/seg_*.npz"))
            print(f"[{i+1}/{len(todo)}] {cid}: {len(windows)} windows total | {segs} segs | {sz/1e9:.1f} GB", flush=True)

    json.dump({"thresholds": TH, "aria_blocklisted": True, "windows": windows},
              open(a.manifest, "w"), indent=1)
    print(f"\nDONE. {len(windows)} windows -> {a.manifest}")
    print(f"segments -> {a.seg_dir} ({len(glob.glob(f'{a.seg_dir}/seg_*.npz'))} files)")


if __name__ == "__main__":
    main()
