"""Materialize clean-clip windows into playable mp4s (for review / inspection).
Per window writes: <id>_raw.mp4 (upright egocentric frames) and, with --overlay,
<id>_overlay.mp4 (target=green + others=gray + hands splatted). Reuses
render_clean_overlay.render_frame.

Usage: materialize_clips.py <manifest.json> [--n 20 | --all] --out_dir DIR [--overlay] [--fps 15]
"""
import argparse, glob, json, os, random
import cv2, numpy as np
from hand_tracking_toolkit import dataset as htt_dataset
from render_clean_overlay import render_frame, DS, STREAM, B


def raw_frame(clip_dir, fi):
    stem = sorted(glob.glob(f"{clip_dir}/*.objects.json"))[fi][:-len(".objects.json")]
    img = cv2.imread(f"{stem}.image_{STREAM}.jpg")
    H, W = img.shape[:2]
    sm = cv2.resize(img, (W // B, H // B))
    return cv2.rotate(sm, cv2.ROTATE_90_CLOCKWISE)


def write_mp4(frames, path, fps):
    h, w = frames[0].shape[:2]
    raw = path + ".raw.mp4"
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    os.system(f"ffmpeg -y -loglevel error -i {raw} -c:v libx264 -crf 21 -pix_fmt yuv420p {path} && rm {raw}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out_dir", default="overlays/clean_validation/vids")
    ap.add_argument("--overlay", action="store_true")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    wins = json.load(open(a.manifest))["windows"]
    if not a.all:
        # one-per-object diverse sample (same policy as the montage renderer)
        random.seed(a.seed)
        by = {}
        for w in wins:
            by.setdefault(w["object_name"], []).append(w)
        pool = list(by.values()); random.shuffle(pool)
        wins = [random.choice(g) for g in pool][:a.n]

    for i, w in enumerate(wins):
        cd = f"{DS}/clips/{w['source_clip']}"
        fa, fb = w["frame_start"], w["frame_end"]
        idxs = list(range(fa, fb + 1))
        base = f"{a.out_dir}/{w['window_id']}_{w['object_name']}"
        write_mp4([raw_frame(cd, fi) for fi in idxs], f"{base}_raw.mp4", a.fps)
        if a.overlay:
            shp = json.load(open(f"{cd}/__hand_shapes.json__"))
            um = htt_dataset.from_umetrack_hand_model_json(shp["umetrack"])
            write_mp4([render_frame(cd, fi, w["target_uid"], um) for fi in idxs],
                      f"{base}_overlay.mp4", a.fps)
        print(f"[{i+1}/{len(wins)}] {w['window_id']} {w['object_name']} "
              f"f{fa}-{fb} ({w['duration_s']}s) -> {base}_*.mp4", flush=True)
    print(f"\nwrote clips to {a.out_dir}")


if __name__ == "__main__":
    main()
