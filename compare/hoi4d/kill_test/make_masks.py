"""Auto hand+object masks for a HOI4D clip via LangSAM (for the kill test).

Object instance disambiguation, v4 (two-kettles / moving-arm / panning-camera):
  1. Detect category candidates on EVERY frame (conf >= 0.3); cache to disk.
  2. Link detections into tracks by greedy box-IoU association (gap <= 8).
  3. Score tracks:
       reject   if mostly inside the hand mask (it IS the hand/arm)
       primary  longest sustained HAND-CONTACT run (dilated boxes touch) — the
                manipulated object is the thing the hand holds; a distractor
                instance never touches the hand
       tiebreak camera-relative displacement: track motion minus the median
                motion of all other tracks (median of static instances ≈ camera
                pan, so image-space pan cancels)
  4. Per frame: write winning track's mask; gaps carry the previous mask.

Writes <clip>/masks/frame_{idx:06d}_masks/{hand.png, object.png} + report.json.
Run in the hort env (has lang_sam + sam2.1 weights).
"""
import argparse, os, glob, json, pickle
import numpy as np
import cv2
from PIL import Image


def box_iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / max(a1 + a2 - inter, 1)


def detect_all(clip, category, n):
    """Per-frame candidates + hand masks, cached (GPU pass runs once per clip)."""
    cache = os.path.join(clip, "masks", "detections.pkl")
    if os.path.exists(cache):
        try:
            with open(cache, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"cache unreadable ({e}); re-detecting")
    from lang_sam import LangSAM
    model = LangSAM(sam_type="sam2.1_hiera_large")

    def predict(img, prompt, box_thr):
        out = model.predict([img], [prompt], box_threshold=box_thr, text_threshold=0.24)[0]
        m = np.asarray(out.get("masks", []))
        b = np.asarray(out.get("boxes", [])).reshape(-1, 4)
        s = np.asarray(out.get("scores", np.zeros(len(b)))).reshape(-1)
        return (m > 0.5 if m.size else np.zeros((0, img.height, img.width), bool)), b, s

    dets, mask_store, hand_pngs, hand_boxes = [], [], [], []
    for t in range(n):
        img = Image.open(os.path.join(clip, "rgb", f"{t:06d}.jpg")).convert("RGB")
        om, ob, osc = predict(img, category, 0.3)
        row = []
        for m, b, s in zip(om, ob, osc):
            if m.sum() < 1500:
                continue
            row.append({"box": b.tolist(), "conf": float(s), "mi": len(mask_store)})
            mask_store.append(cv2.imencode(".png", m.astype(np.uint8) * 255)[1])
        dets.append(row)
        hm, hb, _ = predict(img, "hand", 0.3)
        hand = hm.any(axis=0) if len(hm) else np.zeros((img.height, img.width), bool)
        hand_pngs.append(cv2.imencode(".png", hand.astype(np.uint8) * 255)[1])
        hand_boxes.append(hb.tolist())
        if t % 25 == 0:
            print(f"  detect {t}/{n} ({len(row)} cands)", flush=True)
    data = {"dets": dets, "mask_store": mask_store,
            "hand_pngs": hand_pngs, "hand_boxes": hand_boxes}
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(data, f)
    return data


def build_tracks(dets, n):
    tracks = []
    for t in range(n):
        for d in dets[t]:
            best, best_iou = None, 0.3
            for tr in tracks:
                if t - tr["last_t"] > 8:
                    continue
                iou = box_iou(tr["last_box"], d["box"])
                if iou > best_iou:
                    best, best_iou = tr, iou
            if best is None:
                tracks.append({"frames": {t: d}, "last_box": d["box"], "last_t": t})
            elif t not in best["frames"]:
                best["frames"][t] = d
                best["last_box"], best["last_t"] = d["box"], t
    return tracks


def score_tracks(tracks, data, n, min_len_frac=0.25):
    """-> list of (track, contact_run, rel_disp, hand_frac), best first."""
    hand_boxes = data["hand_boxes"]
    long_tracks = [tr for tr in tracks if len(tr["frames"]) >= min_len_frac * n]

    def centers(tr):
        return {t: ((d["box"][0] + d["box"][2]) / 2, (d["box"][1] + d["box"][3]) / 2)
                for t, d in tr["frames"].items()}
    all_centers = [centers(tr) for tr in long_tracks]

    scored = []
    for i, tr in enumerate(long_tracks):
        ts = sorted(tr["frames"])
        # hand containment (is this track the hand/arm itself?)
        fr = []
        for t in ts[:: max(1, len(ts) // 12)]:
            m = cv2.imdecode(data["mask_store"][tr["frames"][t]["mi"]], 0) > 127
            h = cv2.imdecode(data["hand_pngs"][t], 0) > 127
            fr.append((m & h).sum() / max(m.sum(), 1))
        hand_frac = float(np.median(fr))
        # sustained hand contact (dilated boxes intersect)
        run = best_run = 0
        for t in range(n):
            touch = False
            if t in tr["frames"] and hand_boxes[t]:
                b = tr["frames"][t]["box"]
                touch = any(box_iou([b[0] - 25, b[1] - 25, b[2] + 25, b[3] + 25],
                                    hb) > 0 for hb in hand_boxes[t])
            run = run + 1 if touch else 0
            best_run = max(best_run, run)
        # camera-relative displacement
        c = all_centers[i]
        rel = []
        for t in ts:
            others = [all_centers[j][t] for j in range(len(long_tracks))
                      if j != i and t in all_centers[j]]
            if len(others) >= 1:
                ref0 = np.median([all_centers[j][ts[0]] for j in range(len(long_tracks))
                                  if j != i and ts[0] in all_centers[j]] or [others[0]], axis=0)
                cam = np.median(others, axis=0) - ref0
            else:
                cam = np.zeros(2)
            own = np.array(c[t]) - np.array(c[ts[0]])
            rel.append(np.linalg.norm(own - cam))
        rel_disp = float(np.percentile(rel, 95)) if rel else 0.0
        scored.append((tr, best_run, rel_disp, hand_frac))

    ok = [s for s in scored if s[3] < 0.5]          # not the hand itself
    ok.sort(key=lambda s: (s[1], s[2]), reverse=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--frames", type=int, default=0)
    args = ap.parse_args()

    n = args.frames or len(glob.glob(os.path.join(args.clip, "rgb", "*.jpg")))
    print(f"{args.clip}: {n} frames, prompt '{args.category}'")
    data = detect_all(args.clip, args.category, n)
    tracks = build_tracks(data["dets"], n)
    scored = score_tracks(tracks, data, n)
    for tr, run, disp, hf in scored[:4]:
        print(f"  track len={len(tr['frames'])} contact_run={run} rel_disp={disp:.0f}px hand_frac={hf:.2f}")
    tr = scored[0][0] if scored else None

    prev, carried = None, 0
    for t in range(n):
        outdir = os.path.join(args.clip, "masks", f"frame_{t:06d}_masks")
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "hand.png"), "wb") as f:
            f.write(data["hand_pngs"][t].tobytes())
        obj = None
        if tr and t in tr["frames"]:
            obj = data["mask_store"][tr["frames"][t]["mi"]]
            prev = obj
        elif prev is not None:
            obj, carried = prev, carried + 1
        if obj is None:
            H, W = cv2.imdecode(data["hand_pngs"][t], 0).shape
            obj = cv2.imencode(".png", np.zeros((H, W), np.uint8))[1]
            carried += 1
        with open(os.path.join(outdir, "object.png"), "wb") as f:
            f.write(obj.tobytes())
    rep = {"clip": os.path.basename(args.clip.rstrip('/')), "n_tracks": len(tracks),
           "carried": carried,
           "picked": {"len": len(tr["frames"]) if tr else 0,
                      "contact_run": scored[0][1] if scored else 0,
                      "rel_disp_px": scored[0][2] if scored else 0}}
    with open(os.path.join(args.clip, "masks", "report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print(f"done. {json.dumps(rep['picked'])} carried={carried}")


if __name__ == "__main__":
    main()
