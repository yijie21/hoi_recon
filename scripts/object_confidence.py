"""Per-frame object-reprojection confidence for a finished run.

For every frame, rasterize the reconstructed object mesh at its stage-7 pose and
compare the silhouette against the SAM2 object mask. Three numbers disentangle
"pose is wrong" from "object is occluded by the hand":

  iou        |render ∩ mask| / |render ∪ mask|   — overall agreement. Drops BOTH
             when the pose is off and when the mask shrinks to the visible sliver
             of an occluded object, so it is the pessimistic confidence.
  mask_cov   |render ∩ mask| / |mask|            — fraction of the VISIBLE object
             explained by the reprojection. Robust to occlusion: if the pose is
             right, the visible sliver still lies inside the render. This is the
             best per-frame POSE-confidence proxy.
  occl       fraction of the rendered object silhouette covered by the rendered
             hand — how much of the drop is attributable to the hand.

Also reported: centroid error (px) between mask and render, and mask area.

Outputs (written into <run>/):
  object_confidence.csv     per-frame metrics
  object_confidence.png     metric curves + occlusion shading, low-conf frames marked
  object_confidence_low.png montage of the K lowest-confidence frames
                            (render overlay + mask/render contours + metrics)
  object_confidence.mp4     per-frame reprojection overlay video with a live curve
                            panel below: the metric curves are traced progressively
                            and a cursor marks the current frame, so confidence dips
                            can be matched to what the hand/object are doing.

Usage:
  python scripts/object_confidence.py --run runs/grab [--stage stage7_contact_optim]
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from hoi_recon.bundle import Bundle  # noqa: E402


def rasterize(verts, faces, K, H, W, scale=0.5):
    """Exact mesh silhouette: project verts, fill every triangle (downscaled)."""
    h, w = int(H * scale), int(W * scale)
    Ks = K.copy(); Ks[:2] *= scale
    z = np.clip(verts[:, 2], 1e-6, None)
    u = Ks[0, 0] * verts[:, 0] / z + Ks[0, 2]
    v = Ks[1, 1] * verts[:, 1] / z + Ks[1, 2]
    uv = np.stack([u, v], 1)
    sil = np.zeros((h, w), np.uint8)
    tri = uv[faces].astype(np.int32)                      # (F,3,2)
    ok = (verts[:, 2][faces] > 1e-3).all(1)
    for t in tri[ok]:
        cv2.fillConvexPoly(sil, t, 1)
    return sil.astype(bool)


def ensure_hand_masks(run_dir, frames_dir, T):
    """SAM2 hand masks — image-evidence ground truth for the HAND reprojection,
    independent of every 3D error in the pipeline. Box-prompted from the stage-1
    YOLO hand box (dominant side, largest-box frame), propagated forward and
    backward. Cached under <run>/hand_masks; returns per-frame paths (None where
    SAM2 produced nothing)."""
    hm_dir = os.path.join(run_dir, "hand_masks")
    paths = [os.path.join(hm_dir, f"{t:05d}.npy") for t in range(T)]
    if os.path.isdir(hm_dir) and any(os.path.exists(p) for p in paths):
        return [p if os.path.exists(p) else None for p in paths]
    import torch
    from hoi_recon.config import load_config
    from hoi_recon.backends.real_perception import _load_sam2, _device
    s1 = Bundle.load(os.path.join(run_dir, "stage1_detect_track"))
    boxes, valid = s1["hand_boxes"], s1["hand_valid"].astype(bool)
    dom = 1 if valid[:, 1].sum() >= valid[:, 0].sum() else 0
    areas = np.zeros(T)
    for t in range(T):
        if valid[t, dom]:
            x0, y0, x1, y1 = boxes[t, dom]
            areas[t] = (x1 - x0) * (y1 - y0)
    pf = int(np.argmax(areas))
    print(f"hand masks: SAM2 box prompt on frame {pf} (dominant side "
          f"{'right' if dom else 'left'}), propagating both directions...")
    cfg = load_config(os.path.join(run_dir, "config.yaml"))
    predictor = _load_sam2(cfg)
    os.makedirs(hm_dir, exist_ok=True)
    out = [None] * T
    with torch.inference_mode(), torch.autocast(_device(), dtype=torch.bfloat16):
        state = predictor.init_state(video_path=frames_dir)
        predictor.add_new_points_or_box(state, frame_idx=pf, obj_id=1,
                                        box=boxes[pf, dom].astype(np.float32))
        for rev in (False, True):
            for fidx, _ids, logits in predictor.propagate_in_video(
                    state, start_frame_idx=pf, reverse=rev):
                mm = (logits[0] > 0).squeeze().cpu().numpy().astype(bool)
                np.save(paths[fidx], mm)
                out[fidx] = paths[fidx]
    return out


def overlay_panel(img, sil, m, txt):
    """One annotated panel: render silhouette tinted orange, SAM2 mask contour
    green, render contour orange, metrics in a header bar. img is full-res BGR;
    sil/m are downscaled binary masks of identical shape."""
    ov_img = cv2.resize(img, (sil.shape[1], sil.shape[0]))
    tint = ov_img.copy()
    tint[sil] = (0.45 * tint[sil] + 0.55 * np.array([60, 140, 255])).astype(np.uint8)
    cs, _ = cv2.findContours(sil.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cm, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(tint, cs, -1, (0, 80, 255), 2)        # render = orange/red
    cv2.drawContours(tint, cm, -1, (0, 255, 0), 2)         # SAM2 mask = green
    cv2.rectangle(tint, (0, 0), (tint.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(tint, txt, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return tint


def curve_panel_bg(T, iou, cov, occ, width, height, dpi=100):
    """Static curve background for the video (matplotlib -> BGR array) plus the
    data->pixel transform, so the per-frame trace/cursor can be drawn with cv2."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    x = np.arange(T)
    ax.fill_between(x, 0, np.nan_to_num(occ), color="tab:red", alpha=0.15)
    ax.plot(x, iou, color="tab:blue", lw=1.0, alpha=0.3)
    ax.plot(x, cov, color="tab:green", lw=1.0, alpha=0.3)
    ax.set_xlim(0, T - 1); ax.set_ylim(0, 1.02)
    ax.set_ylabel("score", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)
    ax.legend(["hand occlusion", "IoU", "mask coverage"], loc="lower left",
              fontsize=7, ncol=3, framealpha=0.6)
    fig.tight_layout(pad=0.6)
    fig.canvas.draw()
    bg = np.asarray(fig.canvas.buffer_rgba())[..., 2::-1].copy()   # RGBA -> BGR
    fig_h = bg.shape[0]

    def to_px(xv, yv):
        px, py = ax.transData.transform(np.column_stack([xv, yv])).T
        return np.column_stack([px, fig_h - py]).astype(np.int32)  # mpl origin is bottom-left

    plt.close(fig)
    return bg, to_px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--stage", default="stage7_contact_optim")
    ap.add_argument("--k-low", type=int, default=6, help="frames in the low-confidence montage")
    ap.add_argument("--scale", type=float, default=0.5, help="rasterization scale")
    ap.add_argument("--fps", type=float, default=24.0, help="confidence video fps")
    ap.add_argument("--no-video", action="store_true", help="skip the mp4 export")
    a = ap.parse_args()

    s0 = Bundle.load(os.path.join(a.run, "stage0_preprocess"))
    s7 = Bundle.load(os.path.join(a.run, a.stage))
    K = s0["intrinsics"]
    frames = sorted(glob.glob(os.path.join(a.run, "stage0_preprocess", "frames", "*.jpg")))
    masks_dir = os.path.join(a.run, "stage1_detect_track", "masks")
    T = len(frames)
    H, W = cv2.imread(frames[0]).shape[:2]
    sc = a.scale

    ov, ofc = s7["obj_verts"], s7["obj_faces"].astype(int)
    poses = s7["obj_poses"]
    hv = s7.get("hand_verts")
    hfc = s7.get("hand_faces")
    hfc = hfc.astype(int) if hfc is not None else None

    hm_paths = [None] * T
    if hv is not None and hfc is not None:
        hm_paths = ensure_hand_masks(a.run, os.path.join(a.run, "stage0_preprocess", "frames"), T)

    NANROW = {"iou": np.nan, "dc_iou": np.nan, "mask_cov": np.nan, "occl": np.nan,
              "centroid_err_px": np.nan, "mask_area_px": 0,
              "hand_iou": np.nan, "hand_prec": np.nan, "hand_cov": np.nan,
              "hand_centroid_err_px": np.nan}
    rows = []
    sils = {}
    for t in range(T):
        mp = os.path.join(masks_dir, f"{t:05d}.npy")
        if not os.path.exists(mp):
            rows.append({"frame": t, **NANROW})
            continue
        m = np.load(mp).astype(np.uint8)
        m = cv2.resize(m, (int(W * sc), int(H * sc)), interpolation=cv2.INTER_NEAREST).astype(bool)
        vw = ov @ poses[t, :3, :3].T + poses[t, :3, 3]
        sil = rasterize(vw, ofc, K, H, W, sc)
        sils[t] = sil
        inter = float(np.logical_and(sil, m).sum())
        union = float(np.logical_or(sil, m).sum())
        iou = inter / max(union, 1)
        mask_cov = inter / max(float(m.sum()), 1)
        occl = np.nan
        hsil = None
        if hv is not None and hfc is not None:
            hsil = rasterize(hv[t], hfc, K, H, W, sc)
            occl = float(np.logical_and(hsil, sil).sum()) / max(float(sil.sum()), 1)
        ce = np.nan
        if m.any() and sil.any():
            cm = np.array(np.nonzero(m)).mean(1)[::-1]    # (x,y)
            cs = np.array(np.nonzero(sil)).mean(1)[::-1]
            ce = float(np.linalg.norm(cm - cs) / sc)       # full-res px
        # don't-care IoU: render pixels on the SAM2 HAND mask are excluded from
        # the union — the fair pose score under occlusion (plain IoU is bounded
        # by the visible fraction even for a perfect pose; mask-centroid error is
        # similarly biased toward the visible sliver). Matches the optimizer loss.
        hm = None
        if hm_paths[t] is not None:
            hm = np.load(hm_paths[t]).astype(np.uint8)
            hm = cv2.resize(hm, (sil.shape[1], sil.shape[0]),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
        dc_iou = np.nan
        if hm is not None:
            dc_union = union - float(np.logical_and(sil, np.logical_and(hm, ~m)).sum())
            dc_iou = inter / max(dc_union, 1)
        # hand reprojection vs SAM2 hand mask. Caveat: the SAM2 mask may include
        # some wrist/forearm (contiguous skin), which MANO does not model, so
        # hand_prec (fraction of the MANO render on actual hand pixels) is the
        # fairest registration score; hand_iou is the pessimistic one.
        h_iou = h_prec = h_cov = h_ce = np.nan
        if hsil is not None and hm is not None:
            if hm.sum() > 200 and hsil.any():
                hi = float(np.logical_and(hsil, hm).sum())
                h_iou = hi / max(float(np.logical_or(hsil, hm).sum()), 1)
                h_prec = hi / max(float(hsil.sum()), 1)
                h_cov = hi / max(float(hm.sum()), 1)
                chm = np.array(np.nonzero(hm)).mean(1)[::-1]
                chs = np.array(np.nonzero(hsil)).mean(1)[::-1]
                h_ce = float(np.linalg.norm(chm - chs) / sc)
        rows.append({"frame": t, "iou": iou, "dc_iou": dc_iou, "mask_cov": mask_cov, "occl": occl,
                     "centroid_err_px": ce, "mask_area_px": int(m.sum() / (sc * sc)),
                     "hand_iou": h_iou, "hand_prec": h_prec, "hand_cov": h_cov,
                     "hand_centroid_err_px": h_ce})

    # ---- CSV ----
    csv_path = os.path.join(a.run, "object_confidence.csv")
    with open(csv_path, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader(); wtr.writerows(rows)

    iou = np.array([r["iou"] for r in rows])
    cov = np.array([r["mask_cov"] for r in rows])
    occ = np.array([r["occl"] for r in rows])
    h_iou = np.array([r["hand_iou"] for r in rows])
    h_prec = np.array([r["hand_prec"] for r in rows])
    h_ce = np.array([r["hand_centroid_err_px"] for r in rows])
    valid = ~np.isnan(iou)
    has_hand = bool(np.isfinite(h_iou).any())

    # ---- curves ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n_ax = 2 if has_hand else 1
    fig, axes = plt.subplots(n_ax, 1, figsize=(11, 4.2 * n_ax), dpi=130, sharex=True)
    ax = axes[0] if has_hand else axes
    x = np.arange(T)
    ax.fill_between(x, 0, np.nan_to_num(occ), color="tab:red", alpha=0.15,
                    label="hand occlusion of object (render frac)")
    ax.plot(x, iou, color="tab:blue", lw=1.2, alpha=0.5, label="silhouette IoU (pessimistic)")
    dciou = np.array([r["dc_iou"] for r in rows])
    if np.isfinite(dciou).any():
        ax.plot(x, dciou, color="tab:cyan", lw=1.6,
                label="don't-care IoU (hand excluded — fair under occlusion)")
    ax.plot(x, cov, color="tab:green", lw=1.6,
            label="mask coverage (pose confidence, occlusion-robust)")
    # low-confidence picks: lowest mask_cov among visible frames
    vis = np.where(valid & (np.array([r["mask_area_px"] for r in rows]) > 1500))[0]
    low = vis[np.argsort(cov[vis])[:a.k_low]]
    ax.scatter(low, cov[low], color="tab:orange", zorder=5, s=36,
               label=f"{a.k_low} lowest-confidence frames")
    for t in low:
        ax.annotate(str(t), (t, cov[t]), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color="tab:orange")
    ax.set_ylabel("score"); ax.set_ylim(0, 1.02)
    ax.set_title("Object reprojection confidence over the clip")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.25)
    if has_hand:
        ax2 = axes[1]
        ax2.plot(x, iou, color="tab:blue", lw=1.0, alpha=0.35, label="object IoU (reference)")
        ax2.plot(x, h_iou, color="tab:purple", lw=1.6, label="hand IoU (vs SAM2 hand mask)")
        ax2.plot(x, h_prec, color="tab:orange", lw=1.6,
                 label="hand precision (MANO render on hand pixels; forearm-robust)")
        ax2.set_xlabel("frame"); ax2.set_ylabel("score"); ax2.set_ylim(0, 1.02)
        ax2.set_title("Hand vs object reprojection registration")
        ax2.legend(loc="lower left", fontsize=9)
        ax2.grid(alpha=0.25)
    else:
        ax.set_xlabel("frame")
    fig.tight_layout()
    plot_path = os.path.join(a.run, "object_confidence.png")
    fig.savefig(plot_path)

    # ---- low-confidence montage ----
    panels = []
    for t in sorted(low):
        img = cv2.imread(frames[t])
        sil = sils[t]
        m = np.load(os.path.join(masks_dir, f"{t:05d}.npy")).astype(np.uint8)
        m = cv2.resize(m, (sil.shape[1], sil.shape[0]), interpolation=cv2.INTER_NEAREST)
        txt = (f"f{t}  cov={cov[t]:.2f}  iou={iou[t]:.2f}"
               + (f"  occl={occ[t]:.2f}" if not np.isnan(occ[t]) else ""))
        panels.append(overlay_panel(img, sil, m, txt))
    cols = 2
    rows_m = [np.hstack(panels[i:i + cols]) for i in range(0, len(panels), cols)
              if len(panels[i:i + cols]) == cols]
    if rows_m:
        montage = np.vstack(rows_m)
        low_path = os.path.join(a.run, "object_confidence_low.png")
        cv2.imwrite(low_path, montage)
    else:
        low_path = None

    # ---- dynamic confidence video: overlay on top, live curve panel below ----
    vid_path = None
    if not a.no_video:
        vid_path = os.path.join(a.run, "object_confidence.mp4")
        sh, sw = next(iter(sils.values())).shape            # overlay panel size
        ch = max(180, int(0.42 * sh))                       # curve panel height
        bg, to_px = curve_panel_bg(T, iou, cov, occ, sw, ch)
        ch = bg.shape[0]                                    # actual rendered height
        x = np.arange(T)
        pts_iou = to_px(x, np.nan_to_num(iou))
        pts_cov = to_px(x, np.nan_to_num(cov))
        y0, y1 = to_px([0], [1.0])[0][1], to_px([0], [0.0])[0][1]
        writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 a.fps, (sw, sh + ch))
        for t in range(T):
            img = cv2.imread(frames[t])
            if t in sils:
                m = np.load(os.path.join(masks_dir, f"{t:05d}.npy")).astype(np.uint8)
                m = cv2.resize(m, (sw, sh), interpolation=cv2.INTER_NEAREST)
                txt = (f"f{t:03d}  cov={cov[t]:.2f}  iou={iou[t]:.2f}"
                       + (f"  occl={occ[t]:.2f}" if not np.isnan(occ[t]) else ""))
                top = overlay_panel(img, sils[t], m, txt)
            else:
                top = cv2.resize(img, (sw, sh))
                cv2.rectangle(top, (0, 0), (sw, 26), (0, 0, 0), -1)
                cv2.putText(top, f"f{t:03d}  (no object mask)", (8, 19),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            # curve panel: faint full curves in bg; trace boldly up to t + cursor
            panel = bg.copy()
            if t > 0:
                cv2.polylines(panel, [pts_iou[:t + 1]], False, (180, 119, 31), 2, cv2.LINE_AA)
                cv2.polylines(panel, [pts_cov[:t + 1]], False, (44, 160, 44), 2, cv2.LINE_AA)
            cx = int(pts_cov[t][0])
            cv2.line(panel, (cx, y0), (cx, y1), (60, 60, 230), 1, cv2.LINE_AA)
            for pts, col in ((pts_iou, (180, 119, 31)), (pts_cov, (44, 160, 44))):
                cv2.circle(panel, tuple(pts[t]), 4, col, -1, cv2.LINE_AA)
            writer.write(np.vstack([top, panel]))
        writer.release()

    print(f"frames: {T}  (with mask: {int(valid.sum())})")
    dciou = np.array([r["dc_iou"] for r in rows])
    print(f"object IoU       median={np.nanmedian(iou):.3f}  p10={np.nanpercentile(iou,10):.3f}")
    if np.isfinite(dciou).any():
        print(f"object dc-IoU    median={np.nanmedian(dciou):.3f}  p10={np.nanpercentile(dciou,10):.3f}  (hand-excluded; fair under occlusion)")
    print(f"object mask_cov  median={np.nanmedian(cov):.3f}  p10={np.nanpercentile(cov,10):.3f}")
    print(f"occlusion        median={np.nanmedian(occ):.3f}  max={np.nanmax(occ):.3f}")
    if has_hand:
        oce = np.array([r["centroid_err_px"] for r in rows])
        print(f"hand IoU         median={np.nanmedian(h_iou):.3f}  p10={np.nanpercentile(h_iou,10):.3f}")
        print(f"hand precision   median={np.nanmedian(h_prec):.3f}  p10={np.nanpercentile(h_prec,10):.3f}")
        print(f"centroid err px  object median={np.nanmedian(oce):.1f}  hand median={np.nanmedian(h_ce):.1f}")
    print(f"lowest-confidence frames (by mask_cov): {sorted(low.tolist())}")
    print("wrote:", csv_path)
    print("wrote:", plot_path)
    if low_path:
        print("wrote:", low_path)
    if vid_path:
        print("wrote:", vid_path)


if __name__ == "__main__":
    main()
