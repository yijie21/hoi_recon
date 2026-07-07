"""b6 kill test — gauge-dominance / gauge-transfer check (pre-registered).

Bet under test: per-frame monocular depth-gauge wobble (affine scale/shift per
frame) dominates hand/object depth error, and the STATIC BACKGROUND alone is
enough to recover the per-frame gauge (so a background-only GS bundle
adjustment could fix the foreground upstream of any HOI method).

Protocol (per clip):
  1. MoGe (Ruicheng/moge-vitl, same model as do-as-i-do) depth per frame.
  2. bg pixels  = valid GT ∧ NOT dilated(hand ∪ object masks)
                  ∧ temporally-static by GT (per-pixel |GT_t − median_t GT| < 5 cm).
  3. fg pixels  = (hand ∪ object masks, eroded) ∧ valid GT.
  4. Fits (Huber IRLS, linear GT ≈ a·mono + b):
       global : ONE (a,b) on bg pooled over all frames   ← baseline gauge
       perfr  : (a_t,b_t) on bg of frame t               ← the b6 mechanism
       oracle : (a_t,b_t) on fg of frame t               ← upper bound (uses GT on fg)
  5. Metrics on fg pixels:
       MAE_g/MAE_p/MAE_o   per-frame mean |corrected − GT|
       R_MAE = 1 − ΣMAE_p / ΣMAE_g          ← KILL CRITERION: GO iff R_MAE ≥ 0.5
       bias traces          mean signed fg error per frame (the jitter driver)
       wiggle = std_t(bias) global vs perfr  ← temporal-stability view
       transfer corr        corr_t(a_t^bg, a_t^fg), corr_t(bias_g,t, oracle-gauge shift)
  --selftest: synthesize mono from GT with a known per-frame gauge + noise and
  verify the machinery recovers it (R_MAE ≥ 0.9 expected).

Usage:
  python kill_test.py --clip /workspace/hoi4d/clips/kettle_N15 \
      --masks '/path/masks/frame_{idx:06d}_masks/*.png' [--half] [--selftest]
Outputs <clip>/kill_test/{result.json, moge_depth.npy (cache), figure.png}
"""
import argparse, glob, json, os, sys
import numpy as np
import cv2


# ------------------------------------------------------------------ data
def load_gt_depth(clip, idx):
    d = cv2.imread(os.path.join(clip, "depth", f"{idx:06d}.png"), cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    return d.astype(np.float32) / 1000.0  # mm -> m, 0 = invalid


def load_dyn_mask(pattern, idx, shape):
    """Union of all mask PNGs matching the per-frame pattern."""
    m = np.zeros(shape, bool)
    for f in glob.glob(pattern.format(idx=idx)):
        im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if im is not None:
            m |= im > 127
    return m


def run_moge(clip, n_frames, half, fov_x_deg):
    """Per-frame MoGe depth, cached as <clip>/kill_test/moge_depth.npy [T,H,W] f16."""
    cache = os.path.join(clip, "kill_test", "moge_depth.npy")
    if os.path.exists(cache):
        d = np.load(cache)
        if len(d) >= n_frames:
            return d[:n_frames].astype(np.float32)
    import torch
    from moge.model.v1 import MoGeModel
    model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").cuda().eval()
    outs = []
    for i in range(n_frames):
        img = cv2.imread(os.path.join(clip, "rgb", f"{i:06d}.jpg"))[:, :, ::-1]
        if half:
            img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
        t = torch.tensor(img.copy(), dtype=torch.float32, device="cuda").permute(2, 0, 1) / 255.0
        with torch.no_grad():
            try:
                out = model.infer(t, fov_x=fov_x_deg)
            except TypeError:
                out = model.infer(t)
        d = out["depth"].float().cpu().numpy()
        d[~np.isfinite(d)] = 0
        outs.append(d.astype(np.float16))
        if i % 25 == 0:
            print(f"  moge {i}/{n_frames}", flush=True)
    d = np.stack(outs)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.save(cache, d)
    return d.astype(np.float32)


# ------------------------------------------------------------------ fitting
def huber_affine(x, y, iters=4, delta=0.05, min_pts=200):
    """Robust linear fit y ≈ a x + b (Huber IRLS). Returns (a, b) or None."""
    if x.size < min_pts:
        return None
    w = np.ones_like(x)
    a, b = 1.0, 0.0
    for _ in range(iters):
        W = w.sum()
        mx, my = (w * x).sum() / W, (w * y).sum() / W
        vx = (w * (x - mx) ** 2).sum() / W
        if vx < 1e-12:
            return None
        a = (w * (x - mx) * (y - my)).sum() / W / vx
        b = my - a * mx
        r = np.abs(y - (a * x + b))
        w = np.where(r <= delta, 1.0, delta / np.maximum(r, 1e-9))
    return float(a), float(b)


# ------------------------------------------------------------------ core
def analyze(gt, mono, dyn, half, dilate_px=35, erode_px=5,
            zmin=0.25, zmax=5.0, static_tol=0.05, bg_stride=3):
    """gt/mono: [T,H,W] float32 (0=invalid). dyn: [T,H,W] bool (hand∪object)."""
    T = gt.shape[0]
    if half and gt.shape[1] != mono.shape[1]:
        gt = np.stack([cv2.resize(g, (mono.shape[2], mono.shape[1]),
                                  interpolation=cv2.INTER_NEAREST) for g in gt])
        dyn = np.stack([cv2.resize(m.astype(np.uint8), (mono.shape[2], mono.shape[1]),
                                   interpolation=cv2.INTER_NEAREST) > 0 for m in dyn])
        dilate_px, erode_px = dilate_px // 2, max(2, erode_px // 2)

    valid = (gt > zmin) & (gt < zmax) & (mono > 0) & np.isfinite(mono)
    # temporally-static bg filter (GT available for the test => use it):
    gt_nan = np.where(valid, gt, np.nan)
    med = np.nanmedian(gt_nan, axis=0)
    static = np.abs(gt_nan - med[None]) < static_tol          # [T,H,W]
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1,) * 2)
    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1,) * 2)

    rows, pooled_x, pooled_y = [], [], []
    for t in range(T):
        dyn_dil = cv2.dilate(dyn[t].astype(np.uint8), kd) > 0
        fg = cv2.erode(dyn[t].astype(np.uint8), ke) > 0
        bg_t = valid[t] & ~dyn_dil & static[t]
        fg_t = valid[t] & fg
        xb = mono[t][bg_t][::bg_stride]; yb = gt[t][bg_t][::bg_stride]
        rows.append({"bg_n": int(xb.size), "fg_n": int(fg_t.sum()),
                     "xb": xb, "yb": yb,
                     "xf": mono[t][fg_t], "yf": gt[t][fg_t]})
        pooled_x.append(xb[::4]); pooled_y.append(yb[::4])

    g = huber_affine(np.concatenate(pooled_x), np.concatenate(pooled_y))
    if g is None:
        raise RuntimeError("global bg fit failed")
    ag, bg_ = g

    per = []
    for t, r in enumerate(rows):
        xb, yb, xf, yf = r["xb"], r["yb"], r["xf"], r["yf"]
        pf = huber_affine(xb, yb)
        orc = huber_affine(xf, yf) if r["fg_n"] >= 200 else None
        # near-field bg: bg pixels at the fg's own GT depth range (support surface)
        pn = None
        if r["fg_n"] >= 200:
            lo, hi = np.percentile(yf, 5) - 0.15, np.percentile(yf, 95) + 0.15
            near = (yb > lo) & (yb < hi)
            pn = huber_affine(xb[near], yb[near])
        # disparity-space affine (MiDaS-style error model): 1/GT ~ a*(1/mono)+b
        db, gb = 1.0 / np.maximum(xb, 1e-6), 1.0 / np.maximum(yb, 1e-6)
        pdisp = huber_affine(db, gb, delta=0.05)
        pnd = None
        if r["fg_n"] >= 200 and pn is not None:
            pnd = huber_affine(db[near], gb[near], delta=0.05)
        e = {}
        if r["fg_n"] >= 200:
            df = 1.0 / np.maximum(xf, 1e-6)
            preds = {"g": ag * xf + bg_,
                     "p": pf[0] * xf + pf[1] if pf else None,
                     "pn": pn[0] * xf + pn[1] if pn else None,
                     "pd": 1.0 / np.maximum(pdisp[0] * df + pdisp[1], 1e-6) if pdisp else None,
                     "pnd": 1.0 / np.maximum(pnd[0] * df + pnd[1], 1e-6) if pnd else None,
                     "o": orc[0] * xf + orc[1] if orc else None}
            for tag, pred in preds.items():
                if pred is None:
                    e[tag] = None
                    continue
                res = pred - yf
                e[tag] = {"mae": float(np.mean(np.abs(res))), "bias": float(np.mean(res))}
        per.append({"t": t, "bg_n": r["bg_n"], "fg_n": r["fg_n"],
                    "a_p": pf[0] if pf else None, "b_p": pf[1] if pf else None,
                    "a_pn": pn[0] if pn else None,
                    "a_o": orc[0] if orc else None, "b_o": orc[1] if orc else None,
                    "fg_z_med": float(np.median(yf)) if r["fg_n"] else None,
                    "bg_z_med": float(np.median(yb)) if yb.size else None,
                    "err": e})

    ok = [p for p in per if p["err"].get("g") and p["err"].get("p")]
    mae_g = np.array([p["err"]["g"]["mae"] for p in ok])
    bias_g = np.array([p["err"]["g"]["bias"] for p in ok])
    variants = {}
    for tag in ("p", "pn", "pd", "pnd", "o"):
        sel = [p for p in ok if p["err"].get(tag)]
        if not sel:
            continue
        mae = np.array([p["err"][tag]["mae"] for p in sel])
        bias = np.array([p["err"][tag]["bias"] for p in sel])
        mg = np.array([p["err"]["g"]["mae"] for p in sel])
        variants[tag] = {"fg_MAE_cm": float(mae.mean() * 100),
                         "R_MAE": float(1 - mae.sum() / mg.sum()),
                         "wiggle_cm": float(bias.std() * 100),
                         "frames": len(sel)}
    a_p = np.array([p["a_p"] for p in ok])
    a_o = np.array([p["a_o"] for p in ok if p["a_o"]])
    a_p_m = np.array([p["a_p"] for p in ok if p["a_o"]])
    a_pn_m = np.array([p["a_pn"] for p in ok if p["a_o"] and p["a_pn"]])
    a_o_n = np.array([p["a_o"] for p in ok if p["a_o"] and p["a_pn"]])

    res = {
        "frames_evaluated": len(ok),
        "global_gauge": {"a": ag, "b": bg_},
        "fg_MAE_global_cm": float(mae_g.mean() * 100),
        "wiggle_global_cm": float(bias_g.std() * 100),
        "variants": variants,
        "R_MAE": variants.get("p", {}).get("R_MAE"),
        "gauge_transfer_corr_a": float(np.corrcoef(a_p_m, a_o)[0, 1]) if a_o.size > 3 else None,
        "gauge_transfer_corr_a_near": float(np.corrcoef(a_pn_m, a_o_n)[0, 1]) if a_o_n.size > 3 else None,
        "gauge_range_a": [float(a_p.min()), float(a_p.max())],
        "fg_z_med_m": float(np.median([p["fg_z_med"] for p in ok if p["fg_z_med"]])),
        "bg_z_med_m": float(np.median([p["bg_z_med"] for p in ok if p["bg_z_med"]])),
    }
    return res, per, ok


def selftest(gt, dyn, half):
    """Synthesize mono = (GT - b_t)/a_t + noise; machinery must recover it."""
    T = gt.shape[0]
    rng = np.random.default_rng(0)
    a_true = 1.0 + 0.15 * np.sin(np.arange(T) / 4.0)
    b_true = 0.06 * np.cos(np.arange(T) / 6.0)
    mono = np.zeros_like(gt)
    for t in range(T):
        m = gt[t] > 0
        mono[t][m] = (gt[t][m] - b_true[t]) / a_true[t]
        mono[t][m] *= 1 + rng.normal(0, 0.01, m.sum())          # 1% multiplicative
        mono[t][m] += rng.normal(0, 0.008, m.sum())              # 8 mm additive
    res, per, ok = analyze(gt, mono, dyn, half=False)
    a_rec = np.array([p["a_p"] for p in ok])
    a_tru = np.array([a_true[p["t"]] for p in ok])
    corr = float(np.corrcoef(a_rec, a_tru)[0, 1])
    print(f"[selftest] R_MAE={res['R_MAE']:.3f} (expect ≥0.9), "
          f"corr(a_rec,a_true)={corr:.4f} (expect ≥0.99), "
          f"MAE per-frame {res['variants']['p']['fg_MAE_cm']:.2f} cm (expect ≈1)")
    assert res["R_MAE"] > 0.9 and corr > 0.99, "SELFTEST FAILED"
    print("[selftest] PASS")


def make_figure(per, ok, res, out_png, clip_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ts = [p["t"] for p in ok]
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(ts, [p["a_p"] for p in ok], color="#2a78d6", label="a_t (background fit)")
    axes[0].plot([p["t"] for p in ok if p["a_o"]], [p["a_o"] for p in ok if p["a_o"]],
                 color="#1baf7a", ls="--", label="a_t (foreground oracle)")
    axes[0].axhline(res["global_gauge"]["a"], color="#898781", lw=1, label="a (global)")
    axes[0].set_ylabel("gauge scale a_t"); axes[0].legend(fontsize=8); axes[0].set_title(
        f"{clip_name}: per-frame depth gauge & foreground error  (R_MAE={res['R_MAE']:.2f})")
    axes[1].plot(ts, [p["err"]["g"]["mae"] * 100 for p in ok], color="#898781", label="global gauge")
    axes[1].plot(ts, [p["err"]["p"]["mae"] * 100 for p in ok], color="#2a78d6", label="per-frame bg gauge")
    if any(p["err"].get("pn") for p in ok):
        axes[1].plot([p["t"] for p in ok if p["err"].get("pn")],
                     [p["err"]["pn"]["mae"] * 100 for p in ok if p["err"].get("pn")],
                     color="#eda100", label="per-frame NEAR-bg gauge")
    axes[1].plot([p["t"] for p in ok if p["err"].get("o")],
                 [p["err"]["o"]["mae"] * 100 for p in ok if p["err"].get("o")],
                 color="#1baf7a", ls="--", label="oracle fg gauge")
    axes[1].set_ylabel("fg MAE (cm)"); axes[1].legend(fontsize=8)
    axes[2].plot(ts, [p["err"]["g"]["bias"] * 100 for p in ok], color="#898781", label="global")
    axes[2].plot(ts, [p["err"]["p"]["bias"] * 100 for p in ok], color="#2a78d6", label="per-frame bg")
    axes[2].axhline(0, color="k", lw=.5)
    axes[2].set_ylabel("fg signed bias (cm)"); axes[2].set_xlabel("frame"); axes[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--masks", required=True,
                    help="per-frame glob with {idx}, e.g. '/x/frame_{idx:06d}_masks/*.png'")
    ap.add_argument("--frames", type=int, default=0, help="0 = all rgb frames")
    ap.add_argument("--half", action="store_true", help="run MoGe at half res")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    n = args.frames or len(glob.glob(os.path.join(args.clip, "rgb", "*.jpg")))
    K = np.load(os.path.join(args.clip, "intrin.npy"))
    W_img = cv2.imread(os.path.join(args.clip, "rgb", "000000.jpg")).shape[1]
    fov_x = float(np.degrees(2 * np.arctan(W_img / 2 / K[0, 0])))
    print(f"clip={args.clip} frames={n} fov_x={fov_x:.1f}°")

    gt = np.stack([load_gt_depth(args.clip, i) for i in range(n)])
    dyn = np.stack([load_dyn_mask(args.masks, i, gt.shape[1:]) for i in range(n)])
    print(f"gt valid={np.mean(gt > 0):.2f}  dyn px/frame={dyn.sum(axis=(1,2)).mean():.0f}")

    if args.selftest:
        selftest(gt, dyn, args.half)
        return

    mono = run_moge(args.clip, n, args.half, fov_x)
    res, per, ok = analyze(gt, mono, dyn, half=args.half)

    outdir = os.path.join(args.clip, "kill_test")
    os.makedirs(outdir, exist_ok=True)
    res["clip"] = os.path.basename(args.clip.rstrip("/"))
    res["GO"] = bool(res["R_MAE"] >= 0.5)
    with open(os.path.join(outdir, "result.json"), "w") as f:
        json.dump({"summary": res,
                   "per_frame": [{k: v for k, v in p.items() if k != "err"} | {"err": p["err"]}
                                 for p in per]}, f, indent=1)
    try:
        make_figure(per, ok, res, os.path.join(outdir, "figure.png"), res["clip"])
    except Exception as e:
        print("figure failed:", e)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
