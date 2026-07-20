"""Tier-1 evaluation of the trained bridge flow-matching HOI refiner.

Two tables, both reusing the authored code paths (train.py's val metric machinery and
sample.py's real-coarse state builders) so nothing is re-implemented:

  1. SYNTHETIC MULTI-SCALE VAL — over ALL held-out val segments (participants P0002/P0015,
     windowed exactly as train.py's val: fixed seed+epoch), at FIXED corruption scales
     s in {0.25, 0.5, 1.0}. For each scale we build the val dataset with corrupt_scale=(s,s)
     and report the standard columns for coarse / refined@8 / refined@1. The question:
     does the refiner PRESERVE near-perfect sources (s=0.25 refined must not beat-down coarse)
     while still fixing bad ones (s=1)?

  2. REAL-PAIRS STRATIFIED — over the real FoundationPose coarse pairs
     (coarse_*.npz), stratified by median coarse OBJECT translation error vs the seg GT:
     good (<30 mm) / mid (30-100) / catastrophic (>=100). Per stratum: n, coarse vs
     refined@8 object trans+rot; hand metrics pooled over ONLY the files that carry MANO
     params (report that n). The catastrophic stratum is an upstream re-detection problem,
     not a refinement one (bridge is source-anchored) — reported honestly.

Writes tier1_synthetic.csv and tier1_realpairs.csv under the ckpt's run dir and prints both
tables as markdown.

  CUDA_VISIBLE_DEVICES=1 python -m hoi_flow.eval_tier1 \
      --ckpt hoi_flow/runs/v2_full/ckpt_best.pt --device cuda
"""
import argparse
import csv
import glob
import os

import numpy as np
import torch

from .bridge import sample as bridge_sample
from .data.dataset import HOIFlowDataset
from .sample import COARSE_DIR, SEG_DIR, _PER_FRAME, load_model_enc, load_segment, x0_real
from .state import Normalizer
from .train import _COLS, load_config, make_cond_fn, state_metrics, to_device

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _med(t):
    return float(t.median()) if torch.is_tensor(t) and t.numel() else float("nan")


def _cat(lst):
    return torch.cat(lst) if lst else torch.empty(0)


def _md_table(header, rows, aligns=None):
    """rows: list of lists (already-formatted strings). Returns a markdown table string."""
    out = ["| " + " | ".join(header) + " |"]
    out.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


# ------------------------------------------------------------------ part 1: synthetic
@torch.no_grad()
def eval_synthetic(cfg, norm, model, enc, device, scales, run_dir):
    dcfg, vcfg = cfg["data"], cfg["val"]
    mods = tuple(m for m, on in cfg["conditioning"]["modalities"].items() if on)
    common = dict(seg_dir=dcfg["seg_dir"], seq_len=cfg["seq_len"], normalizer=norm,
                  require_all_modalities=dcfg["require_all_modalities"], modalities=mods,
                  obj_profile=dcfg["obj_profile"], hand_profile=dcfg["hand_profile"])
    n_steps_list = [8, 1]
    variants = ["coarse"] + [f"refined@{n}" for n in n_steps_list]
    results = {}  # (scale, variant) -> {col: median}
    n_segs = None
    for s in scales:
        vd = HOIFlowDataset(split="val", seed=vcfg["seed"], corrupt_scale=(s, s), **common)
        vd.set_epoch(0)
        n_segs = len(vd)
        agg = {v: {c: [] for c in _COLS} for v in variants}
        for i in range(len(vd)):
            seg = to_device(vd[i], device)
            x0, x1, pres = seg["x0"], seg["x1"], seg["presence"]
            cond_full = {k: v.unsqueeze(0) for k, v in seg["cond"].items()}
            cond_fn = make_cond_fn(cond_full, enc, {}, device)
            outs = {"coarse": x0}
            for n in n_steps_list:
                outs[f"refined@{n}"] = bridge_sample(model, x0, cond_fn, pres, n_steps=n,
                                                     window=vcfg["window"], overlap=vcfg["overlap"],
                                                     sigma0=vcfg["sigma0"], seed=0)
            for v in variants:
                mm = state_metrics(outs[v], x1, pres, norm)
                for c in _COLS:
                    if mm[c].numel():
                        agg[v][c].append(mm[c])
            if (i + 1) % 50 == 0:
                print(f"  [synthetic s={s}] {i + 1}/{len(vd)}", flush=True)
        for v in variants:
            results[(s, v)] = {c: _med(_cat(agg[v][c])) for c in _COLS}
        print(f"  [synthetic s={s}] done ({len(vd)} segs)", flush=True)

    # CSV
    csv_path = os.path.join(run_dir, "tier1_synthetic.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scale", "variant"] + _COLS)
        for s in scales:
            for v in variants:
                r = results[(s, v)]
                w.writerow([s, v] + [f"{r[c]:.4f}" for c in _COLS])

    # markdown
    header = ["scale", "variant", "obj_along", "obj_perp", "obj_total", "obj_rot",
              "hand_along", "hand_perp", "theta_mae"]
    rows = []
    for s in scales:
        for v in variants:
            r = results[(s, v)]
            rows.append([f"{s:g}", v] + [f"{r[c]:.2f}" for c in _COLS])
    md = (f"### Table 1 — Synthetic multi-scale val ({n_segs} val segments, P0002/P0015; "
          f"units mm except obj_rot deg, theta_mae rad)\n\n" + _md_table(header, rows))
    return results, md, csv_path


# ------------------------------------------------------------------ part 2: real pairs
def _coarse_hand_presence(coarse, T):
    """[T,2] bool: which hand side the SINGLE-hand real coarse actually fills per frame.
    Real coarse is grasping-hand-only; the segment's `presence` marks BOTH hands, so the
    empty slot (packed at the camera origin) would otherwise contaminate the hand metric
    with a pure along-ray error of magnitude |wrist|. We gate hand metrics to this mask so
    coarse and refined are compared on the same, actually-provided hand."""
    hp = torch.zeros(T, 2, dtype=torch.bool)
    if "hand_mano_global" not in coarse.files:
        return hp
    g = coarse["hand_mano_global"].astype(np.float64)[:T]
    p = coarse["hand_mano_pose"].astype(np.float64)[:T]
    tr = coarse["hand_mano_transl"].astype(np.float64)[:T]
    side = coarse["hand_mano_side"].astype(np.float64)[:T]
    ok = (np.isfinite(g).all(1) & np.isfinite(p).all(1)
          & np.isfinite(tr).all(1) & np.isfinite(side))
    for t in np.where(ok)[0]:
        hp[t, 1 if side[t] > 0.5 else 0] = True
    return hp


@torch.no_grad()
def eval_realpairs(cfg, norm, model, enc, device, coarse_dir, seg_dir, run_dir):
    vcfg = cfg["val"]
    mods = tuple(m for m, on in cfg["conditioning"]["modalities"].items() if on)
    files = sorted(glob.glob(os.path.join(coarse_dir, "coarse_*.npz")))
    recs = []  # per file: {key, has_mano, mc{col:tensor}, mr{col:tensor}}
    for k, cp in enumerate(files):
        wid = os.path.basename(cp)[len("coarse_"):-len(".npz")]
        seg_path = os.path.join(seg_dir, f"seg_{wid}.npz")
        if not os.path.exists(seg_path):
            continue
        coarse = np.load(cp, allow_pickle=True)
        z, x1, valid, presence, cond_full = load_segment(seg_path, mods, device)
        T = min(len(coarse["obj_pose"]), z["obj_pose"].shape[0])
        x0, has_mano = x0_real(coarse, T)
        x1 = x1[:T]
        presence = presence[:T].to(device)
        cond_full = {kk: (v[:, :T] if kk in _PER_FRAME else v) for kk, v in cond_full.items()}
        cond_fn = make_cond_fn(cond_full, enc, {}, device)
        x0n = torch.nan_to_num(norm.transform(x0)).to(device)
        x1n = norm.transform(x1).to(device)
        # model refines whatever the segment says is present (both hands); but hand METRICS
        # are gated to the grasping side the coarse actually provides, for a fair comparison.
        ref8 = bridge_sample(model, x0n, cond_fn, presence, n_steps=8, window=vcfg["window"],
                             overlap=vcfg["overlap"], sigma0=vcfg["sigma0"], seed=0)
        hand_pres = (presence.cpu() & _coarse_hand_presence(coarse, T)).to(device)
        mc = state_metrics(x0n, x1n, hand_pres, norm)  # obj cols ignore presence; hand cols gated
        mr = state_metrics(ref8, x1n, hand_pres, norm)
        key = _med(mc["obj_total"])  # median coarse object translation error (mm)
        recs.append(dict(wid=wid, key=key, has_mano=has_mano, mc=mc, mr=mr))
        if (k + 1) % 40 == 0:
            print(f"  [realpairs] {k + 1}/{len(files)}", flush=True)
    print(f"  [realpairs] done ({len(recs)} pairs with matching segments)", flush=True)

    strata = [("good (<30mm)", lambda x: x < 30.0),
              ("mid (30-100mm)", lambda x: 30.0 <= x < 100.0),
              ("catastrophic (>=100mm)", lambda x: x >= 100.0)]
    obj_cols = ["obj_along", "obj_perp", "obj_total", "obj_rot"]
    hand_cols = ["hand_along", "hand_perp", "theta_mae"]

    csv_path = os.path.join(run_dir, "tier1_realpairs.csv")
    fcsv = open(csv_path, "w", newline="")
    wcsv = csv.writer(fcsv)
    wcsv.writerow(["stratum", "n_files", "n_mano", "variant"] + obj_cols + hand_cols)

    header = ["stratum", "n", "n_mano", "variant", "obj_total", "obj_rot",
              "hand_along", "hand_perp", "theta_mae"]
    rows = []
    for name, pred in strata:
        sr = [r for r in recs if pred(r["key"])]
        n = len(sr)
        nm = sum(1 for r in sr if r["has_mano"])
        for var, mk in (("coarse", "mc"), ("refined@8", "mr")):
            obj = {c: _med(_cat([r[mk][c] for r in sr if r[mk][c].numel()])) for c in obj_cols}
            hnd = {c: _med(_cat([r[mk][c] for r in sr if r["has_mano"] and r[mk][c].numel()]))
                   for c in hand_cols}
            wcsv.writerow([name, n, nm, var]
                          + [f"{obj[c]:.4f}" for c in obj_cols]
                          + [f"{hnd[c]:.4f}" for c in hand_cols])
            rows.append([name, n, nm, var, f"{obj['obj_total']:.1f}", f"{obj['obj_rot']:.1f}",
                         f"{hnd['hand_along']:.1f}", f"{hnd['hand_perp']:.1f}",
                         f"{hnd['theta_mae']:.3f}"])
    fcsv.close()
    md = ("### Table 2 — Real-pairs stratified (FoundationPose coarse pairs; stratum key = "
          "median coarse object translation error vs seg GT; units mm except obj_rot deg, "
          "theta_mae rad; hand cols pooled over the n_mano MANO-carrying files only)\n\n"
          + _md_table(header, rows))
    return recs, md, csv_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(REPO, "hoi_flow/runs/v2_full/ckpt_best.pt"))
    ap.add_argument("--scales", default="0.25,0.5,1.0")
    ap.add_argument("--coarse_dir", default=COARSE_DIR)
    ap.add_argument("--seg_dir", default=SEG_DIR)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--part", default="both", choices=["synthetic", "real", "both"])
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    device = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    scales = [float(s) for s in a.scales.split(",")]

    run_dir = a.out_dir or os.path.dirname(os.path.abspath(a.ckpt))
    os.makedirs(run_dir, exist_ok=True)
    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(REPO, "hoi_flow/configs/base.yaml")
    cfg = load_config(cfg_path, [])
    np_path = cfg["data"]["normalizer"]
    np_path = np_path if os.path.isabs(np_path) else os.path.join(REPO, np_path)
    norm = Normalizer.load(np_path)

    ckpt = torch.load(a.ckpt, map_location=device)
    model, enc = load_model_enc(ckpt, cfg, device)
    print(f"ckpt {a.ckpt}  step {ckpt.get('step')}  backend {ckpt.get('rgb_backbone')}  "
          f"device {device}  scales {scales}", flush=True)

    mds = []
    if a.part in ("synthetic", "both"):
        _, md1, p1 = eval_synthetic(cfg, norm, model, enc, device, scales, run_dir)
        print("\n" + md1 + "\n", flush=True)
        print(f"wrote {p1}", flush=True)
        mds.append(md1)
    if a.part in ("real", "both"):
        _, md2, p2 = eval_realpairs(cfg, norm, model, enc, device, a.coarse_dir, a.seg_dir, run_dir)
        print("\n" + md2 + "\n", flush=True)
        print(f"wrote {p2}", flush=True)
        mds.append(md2)

    # dump a combined markdown blob for easy copy into RESULTS.md
    with open(os.path.join(run_dir, "tier1_tables.md"), "w") as f:
        f.write("\n\n".join(mds) + "\n")
    print(f"wrote {os.path.join(run_dir, 'tier1_tables.md')}", flush=True)


if __name__ == "__main__":
    main()
