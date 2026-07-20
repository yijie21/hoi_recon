"""Train the bridge flow-matching HOI refiner (config-driven).

  python -m hoi_flow.train --config hoi_flow/configs/base.yaml train.name=myrun ...

The loop is the thin glue over the authored contracts: HOIFlowDataset -> (x0 coarse, x1 GT,
valid, presence, cond) ; CondEncoders(cond, drop) -> cross-attn tokens ; bridge_loss(model,
...) -> masked endpoint-MSE. AdamW + bf16 autocast + grad-clip + EMA(model,encoders) for val.

The headline is the VAL routine (`run_val`): on a FIXED set of val segments with a FIXED
corruption seed it refines with bridge.sample at n_steps=8 (multi-step) AND n_steps=1 (the
"regressor control" — same net, one Euler step), and reports object trans (along/perp mm),
object rot (deg), hand wrist trans (along/perp mm, where present) and theta MAE for
coarse / refined@1 / refined@8 — all denormalized, against GT.

CLI overrides are `dotted.key=value` (value parsed as YAML): e.g. train.steps=300 seq_len=32.
Single-GPU by default; torchrun (WORLD_SIZE>1) wraps the model in DDP.
"""
import argparse
import copy
import csv
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

# Big batches (rgb+depth+mask x 32 frames ~ 1.8 GB each) x workers x prefetch overflow the
# container's /dev/shm cap; file_system strategy passes tensors via disk-backed files instead.
torch.multiprocessing.set_sharing_strategy("file_system")

from .bridge import bridge_loss, sample
from .conditioning import CondEncoders
from .data.dataset import HOIFlowDataset
from .geometry import decompose_along_ray, so3_geodesic
from .model import HOIDiT
from .state import Normalizer, unpack_state

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../hoi_recon


# ------------------------------------------------------------------- config
def load_config(path, overrides):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for ov in overrides:
        if "=" not in ov:
            raise SystemExit(f"bad override (want key=value): {ov}")
        k, v = ov.split("=", 1)
        node = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        val = yaml.safe_load(v)
        if isinstance(val, str):  # YAML 1.1 quirk: "2e-4" is a str (no dot) — coerce numeric strings
            try:
                val = float(val)
            except ValueError:
                pass
        node[parts[-1]] = val
    return cfg


def _abspath(p):
    return p if os.path.isabs(p) else os.path.join(REPO, p)


# ------------------------------------------------------------------- EMA
class EMA:
    def __init__(self, module, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(module).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, module):
        for s, p in zip(self.shadow.state_dict().values(), module.state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(self.decay).add_(p.detach().to(s.dtype), alpha=1 - self.decay)
            else:
                s.copy_(p)


# ------------------------------------------------------------------- encoders (dino w/ tinycnn fallback)
def build_encoders(cfg_cond, d_model, device):
    enc_cfg = dict(cfg_cond["modalities"])
    backend = cfg_cond.get("rgb_backbone", "dino")
    enc_cfg["rgb_backbone"] = backend
    enc = CondEncoders(d_model, enc_cfg).to(device)
    if backend == "dino" and enc_cfg.get("rgb"):
        try:  # warm up the lazy DINOv2 load NOW so it is a registered submodule before EMA deepcopy
            enc.encoders["rgb"]._load(device)
        except Exception as e:  # noqa: BLE001 — network/hub failure -> download-free tinycnn
            print(f"[warn] DINOv2 load failed ({type(e).__name__}: {e}); falling back to tinycnn", flush=True)
            enc_cfg["rgb_backbone"] = backend = "tinycnn"
            enc = CondEncoders(d_model, enc_cfg).to(device)
    return enc, backend


# ------------------------------------------------------------------- data plumbing
def to_device(batch, device):
    out = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items() if k != "cond"}
    out["cond"] = {k: v.to(device) for k, v in batch["cond"].items()}
    return out


def cycle(dl, ds):
    ep = 0
    while True:
        ds.set_epoch(ep)
        for b in dl:
            yield b
        ep += 1


def make_cond_fn(cond_full, enc, drop, device):
    """cond_full: dict with a leading batch dim of 1 (per-frame [1,T,...], static [1,C,...]).
    Returns callable(slice)->tokens[1,N,d] re-encoding that window so frame indices are LOCAL
    (0..w-1), matching training and the model's own local frame positions."""
    per_frame = ("rgb", "depth", "seg_mask")

    def fn(sl):
        wb = {}
        for k, v in cond_full.items():
            wb[k] = v[:, sl] if k in per_frame else v
        return enc(wb, drop)[0]
    return fn


# ------------------------------------------------------------------- metrics
def _median(t):
    return float(t.median()) if t.numel() else float("nan")


def state_metrics(x_norm, x1_norm, presence, norm):
    """Per-frame errors of a (normalized) state vs GT, denormalized. Returns tensors."""
    pu = unpack_state(norm.inverse(x_norm.float().cpu()))
    gu = unpack_state(norm.inverse(x1_norm.float().cpu()))
    tg, tp = gu["obj_pose"][:, :3, 3], pu["obj_pose"][:, :3, 3]
    e = tp - tg
    al, pe = decompose_along_ray(e, tg)
    m = dict(
        obj_along=al.abs().squeeze(-1) * 1000.0,
        obj_perp=pe.norm(dim=-1) * 1000.0,
        obj_total=e.norm(dim=-1) * 1000.0,
        obj_rot=torch.rad2deg(so3_geodesic(gu["obj_pose"][:, :3, :3], pu["obj_pose"][:, :3, :3])),
    )
    ha, hp, th = [], [], []
    pres = presence.cpu().bool()
    for h in (0, 1):
        msk = pres[:, h]
        if msk.any():
            wg, wp = gu["hand_wrist"][:, h, :3, 3][msk], pu["hand_wrist"][:, h, :3, 3][msk]
            e2 = wp - wg
            a, p = decompose_along_ray(e2, wg)
            ha.append(a.abs().squeeze(-1) * 1000.0)
            hp.append(p.norm(dim=-1) * 1000.0)
            th.append((pu["hand_thetas"][:, h][msk] - gu["hand_thetas"][:, h][msk]).abs().mean(dim=-1))
    m["hand_along"] = torch.cat(ha) if ha else torch.empty(0)
    m["hand_perp"] = torch.cat(hp) if hp else torch.empty(0)
    m["theta_mae"] = torch.cat(th) if th else torch.empty(0)
    return m


_COLS = ["obj_along", "obj_perp", "obj_total", "obj_rot", "hand_along", "hand_perp", "theta_mae"]


@torch.no_grad()
def run_val(step, val_set, model, enc, norm, vcfg, device, csv_path):
    model.eval(); enc.eval()
    n_steps_list = vcfg["n_steps"]
    variants = ["coarse"] + [f"refined@{n}" for n in n_steps_list]
    agg = {v: {c: [] for c in _COLS} for v in variants}
    for seg in val_set:
        seg = to_device(seg, device)
        x0, x1, pres = seg["x0"], seg["x1"], seg["presence"]
        cond_full = {k: v.unsqueeze(0) for k, v in seg["cond"].items()}
        cond_fn = make_cond_fn(cond_full, enc, {}, device)
        outs = {"coarse": x0}
        for n in n_steps_list:
            outs[f"refined@{n}"] = sample(model, x0, cond_fn, pres, n_steps=n,
                                          window=vcfg["window"], overlap=vcfg["overlap"],
                                          sigma0=vcfg["sigma0"], seed=0)
        for v in variants:
            mm = state_metrics(outs[v], x1, pres, norm)
            for c in _COLS:
                if mm[c].numel():
                    agg[v][c].append(mm[c])
    rows = {}
    for v in variants:
        rows[v] = {c: _median(torch.cat(agg[v][c]) if agg[v][c] else torch.empty(0)) for c in _COLS}

    # print compact table
    hdr = f"{'variant':<12}" + "".join(f"{c:>11}" for c in _COLS)
    print(f"\n[val @ step {step}]  ({len(val_set)} segs)  units: mm except obj_rot deg, theta_mae rad", flush=True)
    print("  " + hdr, flush=True)
    for v in variants:
        print("  " + f"{v:<12}" + "".join(f"{rows[v][c]:>11.2f}" for c in _COLS), flush=True)
    print("", flush=True)

    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["step", "variant"] + _COLS)
        for v in variants:
            w.writerow([step, v] + [f"{rows[v][c]:.4f}" for c in _COLS])
    model.train(); enc.train()
    return rows


# ------------------------------------------------------------------- checkpoint
def save_ckpt(path, step, core, enc, ema_m, ema_e, opt, sched, backend):
    torch.save({"step": step, "model": core.state_dict(), "enc": enc.state_dict(),
                "ema_model": ema_m.shadow.state_dict(), "ema_enc": ema_e.shadow.state_dict(),
                "opt": opt.state_dict(), "sched": sched.state_dict(),
                "rgb_backbone": backend}, path)


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO, "hoi_flow/configs/base.yaml"))
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)

    # DDP (nice-to-have; single-GPU by default)
    world = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world > 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if ddp:
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = (not ddp) or local_rank == 0

    tcfg, dcfg, vcfg = cfg["train"], cfg["data"], cfg["val"]
    seq_len = cfg["seq_len"]
    seed = tcfg["seed"]
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    run_dir = os.path.join(REPO, "hoi_flow/runs", tcfg["name"])
    os.makedirs(run_dir, exist_ok=True)
    if is_main:
        with open(os.path.join(run_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    train_csv = os.path.join(run_dir, "train_log.csv")
    val_csv = os.path.join(run_dir, "val_log.csv")

    norm = Normalizer.load(_abspath(dcfg["normalizer"]))

    # datasets
    common = dict(seg_dir=dcfg["seg_dir"], seq_len=seq_len, normalizer=norm,
                  require_all_modalities=dcfg["require_all_modalities"],
                  modalities=tuple(m for m, on in cfg["conditioning"]["modalities"].items() if on),
                  obj_profile=dcfg["obj_profile"], hand_profile=dcfg["hand_profile"])
    train_ds = HOIFlowDataset(split="train", seed=dcfg["seed"],
                              corrupt_scale=tuple(dcfg.get("corrupt_scale", (1.0, 1.0))), **common)
    val_ds = HOIFlowDataset(split="val", seed=vcfg["seed"], **common)
    val_ds.set_epoch(0)
    n_val = min(vcfg["n_segments"], len(val_ds))
    val_set = [val_ds[i] for i in range(n_val)]  # FIXED corruption/window (fixed seed+epoch)
    if is_main:
        print(f"train segs {len(train_ds)}  val segs {len(val_ds)} (using {n_val})  "
              f"modalities {common['modalities']}", flush=True)

    dl = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, drop_last=True,
                    num_workers=tcfg["num_workers"], collate_fn=HOIFlowDataset.collate,
                    pin_memory=True)

    # model + encoders
    model = HOIDiT(cfg["model"]).to(device)
    enc, backend = build_encoders(cfg["conditioning"], model.d, device)
    if is_main:
        print(f"rgb_backbone in use: {backend}", flush=True)

    core = model
    if ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        enc = nn.parallel.DistributedDataParallel(enc, device_ids=[local_rank], find_unused_parameters=True)
        core = model.module

    ema_m = EMA(core, tcfg["ema_decay"])
    ema_e = EMA(enc.module if ddp else enc, tcfg["ema_decay"])

    params = [p for p in list(core.parameters()) + list((enc.module if ddp else enc).parameters())
              if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tcfg["lr"], weight_decay=tcfg["wd"], betas=(0.9, 0.95))
    steps, warmup = tcfg["steps"], tcfg["warmup"]

    def lr_lambda(s):
        if s < warmup:
            return (s + 1) / max(warmup, 1)
        if tcfg["sched"] == "constant":
            return 1.0
        prog = (s - warmup) / max(steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    start = 0
    best_score = float("inf")
    ckpt_path = os.path.join(run_dir, "ckpt.pt")
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        core.load_state_dict(ck["model"]); (enc.module if ddp else enc).load_state_dict(ck["enc"])
        ema_m.shadow.load_state_dict(ck["ema_model"]); ema_e.shadow.load_state_dict(ck["ema_enc"])
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start = ck["step"]
        if is_main:
            print(f"resumed from {ckpt_path} @ step {start}", flush=True)

    block_weights = cfg["bridge"].get("block_weights")
    sigma0 = cfg["bridge"]["sigma0"]
    dropout = cfg["conditioning"]["dropout"]
    enabled = [m for m in common["modalities"]]

    data_iter = cycle(dl, train_ds)
    model.train(); enc.train()
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time(); t_win = time.time(); win_steps = 0

    for step in range(start, steps):
        batch = to_device(next(data_iter), device)
        drop = {m: (random.random() < dropout.get(m, 0.0)) for m in enabled}
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cond_tokens = enc(batch["cond"], drop)[0]
            loss, logs = bridge_loss(core if not ddp else model, batch["x0"], batch["x1"],
                                     batch["valid"], cond_tokens, batch["presence"],
                                     sigma0=sigma0, block_weights=block_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, tcfg["grad_clip"])
        opt.step(); sched.step()
        ema_m.update(core); ema_e.update(enc.module if ddp else enc)
        win_steps += 1

        if is_main and (step % tcfg["log_every"] == 0 or step == steps - 1):
            sps = win_steps / (time.time() - t_win + 1e-9)
            print(f"step {step:6d}  loss {logs['loss']:.4f}  "
                  f"obj_rot {logs['mse_obj_rot']:.4f} obj_trans {logs['mse_obj_trans']:.4f} "
                  f"hand {logs['mse_hand']:.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                  f"{sps:.2f} it/s  drop={''.join(k[0] for k,d in drop.items() if d) or '-'}",
                  flush=True)
            new = not os.path.exists(train_csv)
            with open(train_csv, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["step", "loss", "mse_obj_rot", "mse_obj_trans", "mse_hand", "lr", "it_s"])
                w.writerow([step, f"{logs['loss']:.5f}", f"{logs['mse_obj_rot']:.5f}",
                            f"{logs['mse_obj_trans']:.5f}", f"{logs['mse_hand']:.5f}",
                            f"{sched.get_last_lr()[0]:.3e}", f"{sps:.3f}"])
            t_win = time.time(); win_steps = 0

        if is_main and step > start and step % tcfg["val_every"] == 0:
            rows = run_val(step, val_set, ema_m.shadow, ema_e.shadow, norm, vcfg, device, val_csv)
            # keep the best-by-val weights: mid-training peaks are otherwise lost to overfitting
            ref = rows[f"refined@{vcfg['n_steps'][0]}"]
            score = ref["obj_total"] + ref["hand_along"]
            if score < best_score:
                best_score = score
                save_ckpt(os.path.join(run_dir, "ckpt_best.pt"), step, core,
                          enc.module if ddp else enc, ema_m, ema_e, opt, sched, backend)
                print(f"  new best @ {step} (obj_total+hand_along = {score:.2f} mm) -> ckpt_best.pt", flush=True)
        if is_main and step > start and step % tcfg["ckpt_every"] == 0:
            save_ckpt(ckpt_path, step, core, enc.module if ddp else enc, ema_m, ema_e, opt, sched, backend)

    if is_main:
        run_val(steps, val_set, ema_m.shadow, ema_e.shadow, norm, vcfg, device, val_csv)
        save_ckpt(ckpt_path, steps, core, enc.module if ddp else enc, ema_m, ema_e, opt, sched, backend)
        dt = time.time() - t0
        vram = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"\ndone: {steps - start} steps in {dt:.1f}s ({(steps - start) / dt:.2f} it/s)  "
              f"peak VRAM {vram:.2f} GB  ckpt -> {ckpt_path}", flush=True)
    if ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
