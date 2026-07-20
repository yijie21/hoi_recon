"""CLI: refine a coarse HOI trajectory with the trained bridge flow model.

  python -m hoi_flow.sample --ckpt <ckpt.pt> --window_id <id>
      [--coarse_dir /workspace/datasets/hot3d/hoi_coarse_pairs] [--seg_dir ...]
      [--out ...] [--n_steps 8] [--synthetic] [--device cpu]

Loads the EMA model+encoders from a training checkpoint (train.py's format:
{'ema_model','ema_enc','rgb_backbone',...}), builds the observation conditioning from the
segment npz (same encode + window-slice pattern as train.py's val), builds the coarse
state x0, runs bridge.sample, denormalizes, writes a refined npz, and prints a metric table
(coarse / refined@n_steps / refined@1) vs the segment GT.

x0 source
---------
--synthetic : x0 = a calibrated corruption of the GT (dataset.py's corrupt path). Works
              before any real coarse pairs exist, and is the plumbing self-test.
default     : x0 from the REAL coarse pair coarse_<window_id>.npz —
              * object : coarse obj_pose (GT-mesh canonical, camera frame) -> directly the
                obj state block.
              * hand   : if the pair carries MANO params (hand_mano_global/pose/transl/side,
                added by joint_opt.py's additive export), FK them to the MANO root, map
                through wrist_offset.json into the UmeTrack-wrist state convention
                (T_cam_from_wrist ~= T_cam_from_manoroot @ C[side]), and convert the FULL
                axis-angle pose to PCA-15 (state thetas). The real coarse is SINGLE-hand
                (grasping hand); it fills only that side's slot, the other stays invalid.
              * FALLBACK (missing MANO params, or per-frame coarse-hand missing / NaN): the
                hand blocks are left INVALID (not GT-corrupted) — presence comes from the
                segment's hands_present and the model refines what's there; the OBJECT is
                still refined. (The ~first batch of coarse pairs predates the MANO export and
                exercises exactly this fallback.)

Convention note (left hand): joint_opt emits RIGHT-hand-model MANO with left frames mirrored
x->-x, and MANO_LEFT on this box is a fabricated right-mirror, so the left path applies the
same x-mirror before C[left]. It is correct-by-construction for the RIGHT hand; the LEFT path
is a reasonable approximation that awaits real left-hand MANO pairs to validate end-to-end.
"""
import argparse
import json
import os
import pickle

import numpy as np
import torch
import torch.hub
from scipy.spatial.transform import Rotation

from .bridge import sample as bridge_sample
from .conditioning import CondEncoders
from .model import HOIDiT
from .state import Normalizer, pack_state, unpack_state
from .data import corrupt as C
from .data.dataset import _corrupt_hand_se3
from .data.solve_wrist_offset import load_offsets
from .train import state_metrics, _COLS, make_cond_fn, load_config

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEG_DIR = "/workspace/datasets/hot3d/hoi_segments"
COARSE_DIR = "/workspace/datasets/hot3d/hoi_coarse_pairs"
MANO_PKL = os.path.join(REPO, "render_and_compare/checkpoints/mano/MANO_RIGHT_np.pkl")
MANO_SCRATCH = ("/tmp/claude-1002/-workspace-code-hoi-recon/"
                "cb99b7a6-8766-4fc9-a273-b7123285115d/scratchpad/mano_models")
_PER_FRAME = ("rgb", "depth", "seg_mask")


# --------------------------------------------------------------- model / encoders (EMA)
def load_model_enc(ckpt, cfg, device):
    """Rebuild the EMA model + encoders from a checkpoint. Loads the frozen DINOv2 weights
    FROM the ckpt (they are stored there), building the arch offline (pretrained=False) so no
    network/hub download is needed."""
    model = HOIDiT(cfg["model"]).to(device)
    model.load_state_dict(ckpt["ema_model"])
    model.eval()

    backend = ckpt.get("rgb_backbone", "dino")
    enc_cfg = dict(cfg["conditioning"]["modalities"])
    enc_cfg["rgb_backbone"] = backend
    if backend == "dino" and enc_cfg.get("rgb"):
        orig = torch.hub.load

        def patched(*a, **k):
            k.setdefault("pretrained", False)
            k.setdefault("verbose", False)
            return orig(*a, **k)

        torch.hub.load = patched
        try:
            enc = CondEncoders(model.d, enc_cfg).to(device)
            enc.encoders["rgb"]._load(device)     # build dino arch offline
        finally:
            torch.hub.load = orig
    else:
        enc = CondEncoders(model.d, enc_cfg).to(device)
    enc.load_state_dict(ckpt["ema_enc"])          # overwrite with trained (incl. dino) weights
    enc.eval()
    return model, enc


# ---------------------------------------------------------------------- segment loading
def load_segment(seg_path, modalities, device):
    """seg npz -> (z, x1 raw state, valid, presence [T,2], cond_full [1,T,...] on device)."""
    z = np.load(seg_path, allow_pickle=True)
    x1, valid = pack_state(
        torch.from_numpy(z["obj_pose"].astype(np.float64)).float(),
        torch.from_numpy(z["hand_wrist"].astype(np.float64)).float(),
        torch.from_numpy(z["hand_mano"][:, :, :15].astype(np.float64)).float())
    present = torch.from_numpy(z["hands_present"].astype(bool))
    wrist_valid = torch.stack([valid[:, 9], valid[:, 33]], dim=-1)
    presence = present & wrist_valid

    cond = {}
    if "rgb" in modalities:
        cond["rgb"] = torch.from_numpy(z["rgb"]).permute(0, 3, 1, 2).float() / 255.0
    if "depth" in modalities and "depth_mm" in z.files:
        cond["depth"] = torch.from_numpy(z["depth_mm"].astype(np.float32) / 1000.0)
    if "seg_mask" in modalities and "seg_mask" in z.files:
        cond["seg_mask"] = torch.from_numpy(z["seg_mask"].astype(np.int64))
    if "obj_points" in modalities:
        cond["obj_points"] = torch.from_numpy(z["obj_verts"].astype(np.float32))
    if "K" in modalities:
        cond["K"] = torch.from_numpy(z["K"].astype(np.float32))
    cond_full = {k: v.unsqueeze(0).to(device) for k, v in cond.items()}
    return z, x1, valid, presence, cond_full


# ------------------------------------------------------------------------- x0 : synthetic
def x0_synthetic(z, obj_prof, hand_prof, seed):
    """Calibrated corruption of the GT (mirrors dataset.py's _corrupt), full T."""
    rng = np.random.default_rng(seed)
    obj_pose = z["obj_pose"].astype(np.float64)
    hand_wrist = z["hand_wrist"].astype(np.float64)
    hand_theta = z["hand_mano"][:, :, :15].astype(np.float64)
    obj_c = C.corrupt_object(obj_pose, obj_prof, rng)
    L = len(hand_wrist)
    wrist_c = np.full((L, 2, 4, 4), np.nan)
    theta_c = np.full((L, 2, 15), np.nan)
    for h in (0, 1):
        wc, tc = _corrupt_hand_se3(hand_wrist[:, h], hand_theta[:, h], hand_prof, rng, 0.05, 4.0)
        wrist_c[:, h], theta_c[:, h] = wc, tc
    x0, _ = pack_state(torch.from_numpy(obj_c).float(),
                       torch.from_numpy(wrist_c).float(),
                       torch.from_numpy(theta_c).float())
    return x0


# ------------------------------------------------------------------------- x0 : real coarse
class _CoarseHand:
    """Lazily-built MANO FK + PCA basis for mapping coarse MANO params into the state's
    (UmeTrack-wrist convention, PCA-15) hand blocks. RIGHT-hand smplx.MANO (use_pca=False,
    flat_hand_mean=True) reproduces joint_opt's MANOLayer full-pose hand."""

    def __init__(self):
        import smplx
        os.makedirs(MANO_SCRATCH, exist_ok=True)
        dst = os.path.join(MANO_SCRATCH, "MANO_RIGHT.pkl")
        if not os.path.exists(dst):
            os.symlink(MANO_PKL, dst)
        self.layer = smplx.MANO(dst, is_rhand=True, use_pca=False, flat_hand_mean=True,
                                batch_size=1)
        with open(MANO_PKL, "rb") as f:
            d = pickle.load(f, encoding="latin1")
        self.comp15 = d["hands_components"][:15].astype(np.float64)     # (15,45)
        self.comp15_pinv = np.linalg.pinv(self.comp15)                  # (45,15)
        self.hands_mean = d["hands_mean"].astype(np.float64)           # (45,)
        self.offsets = load_offsets()                                  # {'left':4x4,'right':4x4}

    def to_state(self, coarse, T):
        """coarse pair -> (wrist [T,2,4,4], thetas [T,2,15]) with only the coarse hand's
        side slot filled (the rest NaN => invalid)."""
        g = coarse["hand_mano_global"].astype(np.float64)[:T]
        p = coarse["hand_mano_pose"].astype(np.float64)[:T]
        tr = coarse["hand_mano_transl"].astype(np.float64)[:T]
        side = coarse["hand_mano_side"].astype(np.float64)[:T]
        wrist = np.full((T, 2, 4, 4), np.nan, np.float32)
        thetas = np.full((T, 2, 15), np.nan, np.float32)
        ok = (np.isfinite(g).all(1) & np.isfinite(tr).all(1)
              & np.isfinite(p).all(1) & np.isfinite(side))
        if not ok.any():
            return wrist, thetas
        idx = np.where(ok)[0]
        out = self.layer(betas=torch.zeros(len(idx), 10),
                         global_orient=torch.zeros(len(idx), 3),   # root-relative FK; global applied below
                         hand_pose=torch.tensor(p[idx], dtype=torch.float32),
                         transl=torch.zeros(len(idx), 3), return_verts=True)
        # MANO root joint (joint 0) is the kinematic PIVOT: its position is J0 (betas=0
        # template), INDEPENDENT of global_orient — verified. So root_cam = mirror@J0 + transl
        # and R_root = R(global). Identical construction to solve_wrist_offset's GT root, so
        # the same offset C transfers. (betas=0 here and in the GT solve => J0 matches.)
        j0 = out.joints[:, 0].detach().numpy()                     # (K,3) = J0 (global-independent)
        pca = (p[idx] - self.hands_mean) @ self.comp15_pinv        # (K,15) -> state thetas
        for k, t in enumerate(idx):
            s = 1 if side[t] > 0.5 else 0
            Rg = Rotation.from_rotvec(g[t]).as_matrix()
            mx = np.diag([1.0, 1.0, 1.0]) if s == 1 else np.diag([-1.0, 1.0, 1.0])
            R_root = mx @ Rg @ mx                                  # mirror for left (fabricated-left)
            root_cam = mx @ j0[k] + tr[t]
            M = np.eye(4); M[:3, :3] = R_root; M[:3, 3] = root_cam
            wrist[t, s] = (M @ self.offsets["right" if s == 1 else "left"]).astype(np.float32)
            thetas[t, s] = pca[k].astype(np.float32)
        return wrist, thetas


def x0_real(coarse, T):
    """object from coarse obj_pose; hand invalid unless MANO params present."""
    obj_pose = coarse["obj_pose"].astype(np.float64)[:T]
    has_mano = "hand_mano_global" in coarse.files
    if has_mano:
        wrist, thetas = _CoarseHand().to_state(coarse, T)
    else:
        wrist = np.full((T, 2, 4, 4), np.nan)
        thetas = np.full((T, 2, 15), np.nan)
    x0, _ = pack_state(torch.from_numpy(obj_pose).float(),
                       torch.from_numpy(wrist).float(),
                       torch.from_numpy(thetas).float())
    return x0, has_mano


# --------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--window_id", required=True)
    ap.add_argument("--coarse_dir", default=COARSE_DIR)
    ap.add_argument("--seg_dir", default=SEG_DIR)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n_steps", type=int, default=8)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    device = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")

    cfg_path = os.path.join(os.path.dirname(os.path.abspath(a.ckpt)), "config.yaml")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(REPO, "hoi_flow/configs/base.yaml")
    cfg = load_config(cfg_path, [])
    modalities = tuple(m for m, on in cfg["conditioning"]["modalities"].items() if on)
    np_path = cfg["data"]["normalizer"]
    np_path = np_path if os.path.isabs(np_path) else os.path.join(REPO, np_path)
    norm = Normalizer.load(np_path)
    vcfg = cfg["val"]

    ckpt = torch.load(a.ckpt, map_location=device)
    model, enc = load_model_enc(ckpt, cfg, device)

    seg_path = os.path.join(a.seg_dir, f"seg_{a.window_id}.npz")
    if not os.path.exists(seg_path):
        raise SystemExit(f"segment not found: {seg_path}")
    z, x1, valid, presence, cond_full = load_segment(seg_path, modalities, device)

    # --- build x0 (and align lengths for the real single-window case) ---
    if a.synthetic:
        stats = C.load_stats()
        x0 = x0_synthetic(z, C.object_profile(stats, cfg["data"]["obj_profile"]),
                          C.hand_profile(stats, cfg["data"]["hand_profile"]), a.seed)
        T = x0.shape[0]
        mode = "synthetic (calibrated GT corruption)"
        has_mano = True
    else:
        cp = os.path.join(a.coarse_dir, f"coarse_{a.window_id}.npz")
        if not os.path.exists(cp):
            raise SystemExit(f"coarse pair not found: {cp} (use --synthetic, or generate it)")
        coarse = np.load(cp, allow_pickle=True)
        T = min(len(coarse["obj_pose"]), z["obj_pose"].shape[0])
        x0, has_mano = x0_real(coarse, T)
        mode = f"real coarse ({'MANO hand' if has_mano else 'OBJECT-ONLY: hand invalid fallback'})"

    # slice GT/cond/presence to T
    x1 = x1[:T]; presence = presence[:T].to(device)
    cond_full = {k: (v[:, :T] if k in _PER_FRAME else v) for k, v in cond_full.items()}
    cond_fn = make_cond_fn(cond_full, enc, {}, device)

    x0n = torch.nan_to_num(norm.transform(x0)).to(device)
    x1n = norm.transform(x1).to(device)

    n_list = [a.n_steps, 1] if a.n_steps != 1 else [1]
    outs = {"coarse": x0n}
    for n in n_list:
        outs[f"refined@{n}"] = bridge_sample(model, x0n, cond_fn, presence, n_steps=n,
                                             window=vcfg["window"], overlap=vcfg["overlap"],
                                             sigma0=vcfg["sigma0"], seed=0)

    # ---- metric table (coarse / refined@n / refined@1) vs seg GT ----
    variants = ["coarse"] + [f"refined@{n}" for n in n_list]
    print(f"\nwindow {a.window_id}  |  mode: {mode}  |  T={T}  device={device}")
    print(f"presence L/R frames: {int(presence[:,0].sum())}/{int(presence[:,1].sum())}  "
          f"units: mm except obj_rot deg, theta_mae rad")
    hdr = f"{'variant':<12}" + "".join(f"{c:>11}" for c in _COLS)
    print("  " + hdr)
    for v in variants:
        mm = state_metrics(outs[v], x1n, presence, norm)
        row = {c: (float(mm[c].median()) if mm[c].numel() else float("nan")) for c in _COLS}
        print("  " + f"{v:<12}" + "".join(f"{row[c]:>11.2f}" for c in _COLS))

    # ---- write refined npz (refined@n_steps) ----
    out_path = a.out or os.path.join(a.coarse_dir, f"refined_{a.window_id}.npz")
    ref = norm.inverse(outs[f"refined@{a.n_steps}"].detach().cpu())
    u = unpack_state(ref)
    np.savez(out_path,
             obj_pose=u["obj_pose"].numpy().astype(np.float32),
             hand_wrist=u["hand_wrist"].numpy().astype(np.float32),
             hand_thetas=u["hand_thetas"].numpy().astype(np.float32))
    print(f"\nwrote {out_path}  {{obj_pose[{T},4,4], hand_wrist[{T},2,4,4], hand_thetas[{T},2,15]}}")


if __name__ == "__main__":
    main()
