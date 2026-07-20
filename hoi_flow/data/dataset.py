"""Dataset glue for the bridge flow-matching HOI refiner.

Wraps the clean HOT3D interaction segments (`seg_*.npz`, see
`compare/hot3d/HOI_CLIPS.md`) and turns each into a training example for the bridge:

  x1  = GT normalized 57-D state trajectory  (the endpoint the model predicts)
  x0  = COARSE normalized state              (a calibrated corruption of x1 — what a
                                              real coarse tracker would emit)
  valid    [t,57]  = GT per-dim validity mask (absent hands/frames -> False)
  presence [t,2]   = per-hand presence flags (model input + hand-metric gating)
  cond dict        = the requested+available observation modalities

State packing / normalization is `state.py`; corruption is the calibrated sampler in
`corrupt.py` (applied in RAW pose space, then packed+normalized). Corruption AND the
random window are deterministic per (seed, epoch, idx) so a fixed epoch gives a fixed
batch (used for the val set); `set_epoch()` reshuffles the corruption each train epoch.

The mmap slice-read (no JPEG decode, no tar) is reimplemented locally so the window and
the corruption share one rng — mirrors `HOISegments` but keeps determinism in our hands.

Fit the normalizer:  python -m hoi_flow.data.dataset --fit_norm [--n 300]
"""
import argparse
import glob
import json
import os
import zipfile

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

from ..state import D as STATE_D, Normalizer, pack_state
from . import corrupt as C

SEG_DIR = "/workspace/datasets/hot3d/hoi_segments"
NORM_PATH = os.path.join(os.path.dirname(__file__), "norm.json")
VAL_PARTICIPANTS = ("P0002", "P0015")   # held-out val participants
_ALL_MODALITIES = ("rgb", "depth", "seg_mask", "obj_points", "K")
_COND_PER_FRAME = ("rgb", "depth", "seg_mask")


def _zip_keys(path):
    with zipfile.ZipFile(path) as zf:
        return set(n.rsplit(".", 1)[0] for n in zf.namelist())


def _participant(path):
    with zipfile.ZipFile(path) as zf, zf.open("meta.npy") as mf:
        return json.loads(str(np.lib.format.read_array(mf, allow_pickle=True)))["participant_id"]


class HOIFlowDataset(Dataset):
    def __init__(self, seg_dir=SEG_DIR, seq_len=32,
                 modalities=("rgb", "depth", "seg_mask", "obj_points", "K"),
                 split="train", require_all_modalities=False,
                 normalizer=NORM_PATH, seed=0,
                 obj_profile="pooled", hand_profile="pooled",
                 theta_sigma=0.05, wrist_rot_deg=4.0, corrupt_scale=(1.0, 1.0)):
        files = sorted(glob.glob(os.path.join(seg_dir, "seg_*.npz")))
        if not files:
            raise FileNotFoundError(f"no seg_*.npz under {seg_dir}")
        if split in ("train", "val"):
            want_val = split == "val"
            files = [f for f in files if (_participant(f) in VAL_PARTICIPANTS) == want_val]
        need_depth = require_all_modalities and (
            "depth" in modalities or "seg_mask" in modalities)
        if need_depth:
            files = [f for f in files if {"depth_mm", "seg_mask"} <= _zip_keys(f)]
        if not files:
            raise RuntimeError(f"no segments left after split={split} require_all={require_all_modalities}")
        self.files = files
        self.seq_len = seq_len
        self.modalities = tuple(modalities)
        self.seed = int(seed)
        self.epoch = 0
        self.theta_sigma = theta_sigma
        self.wrist_rot_deg = wrist_rot_deg
        self.corrupt_scale = tuple(corrupt_scale)
        # calibrated corruption profiles (deep-copied dicts, cheap)
        stats = C.load_stats()
        self.obj_prof = C.object_profile(stats, obj_profile)
        self.hand_prof = C.hand_profile(stats, hand_profile)
        # normalizer: path -> load; Normalizer -> use; None -> identity (raw states, for --fit_norm)
        if isinstance(normalizer, Normalizer):
            self.norm = normalizer
        elif normalizer and os.path.exists(normalizer):
            self.norm = Normalizer.load(normalizer)
        else:
            self.norm = Normalizer()  # identity

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.files)

    def _rng(self, idx):
        return np.random.default_rng([self.seed, self.epoch, int(idx)])

    # -- raw window read (mmap, no decode) -------------------------------------------------
    def _read_window(self, idx, rng):
        z = np.load(self.files[idx], allow_pickle=True, mmap_mode="r")
        keys = set(z.files)
        T = z["obj_pose"].shape[0]
        L = self.seq_len
        if T >= L:
            s = int(rng.integers(0, T - L + 1)); e = s + L
        else:
            s, e = 0, T
        pad = L - (e - s)

        def take(arr):
            a = np.asarray(arr[s:e])
            if pad:  # repeat last frame
                a = np.concatenate([a, np.repeat(a[-1:], pad, axis=0)], axis=0)
            return a

        w = {
            "obj_pose": take(z["obj_pose"]).astype(np.float64),          # [L,4,4]
            "hand_wrist": take(z["hand_wrist"]).astype(np.float64),      # [L,2,4,4]
            "hand_theta": take(z["hand_mano"])[:, :, :15].astype(np.float64),  # [L,2,15]
            "hands_present": take(z["hands_present"]).astype(bool),      # [L,2]
            "obj_verts": np.asarray(z["obj_verts"]).astype(np.float32),  # [Nv,3]
            "K": np.asarray(z["K"]).astype(np.float32),                  # [3,3]
        }
        if "rgb" in self.modalities:
            w["rgb"] = take(z["rgb"])                                    # [L,R,R,3] u8
        if "depth" in self.modalities and "depth_mm" in keys:
            w["depth"] = take(z["depth_mm"]).astype(np.float32) / 1000.0  # [L,R,R] m
        if "seg_mask" in self.modalities and "seg_mask" in keys:
            w["seg_mask"] = take(z["seg_mask"]).astype(np.int64)        # [L,R,R]
        return w

    # -- GT + coarse packing ---------------------------------------------------------------
    def _pack_gt(self, w):
        x1, valid = pack_state(
            torch.from_numpy(w["obj_pose"]).float(),
            torch.from_numpy(w["hand_wrist"]).float(),
            torch.from_numpy(w["hand_theta"]).float(),
        )
        return x1, valid

    def _corrupt(self, w, rng):
        # Per-sample corruption magnitude s ~ U(corrupt_scale): the real coarse stack's error
        # spans near-perfect (benchmark in-hand clips, ~3 mm) to the full calibrated regime;
        # training only at s=1 taught the model to "fix" already-good sources (Tier-2 regression).
        lo, hi = self.corrupt_scale
        s = float(rng.uniform(lo, hi)) if hi > lo else float(lo)
        oprof = _scale_prof(self.obj_prof, s) if s != 1.0 else self.obj_prof
        hprof = _scale_prof(self.hand_prof, s) if s != 1.0 else self.hand_prof
        obj_c = C.corrupt_object(w["obj_pose"], oprof, rng)             # [L,4,4]
        L = w["hand_wrist"].shape[0]
        wrist_c = np.full((L, 2, 4, 4), np.nan)
        theta_c = np.full((L, 2, 15), np.nan)
        for h in (0, 1):
            wc, tc = _corrupt_hand_se3(
                w["hand_wrist"][:, h], w["hand_theta"][:, h], hprof, rng,
                self.theta_sigma * s, self.wrist_rot_deg * s)
            wrist_c[:, h], theta_c[:, h] = wc, tc
        x0, _ = pack_state(
            torch.from_numpy(obj_c).float(),
            torch.from_numpy(wrist_c).float(),
            torch.from_numpy(theta_c).float(),
        )
        return x0

    def gt_state(self, idx):
        """Raw (un-normalized) GT state + valid mask — used to fit the normalizer."""
        w = self._read_window(idx, self._rng(idx))
        return self._pack_gt(w)

    def __getitem__(self, idx):
        rng = self._rng(idx)
        w = self._read_window(idx, rng)
        x1, valid = self._pack_gt(w)
        x0 = self._corrupt(w, rng)
        x1n = self.norm.transform(x1)
        x0n = torch.nan_to_num(self.norm.transform(x0))

        present = torch.from_numpy(w["hands_present"])                  # [L,2]
        # keep presence consistent with the packed wrist validity
        wrist_valid = torch.stack([valid[:, 9], valid[:, 33]], dim=-1)  # [L,2]
        presence = present & wrist_valid

        cond = {}
        if "rgb" in w:
            cond["rgb"] = torch.from_numpy(w["rgb"]).permute(0, 3, 1, 2).float() / 255.0  # [L,3,R,R]
        if "depth" in w:
            cond["depth"] = torch.from_numpy(w["depth"])               # [L,R,R] m
        if "seg_mask" in w:
            cond["seg_mask"] = torch.from_numpy(w["seg_mask"])          # [L,R,R] long
        if "obj_points" in self.modalities:
            cond["obj_points"] = torch.from_numpy(w["obj_verts"])      # [Nv,3]
        if "K" in self.modalities:
            cond["K"] = torch.from_numpy(w["K"])                       # [3,3]

        return {"x1": x1n, "x0": x0n, "valid": valid, "presence": presence, "cond": cond}

    @staticmethod
    def collate(batch):
        out = {
            "x1": torch.stack([b["x1"] for b in batch]),
            "x0": torch.stack([b["x0"] for b in batch]),
            "valid": torch.stack([b["valid"] for b in batch]),
            "presence": torch.stack([b["presence"] for b in batch]),
        }
        cond_keys = batch[0]["cond"].keys()
        out["cond"] = {k: torch.stack([b["cond"][k] for b in batch]) for k in cond_keys}
        return out


_MAG_KEYS = ("bias", "sigma", "sigma_ar", "sigma_white", "jitter", "event_rate", "state_frac")


def _scale_prof(prof, s):
    """Scale a corruption profile's magnitude entries by s (rho/durations untouched)."""
    return {k: _scale_prof(v, s) if isinstance(v, dict)
            else (v * s if k in _MAG_KEYS and isinstance(v, (int, float)) else v)
            for k, v in prof.items()}


def _corrupt_hand_se3(wrist_se3, thetas, hprof, rng, theta_sigma, wrist_rot_deg):
    """Corrupt one hand's (wrist SE3 [L,4,4], thetas [L,15]) via corrupt.py's MANO path.

    SE3 -> (rotvec, transl) 6-vector -> corrupt_hand (along-ray AR + rot wobble + theta
    noise) -> back to SE3. Absent frames (NaN GT) are filled for scipy, corrupted, then
    re-NaN'd so pack_state masks them. view/ray dir defaults to the wrist camera position.
    """
    L = wrist_se3.shape[0]
    finite = (np.isfinite(wrist_se3[:, :3, :4]).reshape(L, -1).all(1)
              & np.isfinite(thetas).all(1))
    wrist_c = wrist_se3.copy()
    theta_c = thetas.copy()
    if finite.any():
        R = wrist_se3[:, :3, :3].copy()
        t = wrist_se3[:, :3, 3].copy()
        badR = ~np.isfinite(R).reshape(L, -1).all(1)
        R[badR] = np.eye(3)
        t[~np.isfinite(t).all(1)] = 0.0
        w6 = np.concatenate([Rotation.from_matrix(R).as_rotvec(), t], axis=1)
        out = C.corrupt_hand({"wrist_xform": w6, "theta": np.nan_to_num(thetas)}, hprof, rng,
                             theta_sigma=theta_sigma, wrist_rot_deg=wrist_rot_deg)
        w6c, thc = out["wrist_xform"], out["theta"]
        wrist_c[:] = np.eye(4)
        wrist_c[:, :3, :3] = Rotation.from_rotvec(w6c[:, :3]).as_matrix()
        wrist_c[:, :3, 3] = w6c[:, 3:6]
        theta_c = thc
    wrist_c[~finite] = np.nan
    theta_c[~finite] = np.nan
    return wrist_c, theta_c


def fit_normalizer(n=300, seg_dir=SEG_DIR, out=NORM_PATH, seq_len=32, seed=0):
    ds = HOIFlowDataset(seg_dir=seg_dir, seq_len=seq_len, split="train",
                        modalities=(), normalizer=None, seed=seed)
    n = min(n, len(ds))
    idxs = np.random.default_rng(seed).choice(len(ds), size=n, replace=False)
    xs, vs = [], []
    for k, i in enumerate(idxs):
        x1, valid = ds.gt_state(int(i))
        xs.append(x1); vs.append(valid)
        if (k + 1) % 50 == 0:
            print(f"  fit_norm: {k + 1}/{n} segments", flush=True)
    norm = Normalizer().fit(xs, vs)
    norm.save(out)
    mean = norm.mean.numpy(); std = norm.std.numpy()
    print(f"fit on {n} train segments ({sum(x.shape[0] for x in xs)} frames) -> {out}")
    print(f"  obj_trans mean(mm) {np.round(mean[6:9] * 1000, 1)}  std(mm) {np.round(std[6:9] * 1000, 1)}")
    print(f"  L wrist_trans std(mm) {np.round(std[15:18] * 1000, 1)}  theta std {np.round(std[18:21], 3)} ...")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit_norm", action="store_true")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seg_dir", default=SEG_DIR)
    ap.add_argument("--out", default=NORM_PATH)
    ap.add_argument("--seq_len", type=int, default=32)
    args = ap.parse_args()
    if args.fit_norm:
        fit_normalizer(n=args.n, seg_dir=args.seg_dir, out=args.out, seq_len=args.seq_len)
    else:
        ap.error("nothing to do; pass --fit_norm")
