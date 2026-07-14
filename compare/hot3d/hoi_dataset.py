"""Fast decode-free PyTorch dataset for the clean HOI segments (precompute_segments.py).
Each sample is one interaction segment: pinhole RGB + GT object trajectory + GT hands +
object shape. Reads pre-decoded .npz via mmap — no JPEG decode, no tar access at train time.

  ds = HOISegments("/workspace/datasets/hot3d/hoi_segments", seq_len=30)
  dl = DataLoader(ds, batch_size=16, num_workers=8, shuffle=True, collate_fn=ds.collate)
  for b in dl: b["rgb"]  # [B, seq_len, 3, 256, 256] float in [0,1]

seq_len=None returns full variable-length segments (use collate=pad_collate).
"""
import glob, json, os
import numpy as np, torch
from torch.utils.data import Dataset


class HOISegments(Dataset):
    def __init__(self, seg_dir, seq_len=30, split=None, rng_seed=0):
        self.files = sorted(glob.glob(os.path.join(seg_dir, "seg_*.npz")))
        if split:  # split by participant to avoid leakage: e.g. {"val": ["P0003","P0010"]}
            hold = set(split.get("val", []))
            keep_val = split.get("which") == "val"
            self.files = [f for f in self.files
                          if (json.loads(str(np.load(f, allow_pickle=True)["meta"]))["participant_id"] in hold) == keep_val]
        self.seq_len = seq_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        z = np.load(self.files[i], allow_pickle=True, mmap_mode="r")
        T = z["rgb"].shape[0]
        if self.seq_len and T >= self.seq_len:
            s = np.random.randint(0, T - self.seq_len + 1); e = s + self.seq_len
        else:
            s, e = 0, T
        rgb = torch.from_numpy(np.asarray(z["rgb"][s:e])).permute(0, 3, 1, 2).float() / 255.0
        out = {
            "rgb": rgb,                                                      # [t,3,R,R]
            "obj_pose": torch.from_numpy(np.asarray(z["obj_pose"][s:e])),    # [t,4,4]
            "hand_mano": torch.from_numpy(np.asarray(z["hand_mano"][s:e])),    # [t,2,21]
            "hand_wrist": torch.from_numpy(np.asarray(z["hand_wrist"][s:e])),  # [t,2,4,4]
            "hand_joints": torch.from_numpy(np.asarray(z["hand_joints"][s:e])),# [t,2,20,3] UmeTrack, cam frame
            "hands_present": torch.from_numpy(np.asarray(z["hands_present"][s:e])),
            "obj_verts": torch.from_numpy(np.asarray(z["obj_verts"])),       # [Nv,3]
            "K": torch.from_numpy(np.asarray(z["K"])),                       # [3,3]
            "obj_uid": int(z["obj_uid"]),
        }
        if self.seq_len and (e - s) < self.seq_len:  # pad short segments
            pad = self.seq_len - (e - s)
            for k in ("rgb", "obj_pose", "hand_mano", "hand_wrist", "hand_joints", "hands_present"):
                out[k] = torch.cat([out[k], out[k][-1:].repeat((pad,) + (1,) * (out[k].ndim - 1))], 0)
        return out

    @staticmethod
    def collate(batch):  # fixed seq_len -> stack; obj_verts/uid kept as lists (variable Nv)
        keys_stack = ("rgb", "obj_pose", "hand_mano", "hand_wrist", "hand_joints", "hands_present", "K")
        out = {k: torch.stack([b[k] for b in batch]) for k in keys_stack}
        out["obj_verts"] = [b["obj_verts"] for b in batch]
        out["obj_uid"] = [b["obj_uid"] for b in batch]
        return out
