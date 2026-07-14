# Building clean single-object HOI clips from HOT3D Aria

How to regenerate the **clean single-object hand-object interaction** dataset used for
training (e.g. a flow-matching HOI generation network). Each "clip" is a segment of a
HOT3D Aria recording in which **exactly one object** is being manipulated by the hand(s).

The output is **not** copied video — it's a *manifest of pointers* plus compact
per-segment tensors, so the dataloader is fast (memory-mapped, no JPEG decode, no tar
access at train time). Names/terms are decoded in [`../../GLOSSARY.md`](../../GLOSSARY.md).

## What you get

- **`/workspace/datasets/hot3d/hoi_clean_manifest.json`** — one record per segment:
  `{window_id, source_clip, frame_start, frame_end, target_uid, object_name, num_hands,
  hands, motion, visibility, sequence_id, participant_id}`.
- **`/workspace/datasets/hot3d/hoi_segments/seg_<window_id>.npz`** — the training tensor
  for that segment (see [spec](#per-segment-tensor-spec)).

Last build: **1,618 segments** over 1,911 Aria clips (benchmark sequences held out), 31 GB.

## Prerequisites

1. **Env `rc5090`** (has `hand_tracking_toolkit`, `trimesh`, `scipy`, `torch`, `hf`).
2. **The Aria half of HOT3D** at `/workspace/datasets/hot3d/` — clip tars + object models +
   clip metadata. Download only what the build reads (reliable `snapshot_download`; the
   `hf download --include` CLI mis-parses multiple patterns):

   ```python
   # env rc5090; HF_HOME=/workspace/huggingface_cache/ with access to bop-benchmark/hot3d
   from huggingface_hub import snapshot_download
   snapshot_download("bop-benchmark/hot3d", repo_type="dataset",
       local_dir="/workspace/datasets/hot3d",
       allow_patterns=["train_aria/*", "test_aria/*", "object_models_eval/*",
                        "clip_definitions.json"])
   ```
   (~250 GB. The build reads the clip **tars directly** — it does not need them extracted.
   Quest3 is **not** used: its cameras are grayscale, no RGB — see [notes](#notes--caveats).)

## Reproduce (one command)

```bash
cd compare/hot3d
python build_hoi_dataset.py          # env rc5090; ~1 h; resumable; disk stays flat
#   --seg_dir /workspace/datasets/hot3d/hoi_segments   (default)
#   --manifest /workspace/datasets/hot3d/hoi_clean_manifest.json
#   --res 256 --fov 90 --nverts 2000  --limit N (for a quick test)
```

It runs one pass over every Aria clip (minus the held-out benchmark sequences): extract
its tar → detect clean windows → blocklist Aria-device objects → precompute per-segment
tensors → delete the temp dir. Progress is written every 25 clips to
`<manifest>.progress.json`; re-running resumes from there.

## The segmentation rules (what makes a segment "clean")

Decided deliberately (see the design interview in the commit history). Per frame, for each
object O: `near(O)` = min 3D distance from UmeTrack hand vertices to the posed object mesh
< 2 cm; `moving(O)` = per-frame pose delta above a floor; `manipulated(O)` = near ∧ moving.

A **window** for target object O is formed by:
1. **Continuity = in-hand.** A maximal run where O is `near` a hand, **bridging gaps ≤ 0.4 s**
   (regrips / brief holds-still — object motion is bursty, so continuity uses *holding*, not motion).
2. **Strict single-object.** Cut the run at any frame where a *different* object is
   `manipulated`. (A second object merely sitting near the hand does **not** cut it.)
3. **Keep only if:** duration ≥ 1 s **and** the window has enough motion (cumulative
   translation **or** rotation range above a floor) **and** it passes a low visibility
   pre-filter (≥1 frame with target visibility > 0.7, median > 0.25).
4. **Bimanual is allowed** on the one object (hand-count stored). Aria-device objects
   (`aria_small`/`aria_large`) are blocklisted.

Thresholds live in `build_hoi_dataset.py` (`TH`) — loosen `min_len` / `win_trans` / `win_rot`
/ `vis_*` for more (noisier) segments; tighten for fewer (cleaner) ones.

## Per-segment tensor spec

`seg_<window_id>.npz` (T = segment length in frames, at 30 fps):

| key | shape | meaning |
|---|---|---|
| `rgb` | `[T,256,256,3]` uint8 | fisheye → **upright pinhole** (90° FOV), resized to 256² |
| `K` | `[3,3]` | pinhole intrinsics for `rgb` (a point projects from a pose via `K`) |
| `obj_pose` | `[T,4,4]` | GT target object pose in the **pinhole camera frame** |
| `obj_verts` | `[2000,3]` | target canonical mesh points (meters) — shape conditioning |
| `obj_uid` | scalar | HOT3D object id |
| `hand_mano` | `[T,2,21]` | GT MANO (thetas[15]+wrist[6]); rows [left,right]; NaN if absent |
| `hand_wrist` | `[T,2,4,4]` | UmeTrack wrist pose, camera frame |
| `hand_joints` | `[T,2,20,3]` | UmeTrack 20 landmarks, camera frame (license-free) |
| `hands_present` | `[T,2]` bool | which hands are present |
| `meta` | json str | window_id, source_clip, sequence_id, participant_id, frames, object_name, fps, fov, res |

All GT is harvested from the HOT3D annotations (no reconstruction pipeline needed).

## Use it (PyTorch dataloader)

```python
from hoi_dataset import HOISegments        # compare/hot3d/hoi_dataset.py
from torch.utils.data import DataLoader
ds = HOISegments("/workspace/datasets/hot3d/hoi_segments", seq_len=16)
dl = DataLoader(ds, batch_size=16, num_workers=8, shuffle=True, collate_fn=HOISegments.collate)
for b in dl:
    b["rgb"]        # [16, 16, 3, 256, 256] float in [0,1]
    b["obj_pose"]   # [16, 16, 4, 4]
    b["hand_joints"]# [16, 16, 2, 20, 3]
```
Decode-free, mmap, ~125 samples/s (~2k frames/s, 8 workers). Leakage-safe train/val split
by participant: `HOISegments(seg_dir, split={"val": ["P0003","P0010"], "which": "train"})`.

## Validate (eyeball a sample)

```bash
# manifest on a small subset (a JSON list of clip ids, or clip dirs)
python gen_clean_clips.py <subset.json> --out clean_subset.json
# montage PNGs: target=green, other objects=gray, hands splatted
python render_clean_overlay.py clean_subset.json --n 20 --out_dir overlays/clean
# playable mp4s (raw clip + mesh overlay)
python materialize_clips.py clean_subset.json --n 20 --overlay --out_dir overlays/clean/vids
```

## Scripts

| file | role |
|---|---|
| `build_hoi_dataset.py` | **the one-command build** (orchestrates the two below per clip) |
| `gen_clean_clips.py` | detect clean windows → manifest (+ distribution stats) |
| `precompute_segments.py` | manifest window → `seg_*.npz` (rectify + harvest GT) |
| `hoi_dataset.py` | the PyTorch `Dataset` / `DataLoader` |
| `render_clean_overlay.py`, `materialize_clips.py` | validation montages / videos |

## Notes & caveats

- **Fisheye → pinhole.** HOT3D Aria RGB is fisheye624; there is no native pinhole. We rectify
  to an upright pinhole (reusing `make_rc_input.py`'s remap) so `rgb`, `K`, `obj_pose` are
  mutually consistent — verified by projecting the object mesh onto its pixels.
- **Quest3 is unusable for RGB.** HOT3D's other device (Quest3) records **grayscale** only
  (streams `1201-1/2`, no color camera). It adds interactions but not RGB — Aria is the only
  RGB modality. So ~1,600 Aria segments is HOT3D's RGB ceiling short of loosening thresholds.
- **Leakage holdout.** The 5 sequences behind the 6 mocap benchmark clips are excluded from
  the build, so this training data never overlaps the evaluation set.
- **More RGB data** would come from a different RGB HOI dataset (DexYCB / HO-Cap / ARCTIC) via
  a `make_rc_input`-style adapter into the same manifest → segment pipeline.
