"""Synthetic capture source — a documented stand-in for FastUMI-Ego capture. Yields
mock ego-clip configs whose grasp tightness sweeps a range, so the collection loop
sees a genuine spread of reconstruction quality (and thus accept/repair/recapture
decisions) WITHOUT changing the SP3 thresholds."""
from __future__ import annotations

_TASKS = ["pick up the object", "move the object", "place the object",
          "lift and hold the object", "grasp and relocate the object"]


def synthetic_source(n, seed, num_frames, tightness_min, tightness_max):
    n = int(n)
    clips = []
    for i in range(n):
        frac = i / (n - 1) if n > 1 else 1.0
        tightness = tightness_min + frac * (tightness_max - tightness_min)
        clips.append({
            "seed": int(seed) + i,
            "num_frames": int(num_frames),
            "mock_tightness": float(tightness),
            "task_description": _TASKS[i % len(_TASKS)],
            "manipulated_object": "object",
            "relational_objects": ["table"],
        })
    return clips


def clip_overrides(clip_cfg):
    return {"mock": True, "seed": clip_cfg["seed"],
            "num_frames": clip_cfg["num_frames"],
            "mock_tightness": clip_cfg["mock_tightness"]}
