"""Stage orchestration: caching, resumability, selective stage execution."""
from __future__ import annotations

import os

from .bundle import Bundle
from .config import Config, save_config
from .stages import (  # noqa: F401
    stage0_ego_io,
    stage1_semantic,
    stage2_track,
    stage3_mesh,
    stage4_hand,
    stage5_ego_comp,
    stage6_contact,
    stage7_eval,
    stage8_quality,
)


class RunContext:
    """Shared state handed to every stage."""

    def __init__(self, cfg: Config, run_dir: str):
        self.cfg = cfg
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)

    def stage_dir(self, name: str) -> str:
        return os.path.join(self.run_dir, name)

    def has(self, name: str) -> bool:
        return Bundle.exists(self.stage_dir(name))

    def load(self, name: str) -> Bundle:
        if not self.has(name):
            raise FileNotFoundError(
                f"stage '{name}' output missing — run that stage first "
                f"(looked in {self.stage_dir(name)})")
        return Bundle.load(self.stage_dir(name))


# ---------------------------------------------------------------------------
# Stage registry
# ---------------------------------------------------------------------------

STAGES = [
    stage0_ego_io,
    stage1_semantic,
    stage2_track,
    stage3_mesh,
    stage4_hand,
    stage5_ego_comp,
    stage6_contact,
    stage7_eval,
    stage8_quality,
]


def _selected(stages_arg, n: int) -> list:
    """Parse a stages selector into a sorted list of stage indices.

    Accepts:
      - None / "all"           → all indices 0..n-1
      - "0,2,5"                → [0, 2, 5]
      - "0-3"                  → [0, 1, 2, 3]
      - "0-3,6"                → [0, 1, 2, 3, 6]
      - "2-"                   → [2, 3, ..., n-1]
      - "-4"                   → [0, 1, 2, 3, 4]
    """
    if stages_arg in (None, "all"):
        return list(range(n))
    out = []
    for part in str(stages_arg).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            a = int(a) if a else 0
            b = int(b) if b else n - 1
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    return sorted(set(i for i in out if 0 <= i < n))


def run_pipeline(cfg: Config, run_dir: str, stages: str = "all") -> RunContext:
    """Run selected stages end-to-end with caching (skip if output exists and not force)."""
    ctx = RunContext(cfg, run_dir)
    save_config(cfg, os.path.join(run_dir, "config.yaml"))
    for i in _selected(stages, len(STAGES)):
        mod = STAGES[i]
        if ctx.has(mod.NAME) and not cfg.force:
            continue
        out = mod.run(ctx)
        out.save(ctx.stage_dir(mod.NAME))
    # Auto-write workbench contract when stage7 was selected and stage6 output exists
    if 7 in _selected(stages, len(STAGES)) and ctx.has("stage6_contact"):
        from .contract import write as _cw
        _cw(ctx)
    return ctx
