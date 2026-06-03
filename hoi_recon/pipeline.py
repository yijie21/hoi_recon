"""Stage orchestration: caching, resumability, selective stage execution."""
from __future__ import annotations

import os
from typing import List

from .bundle import Bundle
from .config import Config, save_config
from .logging_utils import log, stage_banner
from .stages import (
    stage0_preprocess, stage1_detect_track, stage2_hand, stage3_object,
    stage4_align, stage5_coarse_fit, stage6_rectify, stage7_contact_optim,
    stage8_eval,
)

STAGES = [
    stage0_preprocess,
    stage1_detect_track,
    stage2_hand,
    stage3_object,
    stage4_align,
    stage5_coarse_fit,
    stage6_rectify,
    stage7_contact_optim,
    stage8_eval,
]


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


def _selected(stages_arg: str, n: int) -> List[int]:
    """Parse '--stages' like 'all', '0-7', '2', '0,2,4', '3-'."""
    if stages_arg in (None, "all"):
        return list(range(n))
    out = []
    for part in str(stages_arg).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            a = int(a) if a else 0
            b = int(b) if b else n - 1
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    return sorted(set(i for i in out if 0 <= i < n))


def run_pipeline(cfg: Config, run_dir: str, stages: str = "all") -> RunContext:
    ctx = RunContext(cfg, run_dir)
    save_config(cfg, os.path.join(run_dir, "config.yaml"))
    mode = "mock" if cfg.mock else "real"
    sel = _selected(stages, len(STAGES))
    log(f"run_dir={run_dir}  mode={mode}  stages={sel}")

    for i in sel:
        mod = STAGES[i]
        name = mod.NAME
        if ctx.has(name) and not cfg.force:
            log(f"stage {i} ({name}) cached — skip (use --force to recompute)", "ok")
            continue
        stage_banner(i, name, mode)
        out: Bundle = mod.run(ctx)
        out.save(ctx.stage_dir(name))
        log(f"saved {name}: {out}", "ok")
    return ctx
