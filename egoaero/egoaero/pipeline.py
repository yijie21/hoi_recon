"""Stage orchestration: caching, resumability, selective stage execution."""
from __future__ import annotations

import os

from .bundle import Bundle
from .config import Config


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
