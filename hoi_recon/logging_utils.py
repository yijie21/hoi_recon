"""Minimal colored logger shared across stages."""
from __future__ import annotations

import sys
import time

_START = time.time()


def _c(code: str, s: str) -> str:
    if not sys.stderr.isatty():
        return s
    return f"\033[{code}m{s}\033[0m"


def log(msg: str, level: str = "info") -> None:
    t = time.time() - _START
    tag = {
        "info": _c("36", "INFO"),
        "stage": _c("1;35", "STAGE"),
        "ok": _c("32", "OK"),
        "warn": _c("33", "WARN"),
        "err": _c("31", "ERR"),
    }.get(level, "INFO")
    print(f"[{t:7.2f}s] {tag} {msg}", file=sys.stderr, flush=True)


def stage_banner(idx: int, name: str, mode: str) -> None:
    log(f"── stage {idx}: {name}  [{mode}]", "stage")
