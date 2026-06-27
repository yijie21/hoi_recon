"""Real model adapters.

Each adapter wraps a third-party model behind a stable function signature used by
the corresponding stage. Adapters import their heavy dependency lazily and raise
`BackendNotAvailable` (with a concrete setup hint) when the repo or weights are
missing — so `--real` runs fail loudly and informatively rather than crash deep
inside a third-party module.

Wiring a backend = filling in the body of its adapter and pointing it at the cloned
repo in third_party/ and the weights in checkpoints/. The mock path in each stage
documents the exact output contract the adapter must satisfy.
"""
from __future__ import annotations

import os


class BackendNotAvailable(RuntimeError):
    pass


def require_repo(third_party_dir: str, name: str, clone_hint: str) -> str:
    path = os.path.join(third_party_dir, name)
    if not os.path.isdir(path):
        raise BackendNotAvailable(
            f"third-party repo '{name}' not found at {path}.\n"
            f"  -> run: bash scripts/setup_third_party.sh   (or: {clone_hint})"
        )
    return path


def require_ckpt(ckpt_dir: str, rel: str, download_hint: str) -> str:
    path = os.path.join(ckpt_dir, rel)
    if not os.path.exists(path):
        raise BackendNotAvailable(
            f"checkpoint '{rel}' not found at {path}.\n"
            f"  -> {download_hint}\n"
            f"  -> see scripts/download_checkpoints.sh"
        )
    return path
