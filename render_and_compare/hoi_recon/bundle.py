"""On-disk inter-stage IO.

Each stage writes a `Bundle` into its own directory under the run dir:
  <run>/<stage_name>/arrays.npz   numeric tensors (keyed by convention)
  <run>/<stage_name>/meta.json    small json-able scalars / strings / lists
  <run>/<stage_name>/assets.json  paths to large external files (frames, depth, meshes)

Stages read upstream bundles by name. Keys are documented in each stage docstring.
Keeping IO file-based (rather than passing python objects) makes every stage
independently runnable, resumable, and inspectable — exactly what a research /
error-attribution rig needs.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict

import numpy as np


class Bundle:
    def __init__(self, arrays: Dict[str, np.ndarray] | None = None,
                 meta: Dict[str, Any] | None = None,
                 assets: Dict[str, Any] | None = None):
        self.arrays: Dict[str, np.ndarray] = {k: np.asarray(v) for k, v in (arrays or {}).items()}
        self.meta: Dict[str, Any] = dict(meta or {})
        self.assets: Dict[str, Any] = dict(assets or {})

    # -- numeric ---------------------------------------------------------
    def __getitem__(self, k: str) -> np.ndarray:
        return self.arrays[k]

    def get(self, k: str, default=None):
        return self.arrays.get(k, default)

    def set(self, **kw) -> "Bundle":
        for k, v in kw.items():
            self.arrays[k] = np.asarray(v)
        return self

    # -- persistence -----------------------------------------------------
    def save(self, d: str) -> str:
        os.makedirs(d, exist_ok=True)
        np.savez(os.path.join(d, "arrays.npz"), **self.arrays)
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(self.meta, f, indent=2, default=_jsonable)
        with open(os.path.join(d, "assets.json"), "w") as f:
            json.dump(self.assets, f, indent=2, default=_jsonable)
        return d

    @classmethod
    def load(cls, d: str) -> "Bundle":
        arrays: Dict[str, np.ndarray] = {}
        npz = os.path.join(d, "arrays.npz")
        if os.path.exists(npz):
            with np.load(npz, allow_pickle=True) as z:
                arrays = {k: z[k] for k in z.files}
        meta = _load_json(os.path.join(d, "meta.json"))
        assets = _load_json(os.path.join(d, "assets.json"))
        return cls(arrays, meta, assets)

    @staticmethod
    def exists(d: str) -> bool:
        return os.path.exists(os.path.join(d, "meta.json"))

    def __repr__(self) -> str:
        shapes = {k: tuple(v.shape) for k, v in self.arrays.items()}
        return f"Bundle(arrays={shapes}, meta_keys={list(self.meta)}, assets={list(self.assets)})"


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _load_json(p: str) -> dict:
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)
