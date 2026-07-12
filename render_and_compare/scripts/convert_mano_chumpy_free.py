"""Convert MANO_{RIGHT,LEFT}.pkl to chumpy-free `_np.pkl` copies.

The official MANO pickles store some arrays (e.g. `shapedirs`) as chumpy objects, so
`pickle.load` needs `chumpy` importable. The hand-reprojection optimizer (joint_opt.py)
runs in the SAM-3D env (`sam3d5090`), which has **no chumpy**, so it cannot load the raw
pickle. This one-shot converter (run in an env that DOES have chumpy — e.g. `rc5090`)
resolves every chumpy array to plain numpy and writes `MANO_{RIGHT,LEFT}_np.pkl` next to
the originals. `smplx.MANOLayer(model_path=<...>_np.pkl)` then loads without chumpy.

Idempotent: skips a hand whose `_np.pkl` already exists (unless --force).

  python convert_mano_chumpy_free.py [<mano_dir=checkpoints/mano>] [--force]
"""
import argparse
import os
import pickle
import sys

import numpy as np

# restore the numpy aliases chumpy still imports (removed in numpy>=1.24 / numpy 2)
for _n, _t in [("bool", bool), ("int", int), ("float", float), ("complex", complex),
               ("object", object), ("str", str), ("unicode", str)]:
    if not hasattr(np, _n):
        setattr(np, _n, _t)


def convert(mano_dir, force=False):
    try:
        import chumpy  # noqa: F401  (needed so pickle can resolve chumpy.Ch classes)
    except ImportError:
        sys.exit("convert_mano_chumpy_free: `chumpy` not importable in this env — run it in "
                 "one that has chumpy (e.g. rc5090: `pip install --no-build-isolation chumpy`).")
    made = []
    for hand in ("RIGHT", "LEFT"):
        src = os.path.join(mano_dir, f"MANO_{hand}.pkl")
        dst = os.path.join(mano_dir, f"MANO_{hand}_np.pkl")
        if not os.path.exists(src):
            continue
        if os.path.exists(dst) and not force:
            made.append(f"{hand}(exists)")
            continue
        d = pickle.load(open(src, "rb"), encoding="latin1")
        out = {k: (np.array(v.r) if hasattr(v, "r") else v) for k, v in d.items()}
        pickle.dump(out, open(dst, "wb"))
        made.append(f"{hand}->{os.path.basename(dst)}")
    return made


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("mano_dir", nargs="?",
                    default=os.path.join(os.path.dirname(here), "checkpoints", "mano"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    print("MANO chumpy-free:", convert(a.mano_dir, a.force) or "(no MANO_*.pkl found)")
