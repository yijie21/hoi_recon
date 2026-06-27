# egoaero/egoaero/cli.py
import argparse
import os

from .config import load_config
from .pipeline import run_pipeline


def main():
    ap = argparse.ArgumentParser("egoaero")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--video", default=None)
    ap.add_argument("--num-frames", type=int, default=None)
    ap.add_argument("--stages", default="all")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    over = {
        "mock": True if a.mock else None,
        "video": a.video,
        "num_frames": a.num_frames,
        "force": a.force or None,
    }
    cfg = load_config(a.config, {k: v for k, v in over.items() if v is not None})
    ctx = run_pipeline(cfg, a.out, a.stages)
    print("done:", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
