"""Command-line entry point.

Examples:
  # mock mode (runs now, no weights):
  python -m hoi_recon.cli --out runs/demo --mock --num-frames 48

  # real mode (needs third_party/ + checkpoints/):
  python -m hoi_recon.cli --video clip.mp4 --out runs/clip01 --real \
      --hand hamer --object sam3d --depth moge

  # run / resume a subset of stages:
  python -m hoi_recon.cli --out runs/demo --mock --stages 5-8 --force
"""
from __future__ import annotations

import argparse
import sys

from .config import load_config
from .logging_utils import log
from .pipeline import run_pipeline


def build_parser():
    p = argparse.ArgumentParser("hoi-recon", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="run output directory")
    p.add_argument("--config", default=None, help="yaml config (overrides defaults)")
    p.add_argument("--video", default=None, help="input RGB video (real mode)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--mock", dest="mock", action="store_true", help="synthetic scene (default)")
    mode.add_argument("--real", dest="mock", action="store_false", help="use real backends")
    p.set_defaults(mock=None)
    p.add_argument("--num-frames", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--stages", default="all", help="'all' | '0-7' | '2' | '0,2,4' | '5-'")
    p.add_argument("--force", action="store_true", help="recompute cached stages")
    # backend overrides
    p.add_argument("--hand", default=None,
                   choices=["hamer", "wilor", "hawor", "depthlift"])
    p.add_argument("--object", default=None, choices=["sam3d", "bundlesdf", "foundationpose"])
    p.add_argument("--depth", default=None,
                   choices=["moge", "da3", "depth_anything_v2", "metric3d"])
    p.add_argument("--camera", default=None, choices=["vipe", "droid_slam"])
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    overrides = {
        "video": args.video,
        "num_frames": args.num_frames,
        "seed": args.seed,
        "force": args.force or None,
        "backend": {"hand": args.hand, "object": args.object,
                    "depth": args.depth, "camera": args.camera},
    }
    if args.mock is not None:
        overrides["mock"] = args.mock
    cfg = load_config(args.config, overrides)
    if not cfg.mock and not cfg.video:
        log("real mode requires --video", "err")
        return 2
    try:
        run_pipeline(cfg, args.out, args.stages)
    except Exception as e:                       # surface backend wiring errors cleanly
        log(f"{type(e).__name__}: {e}", "err")
        return 1
    log("done.", "ok")
    log(f"view the 4D interaction:  hoi-recon-view --run {args.out}", "info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
