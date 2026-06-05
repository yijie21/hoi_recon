"""Standalone CLI for the reprojection-overlay validation videos.

The pipeline generates these automatically at the end of a real run (see
hoi_recon.viz.reproject); this script re-renders them on demand.

  python scripts/reproject_overlay.py --run runs/grab
"""
import argparse
import os

from hoi_recon.viz.reproject import generate_overlays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/grab")
    ap.add_argument("--stage", default="stage7_contact_optim")
    ap.add_argument("--fps", type=float, default=24.0)
    a = ap.parse_args()
    paths = generate_overlays(a.run, a.stage, a.fps)
    if paths:
        print("wrote:", *[os.path.basename(p) for p in paths], "under", a.run)
    else:
        print(f"no frames / stage in {a.run} (real-mode run required)")


if __name__ == "__main__":
    main()
