"""EgoAERO SP4 — egoaero-collect command-line entry point.

Usage:
    python -m egoaero.dataset.cli --out <dir> --n <K> [--max-attempts M] [--seed S]
    egoaero-collect --out <dir> --n <K> [--max-attempts M] [--seed S]
"""
import argparse, json, os, sys, yaml

from .collect import run_collection


def _dataset_cfg():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "configs", "dataset.yaml")) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser(
        prog=os.path.basename(sys.argv[0]),
        description="EgoAERO SP4 — EgoDex-R closed-loop collection",
    )
    ap.add_argument("--out", required=True,
                    help="Output directory for accepted sequences and summary.json")
    ap.add_argument("--n", type=int, default=None,
                    help="Number of accepted sequences to collect (overrides dataset.yaml)")
    ap.add_argument("--max-attempts", type=int, default=None,
                    help="Max capture attempts (overrides dataset.yaml)")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed for synthetic source generation (default: 0)")
    a = ap.parse_args()

    cfg = _dataset_cfg()
    n_target = a.n if a.n is not None else cfg["collection"]["n_target"]
    if a.max_attempts is not None:
        cfg["collection"]["max_attempts"] = a.max_attempts

    summary = run_collection(a.out, n_target, cfg, seed=a.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
