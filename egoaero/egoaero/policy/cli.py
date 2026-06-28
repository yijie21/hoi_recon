"""CLI entry points for the two-stage residual RL policy.

egoaero-train
    Train Stage-I (wrist-tracking) and Stage-II (residual contact) PPO policies
    on a completed SP1 reconstruction run.

egoaero-eval
    Roll out saved pi_I / pi_R policies and print App-H metrics.

Both commands are registered as console scripts in pyproject.toml; the
``train`` / ``eval`` subcommand selects the action.  Heavy deps (mujoco,
stable_baselines3) are imported lazily inside the handlers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _policy_cfg() -> dict:
    """Load egoaero/configs/policy.yaml relative to this file."""
    import yaml  # lightweight; safe to import at module level but kept lazy for symmetry

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(here, "configs", "policy.yaml")
    with open(cfg_path) as fh:
        return yaml.safe_load(fh)


def _hand_xml() -> str:
    """Return the absolute path to the vendored Shadow Hand MJCF."""
    pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(pkg_root, "assets", "shadow_hand", "right_hand.xml")


def _cmd_train(args: argparse.Namespace) -> None:
    from .task import build_task
    from .train import train_two_stage

    cfg = _policy_cfg()
    task = build_task(args.run, _hand_xml(), cfg)
    train_two_stage(task, cfg, budget=args.budget, out_dir=args.out, seed=args.seed)
    print("trained ->", os.path.abspath(args.out))


def _cmd_eval(args: argparse.Namespace) -> None:
    from stable_baselines3 import PPO

    from .evaluate import evaluate
    from .task import build_task

    cfg = _policy_cfg()
    task = build_task(args.run, _hand_xml(), cfg)
    pi_I = PPO.load(os.path.join(args.policy, "pi_I"))
    pi_R = PPO.load(os.path.join(args.policy, "pi_R"))
    metrics = evaluate(task, pi_I, pi_R)
    print(json.dumps(metrics, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(
        prog=os.path.basename(sys.argv[0]),
        description="EgoAERO SP2 — two-stage residual RL policy",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # --- train subcommand ---
    t = sub.add_parser("train", help="Train pi_I (Stage-I) + pi_R (Stage-II) and save.")
    t.add_argument("--run", required=True, metavar="DIR",
                   help="Path to a completed SP1 reconstruction run directory.")
    t.add_argument("--out", required=True, metavar="DIR",
                   help="Output directory for pi_I.zip and pi_R.zip.")
    t.add_argument("--budget", default="smoke", choices=["smoke", "real"],
                   help="Training budget (smoke=512 steps/stage; real=1.5M steps/stage).")
    t.add_argument("--seed", type=int, default=0, metavar="N",
                   help="Random seed passed to SB3 PPO (default 0).")

    # --- eval subcommand ---
    e = sub.add_parser("eval", help="Evaluate a saved policy and print App-H metrics.")
    e.add_argument("--run", required=True, metavar="DIR",
                   help="Path to the SP1 reconstruction run directory used for training.")
    e.add_argument("--policy", required=True, metavar="DIR",
                   help="Directory containing pi_I.zip and pi_R.zip.")

    a = ap.parse_args()
    if a.cmd == "train":
        _cmd_train(a)
    else:
        _cmd_eval(a)


if __name__ == "__main__":
    main()
