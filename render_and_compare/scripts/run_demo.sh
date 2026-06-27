#!/usr/bin/env bash
# End-to-end mock run (no weights needed) + print the error-attribution report.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m hoi_recon.cli --out runs/demo --mock --num-frames "${1:-48}" --force
echo
echo "report: runs/demo/stage8_eval/report.json"
echo "pseudo-GT for the feed-forward model: runs/demo/stage8_eval/pseudo_gt.npz"
