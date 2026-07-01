#!/usr/bin/env bash
# Launch the EgoAERO 4D-HOI viser viewer.
#
# Usage:
#   scripts/view_demo.sh                       # mock pipeline run (generates one if missing)
#   scripts/view_demo.sh <run_dir> [stage]     # any run dir / stage bundle
#   scripts/view_demo.sh viz_output/wild6_viser_scene.npz   # the real wild6 reconstruction
#
# viser serves an interactive browser app on PORT (default 8080). To reach it from
# your laptop, SSH-forward the port (bypasses auth, nothing exposed publicly):
#   ssh -p $VAST_TCP_PORT_22 -L 8080:127.0.0.1:8080 root@$PUBLIC_IPADDR
# then open http://localhost:8080
set -euo pipefail
cd "$(dirname "$0")/.."

# env with viser (the base venv has no numpy; forehoi has viser+numpy+egoaero)
PY="${EGOAERO_PY:-/workspace/miniconda3/envs/forehoi/bin/python}"
RUN="${1:-runs/demo}"
STAGE="${2:-stage6_contact}"
PORT="${PORT:-8080}"

# generate a mock run if the default target is missing
if [ "$RUN" = "runs/demo" ] && [ ! -d "runs/demo/stage6_contact" ]; then
  echo "no run at runs/demo — generating a mock one ..."
  "$PY" -m egoaero.cli --out runs/demo --mock --num-frames 32
fi

echo "launching viser viewer:  run=$RUN  stage=$STAGE  port=$PORT"
"$PY" -m egoaero.viz.viser_app --run "$RUN" --stage "$STAGE" --port "$PORT"
