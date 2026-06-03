#!/usr/bin/env bash
# Run the mock pipeline (if needed) and open the 4D HOI viewer.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${1:-runs/demo}"
STAGE="${2:-stage7_contact_optim}"
if [ ! -f "$RUN/stage7_contact_optim/meta.json" ]; then
  echo "no run found at $RUN — generating one..."
  python -m hoi_recon.cli --out "$RUN" --mock
fi
echo "launching viser viewer for $RUN ($STAGE) ..."
python -m hoi_recon.viz.viser_app --run "$RUN" --stage "$STAGE"
