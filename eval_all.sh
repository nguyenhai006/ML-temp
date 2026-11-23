#!/usr/bin/env bash
# Full 30-scenario eval of one policy, one process per building (6 in parallel), then score.
# Usage: ./eval_all.sh [policy] [results_dir]
set -euo pipefail
cd "$(dirname "$0")"
POLICY=${1:-transfer}
OUT=${2:-results/$POLICY}
mkdir -p "$OUT"
ls smart_buildings/training_scenarios | xargs -P 6 -I{} sh -c \
  "uv run python -m smart_buildings.evaluate --policies $POLICY --buildings {} --results-dir $OUT > $OUT/{}.log 2>&1"
uv run python score_leaderboard.py "$OUT" --policy "$POLICY" --expect 30
