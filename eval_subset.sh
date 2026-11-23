#!/usr/bin/env bash
# Quick full-length eval of one policy on 3 hard scenarios (cold / hot / mixed), run in parallel,
# then print its total reward next to the shipped baselines. Higher (less negative) is better.
# Usage: ./eval_subset.sh <policy> [results_dir]
set -euo pipefail
cd "$(dirname "$0")"
POLICY=$1
OUT=${2:-results/subset_$POLICY}
SCEN="building_train_1:fairbanks_1 building_train_1:phoenix_5 building_train_2:denver_8"
mkdir -p "$OUT"
for s in $SCEN; do
  b=${s%%:*}; w=${s##*:}
  uv run python -m smart_buildings.evaluate --policies "$POLICY" --buildings "$b" --weathers "$w" \
    --results-dir "$OUT" > "$OUT/$b.$w.log" 2>&1 &
done
wait
uv run python - "$OUT" "$POLICY" <<'EOF'
import json, sys
from pathlib import Path
out, pol = Path(sys.argv[1]), sys.argv[2]
# baseline_returns.json does NOT match this machine's simulator (fairbanks_1 pid_water: -1829 shipped vs
# -2607 local), so compare against local pid_water runs (results/pid_water_ref = pid_water, run under the old transfer key) instead.
for f in sorted(out.glob(f"*/*/{pol}/summary.json")):
    key = f"{f.parts[-4]}/{f.parts[-3]}"
    mine = json.load(open(f))["total_reward"]
    ref = Path(f"results/pid_water_ref/{key}/transfer/summary.json")
    ref_s = f"{json.load(open(ref))['total_reward']:9.1f}" if ref.exists() else "  pending"
    print(f"{key:32s} {pol}={mine:9.1f}  local_pid_water={ref_s}")
EOF
