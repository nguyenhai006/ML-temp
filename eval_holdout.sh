#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 || $# > 2 )); then
    printf 'Usage: %s <policy> [results_dir]\n' "$0" >&2
    exit 2
fi

cd "$(dirname "$0")"
policy=$1
out=${2:-results/validation_$policy}
scenarios_dir=smart_buildings/validation_scenarios
steps=8640

shopt -s nullglob
configs=("$scenarios_dir"/*/*/config.gin)
if (( ${#configs[@]} == 0 )); then
    printf 'No scenarios found in %s\n' "$scenarios_dir" >&2
    exit 1
fi

mkdir -p -- "$out"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR="$out/.cache/matplotlib" NUMBA_CACHE_DIR="$out/.cache/numba"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

printf '%s\0' "${configs[@]}" | xargs -0 -n 1 -P 3 bash -c '
    policy=$1 out=$2 scenarios_dir=$3 steps=$4 config=$5
    scenario=${config#"$scenarios_dir"/}
    scenario=${scenario%/config.gin}
    building=${scenario%/*}
    weather=${scenario#*/}
    mkdir -p -- "$out/$scenario"
    log="$out/$scenario/evaluate.log"
    printf "Running %s on %s\n" "$policy" "$scenario"
    if ! uv run --no-sync python -m smart_buildings.evaluate \
        --policies "$policy" --buildings "$building" --weathers "$weather" \
        --scenarios-dir "$scenarios_dir" --results-dir "$out" --steps "$steps" \
        > "$log" 2>&1; then
        printf "Evaluation failed for %s. See %s\n" "$scenario" "$log" >&2
        exit 1
    fi
' bash "$policy" "$out" "$scenarios_dir" "$steps"

uv run --no-sync python - "$policy" "$out" "$steps" "${configs[@]}" <<'PY'
import json
import math
import sys
from pathlib import Path
from statistics import fmean

policy, results_dir, step_count, *configs = sys.argv[1:]
steps = int(step_count)
rows = []
for config in configs:
    scenario = Path(*Path(config).parts[-3:-1])
    summary_path = Path(results_dir) / scenario / "comparison.json"
    try:
        summary = json.loads(summary_path.read_text())[policy]
        if summary["total_steps"] != steps:
            raise ValueError(f"expected {steps} steps, got {summary['total_steps']}")
        reward = float(summary["total_reward"])
        if not math.isfinite(reward):
            raise ValueError(f"invalid total reward {reward}")
    except (OSError, KeyError, TypeError, ValueError) as error:
        sys.exit(f"Cannot score {scenario} from {summary_path}: {error}")
    rows.append((scenario, reward))

score = fmean(abs(reward / steps) for _, reward in rows)
print(f"\n{policy} leaderboard score = {score:.8f} over {len(rows)} scenarios (lower is better)")
print(f"{'Scenario':<32} {'Total reward':>14}")
for scenario, reward in rows:
    print(f"{str(scenario):<32} {reward:14.6f}")
PY
