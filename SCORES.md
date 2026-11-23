# Team scores

Local score = `score_leaderboard.py` over the 30 shipped scenarios, 8640 steps, lower is better.
Do not compare with `baseline_returns.json`: it does not match our simulator runs (fairbanks_1 pid_water: -1829 shipped vs -2607 local).

## Rules
- One policy per file in `smart_buildings/baselines/`, one branch per person. Never commit `results/`.
- A policy replaces `transfer` only if it beats the current one on all 30 scenarios **and** on our own held-out generated buildings.
- Kaggle: 5 submissions/day for the whole team. Max 2 each, keep 1 spare. Log every submission below.
- Commands: `./eval_subset.sh <policy>` (3 hard scenarios, ~20 min), `./eval_all.sh <policy> results/<name>` (all 30 + score).

## Machine check (run `./eval_subset.sh setback` once on each machine, numbers must match)
| scenario | Dat | Hai |
|---|---:|---:|
| building_train_1/fairbanks_1 | -933.7 | |
| building_train_1/phoenix_5 | -401.4 | |
| building_train_2/denver_8 | -145.1 | |

## Local results (30 scenarios)
| date | policy | who | local score | notes |
|---|---|---|---:|---|
| 2026-11-21 | pid_water | Dat | 0.05798 | best shipped baseline, run locally |
| 2026-11-21 | setback | Dat (Codex) | 0.02444 | beats pid_water on 29/30; now `transfer` |
| 2026-11-21 | zone (margin inside band) | Dat | - | dropped: subset only; hurt phoenix_5 badly |

## Kaggle submissions
| date | who | policy / notebook version | local score | public LB |
|---|---|---|---:|---:|
