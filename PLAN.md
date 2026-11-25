# Plan (updated 2026-11-23)

## Direction
Base: **SADR v14** (`smart_buildings/baselines/sadr_policy.py`, vendored unmodified, MIT, best public controller; key `sadr`).
On 3 hard scenarios it loses about 1/3 of what our old `setback` lost (fairbanks_1 -377 vs -934, phoenix_5 -125 vs -401, denver_8 -38 vs -145).

We beat it in two ways that public entries do not combine:

1. **Online-identified digital twin (main bet).** The runtime metadata gives floor plan, zone map, floors, AHU assignment and schedule; only the 6 envelope materials (and per-zone occupants) are hidden. A reduced twin of the shipped simulator, fitted to the first day of observed zone temperatures, predicts the next day to 0.045 K RMSE and runs at 0.0024 s/step (spike: `twin/`, `results/twin_spike/report.md`). Use it each night to choose setback depth and morning start time per AHU, where SADR's first-order model only reaches R^2 0.41.
2. **Distribution tuning.** Tune SADR's (and later the twin's) constants with an evolution strategy / CMA-ES over many generated buildings, never on the 30 shipped scenarios alone.

Guard against fooling ourselves: the public leaderboard uses 33% of the test data and the top 4 are within 1.5%, so a change is real only if it wins on **our validation set** (`smart_buildings/validation_scenarios`, 8 new buildings x 2 strips, `./eval_holdout.sh <policy>`) as well as on the 30 shipped scenarios.

Not allowed (disqualification risk): the gas-billing quirk, replaying simulator random seeds (occupancy/convection), reading config.gin or weather files at runtime, anything keyed to a scenario.

## Who does what

| Owner | Task | Files | Done when |
|---|---|---|---|
| Dat | `sadr_twin`: SADR + twin planning (setback depth, start time) | `smart_buildings/baselines/sadr_twin*.py`, `twin/` | beats `sadr` on subset + validation, policy time < 450 s/episode |
| Dat | Kaggle submissions and final selection | `kaggle/` | |
| Hai | 1. Machine check: `./eval_subset.sh setback`, compare with SCORES.md | SCORES.md | numbers match |
| Hai | 2. Validation baselines: `./eval_holdout.sh pid_water`, `setback`, `sadr` | SCORES.md | 3 scores logged |
| Hai | 3. Tuning harness: ES/CMA-ES over SADR's DEFAULT_PARAMS on generated buildings (new seeds, not the validation ones), objective = mean loss on 7-day windows, promote only if it wins on `eval_holdout.sh` | `tune/` (new), `smart_buildings/baselines/sadr_tuned_policy.py` | tuned params beat `sadr` on validation |
| Hai | 4. Paper skeleton (OpenReview, final deadline 2027-01-15): related work + method outline | `paper/` | outline reviewed |

Rules: one branch per person (`dat/...`, `hai/...`), merge into `dev` only with numbers in SCORES.md. Never edit another owner's files. Never commit `results/`. Full 8640-step runs only for comparisons.
