"""Audit, oracle replay and evaluation helpers. The fixture is confined here and in measure."""
import argparse
import ast
import csv
import json
import pickle
import re
from time import perf_counter

import numpy as np

from twin.measure import FIXTURE, RESULT
from twin.reconstruct import ObservedWeather, PROPERTIES, clock_anchor, make_environment, reconstruct


def metadata():
    with (RESULT / "metadata.pkl").open("rb") as src:
        return pickle.load(src)


def scalar_bindings(script):
    return {k: ast.literal_eval(v) for k, v in re.findall(
        r"^(\w+)\s*=\s*([\d.]+|'[^']*'|\"[^\"]*\")\s*(?:#.*)?$", script, re.M)}


def audit():
    from gin.config_parser import ConfigParser, BindingStatement
    from smart_buildings import evaluate
    def bindings(script):
        return {(s.scope, s.selector, s.arg_name): repr(s.value)
                for s in ConfigParser(script, evaluate.gin.config.ParserDelegate())
                if isinstance(s, BindingStatement)}
    actual = bindings((FIXTURE / "config.gin").read_text())
    built = bindings(reconstruct(metadata()))
    diff = {"/".join(filter(None, k)): [built.get(k), actual.get(k)]
            for k in sorted(actual.keys() | built.keys()) if actual.get(k) != built.get(k)}
    report = dict(total=len(actual), reconstructed=len(built), differences=diff)
    (RESULT / "bindings.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report)


def sensors(meta, obs):
    indices = meta["zone_temp_indices"]
    names = [meta["observation_field_names"][i].split("_zone_air")[0] for i in indices]
    norm = meta["observation_normalization_metadata"]["zone_air_temperature_sensor"]
    temps = obs[..., indices].astype(float) * norm["sample_std"] + norm["sample_mean"]
    oat = next(i for i, n in enumerate(meta["observation_field_names"])
               if n.endswith("_outside_air_temperature_sensor"))
    return names, temps, obs[..., oat].astype(float)


def score(meta, real_obs, predicted, label, timing, setup_s, **other):
    names, actual, _ = sensors(meta, real_obs)
    err = predicted - actual[1:1 + len(predicted)]
    rmse = lambda e: float(np.sqrt(np.mean(e ** 2)))
    metrics = dict(rmse_K=rmse(err), day1_K=rmse(err[:288]), day2_K=rmse(err[288:]),
                   max_abs_K=float(abs(err).max()), last_step_K=rmse(err[-1]),
                   steps=len(err), setup_s=setup_s, seconds_per_step=np.mean(timing),
                   per_zone_K=dict(zip(names, np.sqrt(np.mean(err**2, axis=0)))), **other)
    (RESULT / f"{label}.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(label, {k: v for k, v in metrics.items() if k != "per_zone_K"}, flush=True)
    return metrics


def full_replay(oracle):
    meta = metadata()
    trace = np.load(RESULT / "trace.npz")
    obs = trace["obs"]
    continuous, discrete = trace["continuous"], trace["discrete"]
    _, _, oat = sensors(meta, obs)
    if oracle:
        cfg = scalar_bindings((FIXTURE / "config.gin").read_text())
        materials = {k: cfg[k] for k in PROPERTIES}
        h, occupants, anchor = cfg["convection_coefficient"], cfg["work_occupancy"], cfg["start_timestamp"]
    else:
        materials = dict(json.loads((RESULT / "fit_s12.json").read_text())["params"])
        h, occupants, anchor = materials.pop("film"), 1, clock_anchor(meta, obs[0])
    weather = ObservedWeather(oat[0], h)
    start = perf_counter()
    env = make_environment(meta, weather, materials, occupants, anchor)
    sim = env.building.simulator
    occ_idx = meta["observation_field_names"].index("num_occupants")
    mode_idx = meta["observation_field_names"].index("comfort_mode_now")
    if not oracle:
        class MeanOccupants:
            count = 0.
            def average_zone_occupancy(self, zone_id, start_time, end_time):
                return self.count
        people = MeanOccupants()
        sim._occupancy = people
        # Exactly the same measured exogenous inputs as the reduced model.
        sim.hvac.is_comfort_mode = lambda timestamp: bool(obs[i, mode_idx])
    i = 0
    env.reset()
    setup = perf_counter() - start
    preds, times = [], []
    for i in range(len(continuous)):
        weather.temperature = oat[i]
        if not oracle:
            people.count = obs[i, occ_idx] / len(meta["zone_temp_indices"])
        t = perf_counter()
        x, *_ = env.step(dict(continuous_action=continuous[i], discrete_action=discrete[i]))
        times.append(perf_counter() - t)
        preds.append(sensors(meta, x)[1])
    label = "oracle" if oracle else "full_identified"
    np.savez_compressed(RESULT / f"{label}_trace.npz", predictions=preds, seconds=times)
    score(meta, obs, preds, label, times, setup,
          warm_seconds_per_step=float(np.mean(times[1:])),
          real_seconds_per_step=float(trace["timing"].mean()))


def export_zone_errors():
    labels = ["oracle", "coarse_oracle_s12", "prior_s12", "identified_s12",
              "full_identified", "sadr_transfer"]
    errors = [json.loads((RESULT / (label + ".json")).read_text())["per_zone_K"]
              for label in labels]
    with (RESULT / "zones.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["zone"] + [label + "_rmse_K" for label in labels])
        for zone in sorted(errors[0], key=lambda v: int(v.split("_")[1])):
            writer.writerow([zone] + [item[zone] for item in errors])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["audit", "oracle", "full_identified", "zones"])
    args = parser.parse_args()
    if args.mode == "audit":
        audit()
    elif args.mode == "zones":
        export_zone_errors()
    else:
        full_replay(args.mode == "oracle")
