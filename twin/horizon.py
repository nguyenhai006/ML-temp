"""Budget and weather-forecast diagnostics, conditional on the PID action path."""
import argparse
import json
from time import perf_counter

import numpy as np

from twin.experiment import RESULT, metadata, score, sensors
from twin.reduced import Geometry, Inputs, Mesh


def state(model):
    return model.air.copy(), model.mass.copy(), model.zones.copy(), model.tank


def restore(model, values):
    model.air, model.mass, model.zones, model.tank = values


def run():
    meta = metadata()
    trace = np.load(RESULT / "trace.npz")
    obs, cont, disc = trace["obs"], trace["continuous"], trace["discrete"]
    labels, actual, oat = sensors(meta, obs)
    fields = meta["observation_field_names"]
    mode = obs[:-1, fields.index("comfort_mode_now")]
    occ = obs[:-1, fields.index("num_occupants")]
    start = perf_counter()
    geo = Geometry(meta)
    end_geo = perf_counter()
    geometry_s = end_geo - start
    mesh = Mesh(geo, 12)
    end_projection = perf_counter()
    params = dict(json.loads((RESULT / "fit_s12.json").read_text())["params"])
    model = mesh.solver(params, params.pop("film"))
    built = perf_counter()
    order = [geo.names.index(v) for v in labels]
    exogenous = Inputs(geo, cont, disc, oat[:-1], mode, occ)
    init_input_s = perf_counter() - built
    start = perf_counter()
    replay = exogenous.run_steps(model, 576)[:, order]
    batch_s = (perf_counter() - start) / 576
    model.reset()
    start = perf_counter()
    for i in range(576):
        sl = slice(i, i + 1)
        one = Inputs(geo, cont[sl], disc[sl], oat[sl], mode[sl], occ[sl])
        one.run_steps(model, 1, reset=False)
    step_s = (perf_counter() - start) / 576

    # Keep a copy of the reconstructed state at each forecast origin. Update
    # the live path with observed inputs, never teacher-force its zone state.
    model.reset()
    exogenous.run_steps(model, 288)
    horizons = [12, 36, 72]
    err = {n: [] for n in horizons}
    base = {n: [] for n in horizons}
    oat_err = {n: [] for n in horizons}
    for i in range(288, 576, 12):
        origin = state(model)
        for n in horizons:
            if i + n > 576:
                continue
            sl = slice(i, i + n)
            # Isolate OAT forecast error: both arms see the identical recorded
            # actions, comfort flags and aggregate occupancy for the horizon.
            # This measures sensitivity, not the return of a deployed planner.
            held = Inputs(geo, cont[sl], disc[sl], np.full(n, oat[i]), mode[sl], occ[sl])
            pred = held.run_steps(model, n, reset=False)[:, order]
            err[n].append(pred[-1] - actual[i+n])
            base[n].append(replay[i+n-1] - actual[i+n])
            oat_err[n].append(oat[i] - oat[i+n])
            restore(model, origin)
        exogenous.run_steps(model, 12, start=i, reset=False)
    rmse = lambda a: float(np.sqrt(np.mean(np.square(a))))
    delta = np.diff(actual[289:], axis=0)
    delta_resid = np.diff(replay[288:], axis=0) - delta
    scores = dict(geometry_s=geometry_s, projection_s=end_projection-end_geo,
                  material_setup_s=built-end_projection, batch_input_setup_s=init_input_s,
                  batch_s_per_step=batch_s, streaming_s_per_step=step_s,
                  pooled_replay_K=rmse(replay-actual[1:]),
                  day2_change_R2=float(1 - np.sum(delta_resid**2) / np.sum((delta-delta.mean())**2)),
                  endpoint={n//12: dict(origins=len(err[n]), persistence_temp_rmse_K=rmse(err[n]),
                                        observed_oat_temp_rmse_K=rmse(base[n]), oat_rmse_K=rmse(oat_err[n]))
                            for n in horizons})
    (RESULT / "horizons.json").write_text(json.dumps(scores, indent=2) + "\n")
    print(scores)


def action_transfer():
    """Replay a controller never used in calibration and check the capacity ridge."""
    meta = metadata()
    data = np.load(RESULT / "sadr_trace.npz")
    training = np.load(RESULT / "trace.npz")
    obs = data["obs"]
    names, _, oat = sensors(meta, obs)
    fields = meta["observation_field_names"]
    begin = perf_counter()
    geo = Geometry(meta)
    mesh = Mesh(geo, 12)
    order = [geo.names.index(v) for v in names]
    v = dict(json.loads((RESULT / "fit_s12.json").read_text())["params"])
    h = v.pop("film")
    model = mesh.solver(v, h)
    inputs = Inputs(geo, data["continuous"], data["discrete"], oat[:-1],
                    obs[:-1, fields.index("comfort_mode_now")], obs[:-1, fields.index("num_occupants")])
    built = perf_counter()
    pred = inputs.run_steps(model, 576)[:, order]
    elapsed = perf_counter() - built

    # Different admissible density and specific heat at the fitted product.
    capacity = v["exterior_cv_density"] * v["exterior_cv_heat_capacity"]
    v["exterior_cv_heat_capacity"] = 980.
    v["exterior_cv_density"] = capacity / 980.
    alt = inputs.run_steps(mesh.solver(v, h), 576)[:, order]
    different = (np.any(training["continuous"] != data["continuous"], axis=1) |
                 np.any(training["discrete"] != data["discrete"], axis=1))
    score(meta, obs, pred, "sadr_transfer", [elapsed / 576], built - begin,
          distinct_actions=int(different.sum()), alternative_capacity_J_m3K=capacity,
          alternative_density=v["exterior_cv_density"], alternative_cp=v["exterior_cv_heat_capacity"],
          capacity_pair_max_abs_K=float(np.max(np.abs(alt-pred))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-transfer", action="store_true")
    if parser.parse_args().action_transfer:
        action_transfer()
    else:
        run()
