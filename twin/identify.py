"""Fit the five independent thermal combinations plus a nuisance film coefficient.

All selection uses day one of the trace. Evaluation on day two uses the same
actions and the contemporaneous outside temperature, comfort flag and building
occupancy. No observed zone temperatures are applied after initialization.
"""
import argparse
import json
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares

from twin.experiment import RESULT, metadata, score, sensors
from twin.reconstruct import PROPERTIES
from twin.reduced import Geometry, Inputs, Mesh

KEYS = [k for k in PROPERTIES if k not in ("exterior_cv_density", "exterior_cv_heat_capacity")]
CAP = "exterior_volumetric_capacity"
KEYS += [CAP, "film"]
BOUNDS = [PROPERTIES[k][:2] for k in KEYS[:-2]] + [(350 * 840, 2400 * 1000), (5., 100.)]
LOW, HIGH = np.log(BOUNDS).T


def unpack(x):
    v = dict(zip(KEYS, np.exp(LOW + x * (HIGH - LOW))))
    # Choose a representative of the equivalence class inside both generator ranges.
    capacity = v.pop(CAP)
    v["exterior_cv_heat_capacity"] = float(np.clip(920., capacity / 2400, capacity / 350))
    v["exterior_cv_density"] = capacity / v["exterior_cv_heat_capacity"]
    return v


def run(stride, iterations, check=False):
    t0 = perf_counter()
    m = metadata()
    trace = np.load(RESULT / "trace.npz")
    obs = trace["obs"]
    names, actual, oat = sensors(m, obs)
    geo = Geometry(m)
    t1 = perf_counter()
    mesh = Mesh(geo, stride)
    t2 = perf_counter()
    inds = [geo.names.index(v) for v in names]
    f = m["observation_field_names"]
    inp = Inputs(geo, trace["continuous"], trace["discrete"], oat[:-1],
                 obs[:-1, f.index("comfort_mode_now")], obs[:-1, f.index("num_occupants")])
    attempts = []
    def simulate(v, n):
        v = dict(v)
        start = perf_counter()
        model = mesh.solver(v, v.pop("film"))
        built = perf_counter()
        pred = inp.run_steps(model, n)[:, inds]
        return pred, built - start, perf_counter() - built

    def residual(x):
        pred, setup, rollout = simulate(unpack(x), 288)
        err = (pred - actual[1:289]).ravel()
        attempts.append(dict(rmse_K=float(np.sqrt(np.mean(err**2))), params=unpack(x),
                             seconds=setup + rollout))
        return err

    def evaluate(v, label):
        pred, setup, rollout = simulate(v, 576)
        return score(m, obs, pred, label + f"_s{stride}", [rollout / 576], setup,
                     geometry_s=t1-t0, projection_s=t2-t1, input_precompute_s=input_s,
                     nodes=mesh.size, full_cells=len(geo.cells), params=v)

    input_s = perf_counter() - t2
    print("built", t1-t0, t2-t1, input_s, mesh.size, flush=True)
    x0 = np.full(len(KEYS), .5)
    if check:
        # Q2 only: measure the approximation floor at true physical parameters.
        from twin.experiment import FIXTURE, scalar_bindings
        truth = scalar_bindings((FIXTURE / "config.gin").read_text())
        evaluate(dict({k: truth[k] for k in PROPERTIES}, film=truth["convection_coefficient"]),
                 "coarse_oracle")
        return
    evaluate(unpack(x0), "prior")
    start_fit = perf_counter()
    # Absolute differences in scaled log space keep the Jacobian stable even
    # near bounds and include all nuisance-parameter costs in the timer.
    def jac(x):
        eps = .001
        base = residual(x)
        columns = []
        for j in range(len(x)):
            point = x.copy()
            point[j] += eps if x[j] < 1 - eps else -eps
            columns.append((residual(point) - base) / (point[j] - x[j]))
        return np.column_stack(columns)

    fit = least_squares(residual, x0, jac=jac, bounds=(0, 1), max_nfev=iterations,
                        ftol=2e-5, gtol=1e-6, xtol=1e-5)
    elapsed = perf_counter() - start_fit
    v = unpack(fit.x)
    evaluate(v, "identified")
    details = dict(fit_s=elapsed, attempts=attempts, evaluations=len(attempts),
                   jacobian_singular_values=np.linalg.svd(fit.jac, compute_uv=False).tolist(),
                   params=v, message=fit.message, training_steps=288, heldout_steps=288)
    (RESULT / f"fit_s{stride}.json").write_text(json.dumps(details, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stride", type=int, default=12)
    p.add_argument("--iterations", type=int, default=9)
    p.add_argument("--oracle", action="store_true")
    a = p.parse_args()
    run(a.stride, a.iterations, a.oracle)
