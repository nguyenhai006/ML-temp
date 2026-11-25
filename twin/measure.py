"""Research harness for a single training fixture. Run as python -m twin.measure."""
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
from smart_buildings import evaluate as ev

RESULT = Path("results/twin_spike")
FIXTURE = ev.DEFAULT_SCENARIOS_DIR / "building_train_1" / "atlanta_2"


def record(policy="pid_water"):
    RESULT.mkdir(parents=True, exist_ok=True)
    before = time.perf_counter()
    env = ev.load_hybrid_environment(FIXTURE / "config.gin")
    meta = ev.create_building_meta_data(env, FIXTURE)
    if policy == "pid_water":
        with (RESULT / "metadata.pkl").open("wb") as stream:
            pickle.dump(meta, stream)
    controller = ev.initialize_policy(ev.POLICY_MAP[policy], meta, env.action_spec())
    obs, _ = env.reset()
    observations, continuous, discrete = [obs.copy()], [], []
    timing = []
    for _ in range(576):
        action = ev.validate_action(controller.get_action(obs), env.action_spec())
        start = time.perf_counter()
        obs, _, terminated, truncated, _ = env.step(action)
        assert not (terminated or truncated)
        timing.append(time.perf_counter() - start)
        observations.append(obs.copy())
        continuous.append(action["continuous_action"].copy())
        discrete.append(action["discrete_action"].copy())
    filename = "trace.npz" if policy == "pid_water" else f"{policy}_trace.npz"
    np.savez_compressed(RESULT / filename, obs=observations, continuous=continuous,
                        discrete=discrete, timing=timing)
    print("Reference wall seconds", time.perf_counter() - before)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["sadr", "pid_water"], default="pid_water")
    record(p.parse_args().policy)
