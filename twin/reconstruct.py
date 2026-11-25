"""Reconstruct generator bindings from participant metadata, without scenario files.

The returned gin text uses symbolic asset names for the two arrays and weather.
In a separate research simulator process, make_environment injects the arrays and
a scalar OAT input directly. Gin and the stock convection kernel have process-
global state, so do not install this second environment inside a running evaluator.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from smart_buildings.smart_control.utils import scenario_generator as sg
from smart_buildings.smart_control.utils.materials import PROPERTIES


def reconstruct(metadata):
    """Return the public generator's gin bindings with placeholders for assets.

    Unknown scalars take generator defaults. Geometry is supplied by the caller
    in memory when materializing the environment. No geometry generation or IO.
    """
    cfg = sg.ScenarioConfig(name="metadata_twin", write_metrics_dir=False)
    cfg.setpoint_schedule = sg.SetpointScheduleConfig(
        metadata["morning_start_hour"], metadata["evening_start_hour"],
        *metadata["comfort_temp_window"], *metadata["eco_temp_window"],
        str(metadata["time_zone"]))
    cfg.materials = sg.MaterialsConfig()
    groups = metadata["ahu_zone_assignments"]
    groups = [groups[f"ahu_{i}"] for i in range(1, len(groups) + 1)]
    script = sg._generate_gin_config("<floor_plan>", "<zone_map>", groups, cfg)
    # Replace the generator's default sinusoid with the recorded-weather
    # interface. Its CSV and strip-specific constant are unavailable.
    i = script.index("convection_coefficient =")
    j = script.index("weather_controller = @sim/WeatherController()", i)
    j += len("weather_controller = @sim/WeatherController()")
    weather = f"""convection_coefficient = {cfg.weather.convection_coefficient}
weather_csv_path = '<observed OAT>'

sim/ReplayWeatherController:
  local_weather_path = %weather_csv_path
  convection_coefficient = %convection_coefficient

weather_controller = @sim/ReplayWeatherController()"""
    return script[:i] + weather + script[j:]


def clock_anchor(metadata, first_obs):
    """A neutral calendar week with the observed initial weekday and minute."""
    fields = metadata["observation_field_names"]
    def phase(name, period):
        x = first_obs[fields.index(name + "_cos_000")]
        y = first_obs[fields.index(name + "_sin_000")]
        return round(np.arctan2(y, x) * period / (2 * np.pi)) % period
    return datetime(2001, 1, 8) + timedelta(days=phase("dow", 7),
                                             minutes=phase("hod", 1440))


@dataclass
class ObservedWeather:
    temperature: float
    coefficient: float

    def get_current_temp(self, timestamp):
        return self.temperature

    def get_air_convection_coefficient(self, timestamp):
        return self.coefficient


def make_environment(metadata, weather, materials=None, occupants=1, start=None):
    """Build the full research twin using arrays, a clock and scalar weather.

    'start' may be the result of clock_anchor. For otherwise missing inputs,
    pass estimates from observations or a prior. No real environment is read.
    """
    from smart_buildings import evaluate as ev  # registers simulator gin types
    gin = ev.gin
    script = reconstruct(metadata)
    # Bind every weather reference to one mutable object (including the AHUs).
    script = script.replace("@sim/ReplayWeatherController()", "None")
    script = script.replace('"<floor_plan>"', "None").replace('"<zone_map>"', "None")
    gin.clear_config()
    with gin.unlock_config():
        gin.parse_config(script)
        def macro(name, value):
            gin.bind_parameter(name + "/gin.macro.value", value)
        macro("weather_controller", weather)
        for k, v in (materials or {}).items():
            if k not in PROPERTIES:
                raise ValueError(k)
            macro(k, float(v))
        macro("work_occupancy", occupants)
        if start is not None:
            macro("start_timestamp", str(start))
        for name in ("floor_plan", "zone_map"):
            gin.bind_parameter("sim/FloorPlanBasedBuilding." + name, metadata[name])
        gin.bind_parameter("hvac/SetpointSchedule.holidays", metadata["holidays"])
    env = ev.hybrid_action_environment.HybridActionEnvironment()
    # Gin deep-copies non-configurable Python bindings for each instantiation.
    # Share one live sensor source across the simulator and its AHUs.
    sim = env.building.simulator
    sim._weather_controller = weather
    for ahu in sim.hvac.air_handler.ahus:
        ahu._sim_weather_controller = weather
    return env
