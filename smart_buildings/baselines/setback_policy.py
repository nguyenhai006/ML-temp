"""Schedule setback with per-zone, online estimates of morning recovery."""

from math import atan2, pi

import numpy as np

from smart_buildings.baselines.pid import PIDPolicy
from smart_buildings.baselines.pid_plus import ECON_K_PER_PA, MIN_STATIC_PRESSURE_PA


class SetbackPolicy(PIDPolicy):
    """Coast in eco mode and work back from the next comfort deadline."""

    name = "setback"

    # One observation and decision every five minutes.
    STEP_HOURS = 1 / 12
    MAX_LEAD_HOURS = 4.0
    VACANT_SETBACK_K = 0.65
    SAFETY_MARGIN_K = 3.0
    # Initial estimate of passive drift in K per step per K of outdoor gap.
    # An observed per-zone correction accounts for envelope and internal gains.
    OUTDOOR_DRIFT = 0.004
    # Initial ratio of coil water and zone air heat-capacity rates at full flow.
    COIL_CAPACITY_RATIO = 0.23
    MAX_DISCHARGE_K = 308.15
    MIN_DAMPER = 0.10

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        fields = {name: i for i, name in enumerate(self.field_names)}
        self._comfort_now = fields["comfort_mode_now"]
        self._comfort_soon = fields["comfort_mode_soon"]
        self._occupancy = fields["num_occupants"]
        occupancy_scale = building_meta_data.get("occupancy_normalization_constant", 0.0)
        self._empty_occupancy = -occupancy_scale / (1.0 + occupancy_scale)

        zones = []
        self._ahu_zones = []
        for ctrl in self.ahu_controllers:
            start = len(zones)
            zones.extend(ctrl["zones"])
            self._ahu_zones.append(slice(start, len(zones)))
        self._temp_obs = np.array([obs for _, obs, _ in zones], dtype=int)
        self._low_obs = np.array([
            fields.get(f"{vav}_zone_air_heating_temperature_setpoint_sensor", self._comfort_now)
            for vav, _, _ in zones
        ], dtype=int)
        self._high_obs = np.array([
            fields.get(f"{vav}_zone_air_cooling_temperature_setpoint_sensor", self._comfort_now)
            for vav, _, _ in zones
        ], dtype=int)
        self._damper_cmd = np.array([
            cmds["supply_air_damper_percentage_command"] for _, _, cmds in zones
        ], dtype=int)
        self._reheat_cmd = np.array([
            cmds["reheat_valve_percentage_command"] for _, _, cmds in zones
        ], dtype=int)
        outside_fields = [
            i for name, i in fields.items() if name.endswith("_outside_air_temperature_sensor")
        ]
        self._outdoor_obs = outside_fields[0]
        outdoor_normalization = self.observation_normalization_metadata.get(
            "outside_air_temperature_sensor", {}
        )
        self._outdoor_mean = outdoor_normalization.get("sample_mean", 0.0)
        self._outdoor_std = outdoor_normalization.get("sample_std", 1.0)
        self._water_obs = fields.get("hws_supply_water_temperature_sensor")
        water_normalization = self.observation_normalization_metadata.get(
            "supply_water_temperature_sensor", {}
        )
        self._water_mean = water_normalization.get("sample_mean", 0.0)
        self._water_std = water_normalization.get("sample_std", 1.0)

        normalization = building_meta_data.get("action_normalization_metadata", {})
        self._water_range = self._native_range(
            normalization, "supply_water_temperature_setpoint", 310.0, 350.0
        )
        self._supply_ranges = [
            self._native_range(
                normalization, f"{ctrl['ahu_id']}_supply_air_temperature_setpoint", 285., 305.
            )
            for ctrl in self.ahu_controllers
        ]
        self._pressure_ranges = [
            self._native_range(
                normalization, f"{ctrl['ahu_id']}_supply_air_static_pressure_setpoint", 0., 2000.
            )
            for ctrl in self.ahu_controllers
        ]
        self._off = np.full(self.n_continuous, -1.0, dtype=np.float32)
        self._off[self.recirculation_indices] = 1.0
        self.reset()

    @staticmethod
    def _native_range(normalization, field, minimum, maximum):
        entry = normalization.get(field, {})
        return entry.get("min_native_value", minimum), entry.get("max_native_value", maximum)

    def reset(self):
        super().reset()
        self._previous_temp = None
        if hasattr(self, "_temp_obs"):
            n = len(self._temp_obs)
            self._drift_correction = np.zeros(n)
            self._heating_rate = np.full(n, 0.7)
            self._cooling_rate = np.full(n, 0.7)
            self._previous_forcing = np.zeros(n)
            self._previous_fan = np.zeros(n, dtype=bool)
            self._observed_forcing = np.zeros(n)
            self._observed_fan = np.zeros(n, dtype=bool)
            self._previous_airflow = np.zeros(n)
            self._previous_discharge = np.zeros(n)
            self._modeled_change = np.zeros(n)
            self._air_response = np.ones(n)
            self._reheat_bias = np.zeros(n)
            self._preconditioning = np.zeros(n, dtype=bool)
            self._opening_slack = np.zeros(n)
            self._opening_offset = 0.0
            self._was_soon = None
            self._was_comfort = None

    def _steps_until_comfort(self, obs, soon):
        """Decode the countdown and correct the clock at observed soon edges."""
        hour = atan2(obs[self.hod_sin_index], obs[self.hod_cos_index]) * 12 / pi % 24
        day = round(atan2(obs[self.dow_sin_index], obs[self.dow_cos_index]) * 7 / (2 * pi)) % 7
        opening = (self.morning_start_hour + self._opening_offset) % 24
        if soon and self._was_soon is False:
            opening = (hour + 1.0) % 24
            self._opening_offset = opening - self.morning_start_hour

        hours_remaining = (opening - hour) % 24
        if soon:
            # The flag is the one-hour schedule lookahead, including holidays.
            # Fall back to one step if the clock and flag disagree.
            return min(max(hours_remaining, self.STEP_HOURS), 1.0) / self.STEP_HOURS
        if (
            (day + (hour >= opening)) % 7 < 5
            and 1.0 + self.STEP_HOURS / 2 < hours_remaining <= self.MAX_LEAD_HOURS
        ):
            return hours_remaining / self.STEP_HOURS
        return np.inf

    def _learn_rates(self, temperature, outdoor):
        """Learn passive drift and the response to substantial HVAC commands."""
        passive = self.OUTDOOR_DRIFT * (outdoor - temperature)
        if self._previous_temp is not None:
            change = temperature - self._previous_temp
            # VAVs report the temperature at the start of the previous step.
            # Match the observed change to the action that has reached them.
            unconditioned = ~self._observed_fan
            self._drift_correction[unconditioned] += 0.18 * np.clip(
                (change - passive - self._drift_correction)[unconditioned], -0.35, 0.35
            )
            residual = change - passive - self._drift_correction
            responsive = np.abs(self._modeled_change) >= 0.30
            measured_response = residual[responsive] / self._modeled_change[responsive]
            self._air_response[responsive] += 0.12 * (
                np.clip(measured_response, 0.25, 1.75) - self._air_response[responsive]
            )
            for sign, rate in ((1, self._heating_rate), (-1, self._cooling_rate)):
                forcing = sign * self._observed_forcing
                sample = forcing >= 0.40
                observed_rate = sign * residual[sample] / forcing[sample]
                rate[sample] += 0.10 * (np.clip(observed_rate, 0.12, 2.5) - rate[sample])
        return np.clip(passive + self._drift_correction, -1.0, 1.0)

    def get_action(self, obs):
        continuous = self._off.copy()
        discrete = np.zeros(self.n_discrete, dtype=np.int32)
        if not self._temp_obs.size:
            return {"continuous_action": continuous, "discrete_action": discrete}

        temperature = self._denormalize_zone_temp(obs[self._temp_obs].astype(float))
        outdoor = float(obs[self._outdoor_obs])
        # AHU systems prefix the measurement name, so their outdoor sensors
        # may report raw kelvin despite the generic normalizer in metadata.
        if not 170.0 < outdoor < 360.0:
            outdoor = outdoor * self._outdoor_std + self._outdoor_mean
        occupied = float(obs[self._occupancy]) > self._empty_occupancy + 1e-6
        comfort = obs[self._comfort_now] > 0.5
        soon = obs[self._comfort_soon] > 0.5
        comfort_low, comfort_high = self.comfort_temp_window
        comfort_width = comfort_high - comfort_low
        # Colder intake requires more reheat to hold the same room temperature.
        # Move the heating target inward to account for proportional droop.
        heating_margin = min(
            0.4 * comfort_width,
            0.25 + 0.035 * max(0.0, comfort_low - outdoor - 6.0),
        )
        cooling_margin = min(0.25, 0.25 * comfort_width)
        arrival_low = comfort_low + heating_margin
        cooling_buffer = min(0.45, 0.15 * comfort_width, 0.06 * max(0.0, outdoor - comfort_high))
        arrival_high = comfort_high - cooling_margin - cooling_buffer
        default_low, default_high = self.comfort_temp_window if comfort else self.eco_temp_window
        sensor_low, sensor_high = obs[self._low_obs], obs[self._high_obs]
        # At a schedule transition the VAV sensors can still show the old band.
        valid = (
            (np.abs(sensor_low - default_low) < 2.0)
            & (np.abs(sensor_high - default_high) < 2.0)
            & (sensor_low < sensor_high)
        )
        low = np.where(valid, sensor_low, default_low)
        high = np.where(valid, sensor_high, default_high)
        drift = self._learn_rates(temperature, outdoor)
        # Estimate where the previous action left each zone while its sensor
        # reading is still in flight. Air at low flow moves a zone less.
        exchange = self._previous_airflow / (self._previous_airflow + 0.15)
        self._modeled_change = exchange * (self._previous_discharge - temperature)
        pending = self._air_response * self._modeled_change
        predicted = temperature + drift + np.clip(pending, -8.0, 8.0)

        heating_speed = np.maximum(0.08, self._heating_rate + np.minimum(drift, 0.0))
        cooling_speed = np.maximum(0.08, self._cooling_rate - np.maximum(drift, 0.0))
        remaining = np.inf
        if comfort:
            if self._was_comfort is False:
                # Correct the next start for the recovery time we still
                # needed when this comfort period began.
                late = np.maximum(
                    (comfort_low - predicted) / heating_speed,
                    (predicted - comfort_high) / cooling_speed,
                )
                self._opening_slack *= 0.7
                self._opening_slack += 0.3 * np.clip(late, 0.0, 18.0)
            # A small occupied margin makes room for the next five minutes of drift.
            target_low = low + heating_margin
            target_high = high - cooling_margin if occupied else high
            self._preconditioning.fill(False)
        else:
            setback = 0.0 if occupied else self.VACANT_SETBACK_K
            target_low, target_high = low - setback, high + setback
            remaining = self._steps_until_comfort(obs, soon)
            if np.isfinite(remaining):
                # The passive forecast is deliberately short to limit extrapolation
                # from a transient at fan shutoff or from a changing outdoor load.
                arrival_temp = predicted + drift * min(remaining, 8.0)
                heat_needed = np.maximum(
                    0.0, arrival_low - np.minimum(predicted, arrival_temp)
                )
                cool_needed = np.maximum(
                    0.0, np.maximum(predicted, arrival_temp) - arrival_high
                )
                recovery = np.maximum(heat_needed / heating_speed, cool_needed / cooling_speed)
                lead = 3.0 + 1.6 * recovery + self._opening_slack
                self._preconditioning |= (recovery > 0.0) & (remaining <= lead)
                target_low = np.where(
                    self._preconditioning, np.maximum(low, arrival_low), target_low
                )
                target_high = np.where(
                    self._preconditioning, np.minimum(high, arrival_high), target_high
                )
            else:
                self._preconditioning.fill(False)

        # Correct sustained cold shortfall during warm-up and comfort hours.
        # Let this correction decay outside those hours so the building can coast.
        maintain_comfort = (comfort | self._preconditioning) & (outdoor < comfort_low - 8.0)
        warmup = self._preconditioning & maintain_comfort
        self._reheat_bias *= np.where(maintain_comfort, 0.995, 0.85)
        self._reheat_bias += np.where(
            maintain_comfort,
            np.where(warmup, 0.18, 0.09) * np.clip(arrival_low - predicted, -2.0, 4.0),
            0.0,
        )
        np.clip(self._reheat_bias, 0.0, np.where(warmup, 7.5, 4.0), out=self._reheat_bias)
        heating = np.clip(
            (target_low - predicted) / 2.1 + 0.10 * self._reheat_bias, 0.0, 1.0
        )
        cooling = np.clip((predicted - target_high) / 1.35, 0.0, 1.0)
        # Reach full demand before the 5 K boundary, using the colder/hotter
        # observation if the one-step forecast would mask a large excursion.
        safety_low, safety_high = low, high
        if remaining <= 4.0:
            safety_low, safety_high = np.maximum(low, comfort_low), np.minimum(high, comfort_high)
        heating[np.minimum(temperature, predicted) < safety_low - self.SAFETY_MARGIN_K] = 1.0
        cooling[np.maximum(temperature, predicted) > safety_high + self.SAFETY_MARGIN_K] = 1.0

        self._previous_temp = temperature
        self._observed_forcing[:] = self._previous_forcing
        self._observed_fan[:] = self._previous_fan
        self._previous_forcing.fill(0.0)
        self._previous_fan.fill(False)
        self._previous_airflow.fill(0.0)
        self._previous_discharge.fill(0.0)
        self._was_soon = bool(soon)
        self._was_comfort = bool(comfort)
        heating_load = float(heating.max())
        pump_fraction = min(1.0, 0.40 + 0.60 * heating_load)
        water_setpoint = min(1.0, 0.20 + 1.25 * heating_load)
        min_water, max_water = self._water_range
        requested_water = min_water + 0.5 * (water_setpoint + 1) * (max_water - min_water)
        water_temp = requested_water
        if self._water_obs is not None:
            water_temp = float(obs[self._water_obs]) * self._water_std + self._water_mean
        water_capacity = self.COIL_CAPACITY_RATIO * pump_fraction ** 0.5
        any_reheat = False

        for i, (ctrl, zones) in enumerate(zip(self.ahu_controllers, self._ahu_zones)):
            heat, cool = heating[zones], cooling[zones]
            max_heat = float(heat.max(initial=0.0))
            max_cool = float(cool.max(initial=0.0))
            load = max(max_heat, max_cool)
            if load <= 0.02:
                continue

            discrete[ctrl["run_discrete_idx"]] = 1
            self._previous_fan[zones] = True
            min_supply, max_supply = self._supply_ranges[i]
            min_pressure, max_pressure = self._pressure_ranges[i]
            floor = max(MIN_STATIC_PRESSURE_PA, min_pressure)
            # The reheat coil can warm only so much intake air. Keep heating
            # airflow low enough for the coil to reach a useful discharge
            # temperature, especially while the water loop is still warming.
            indoor = float(predicted[zones].mean())
            mixed_at_max_recirc = 0.7 * indoor + 0.3 * outdoor
            heating_flow_limit = np.clip(
                water_capacity * max(10.0, water_temp - mixed_at_max_recirc)
                / max(3.0, self.MAX_DISCHARGE_K - mixed_at_max_recirc),
                0.24,
                0.70,
            )
            airflow_request = np.maximum(heat * heating_flow_limit, cool)
            # Airflow scales with sqrt(pressure). Open dampers first and raise
            # pressure only when more flow is needed by this handler's zones.
            flow = max((floor / max_pressure) ** 0.5, float(airflow_request.max()))
            pressure = min(max_pressure, max_pressure * flow ** 2)
            continuous[ctrl["pressure_idx"]] = (
                2 * (pressure - min_pressure) / (max_pressure - min_pressure) - 1
            )

            damper = airflow_request / flow
            airflow = np.maximum(self.MIN_DAMPER, damper) * flow
            indoor = float(np.average(predicted[zones], weights=airflow))
            recirc = 0.7
            supply = max_supply
            if max_cool > 0.02:
                # A lightly loaded zone still gets some supply whenever a
                # neighbor calls for cold air. Keep that air near the bottom
                # of the current band to limit abrupt overcooling.
                coldest_supply = max(min_supply, float(target_low[zones].max()) - 2.25)
                supply = min(max_supply, coldest_supply + 2.0 * (1.0 - max_cool))
                if max_heat <= 0.02 and indoor - outdoor > ECON_K_PER_PA * pressure:
                    recirc = min(0.7, max(0.0, (supply - outdoor) / (indoor - outdoor)))
            continuous[ctrl["temp_idx"]] = 2 * (supply - min_supply) / (max_supply - min_supply) - 1
            continuous[self.recirculation_indices[i]] = 2 * recirc - 1

            # Modulate reheat against entering air, water temperature and
            # airflow. Temper cool supply air near the heating edge even when
            # a neighboring zone is the one that called for the fan.
            entering = min(supply, recirc * indoor + (1.0 - recirc) * outdoor)
            discharge = np.minimum(
                self.MAX_DISCHARGE_K,
                target_low[zones] + 0.85
                + np.maximum(-2.0, 0.65 * (target_low[zones] - predicted[zones]))
                + self._reheat_bias[zones],
            )
            if max_heat > 0.02 and outdoor < default_low - 8.0:
                # A warmed zone still receives cold outdoor air through its
                # minimum damper when another zone needs heat. Temper that
                # air, including for warm zones calling for gentle cooling.
                discharge = np.where(
                    cool > 0.02,
                    np.maximum(
                        target_low[zones] - 0.25,
                        target_high[zones] - 1.0 - 3.0 * cool,
                    ),
                    discharge,
                )
                temper = True
            else:
                temper = (cool <= 0.02) & (
                    (heat > 0.02) | (predicted[zones] < target_low[zones] + 1.1)
                )
            rise_at_full_flow = water_capacity * max(10.0, water_temp - entering)
            valve = np.where(
                temper,
                np.clip((discharge - entering) * airflow / rise_at_full_flow, 0.0, 1.0),
                0.0,
            )
            continuous[self._damper_cmd[zones]] = 2 * damper - 1
            continuous[self._reheat_cmd[zones]] = 2 * valve - 1
            self._previous_forcing[zones] = heat - cool
            self._previous_airflow[zones] = airflow
            self._previous_discharge[zones] = entering + valve * rise_at_full_flow / airflow
            any_reheat |= bool(np.any(valve > 0.0))

        if any_reheat and self.water_temp_idx is not None:
            # Favor a warm loop with modest pump head for light reheat. Raise
            # both as a zone approaches its full heating request.
            discrete[self.water_discrete_idx] = 1
            continuous[self.water_temp_idx] = water_setpoint
            if self.water_pressure_idx is not None:
                continuous[self.water_pressure_idx] = 2 * pump_fraction - 1

        return {"continuous_action": continuous, "discrete_action": discrete}
