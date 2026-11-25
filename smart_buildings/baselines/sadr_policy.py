# Vendored from https://github.com/neelsirivara/neurips2026-smart-buildings
# submission/candidates/transfer_policy_v14.py (SADR v14), MIT License, Copyright (c) 2026 neelsirivara.
# Unmodified below; our changes go in subclasses elsewhere.
"""SADR: Schedule-Aware Demand-based Reset controller for the SB26 challenge (v12, final; round 10, r10_K:
wk_ext_lo 12, season_ext_steps 2016, season_oat_k 288.15, mpr_on 1, mpr_burst 0, mpr_safety 1.3 on top of v11; every other round-10 switch off).

v13 = v12 + round-12 CMA constants with v12's setback depths (r12var; round13/choice.json).

v14 = v13 + cooling handover ramp st_hand_c_h=1.0 (struct_r1 HA; decision_HA_v13fresh.json)

struct_r1 research copy (experiments/src/policies_struct/sadr_struct.py, NOT a submission file): v13 plus four
structural switches, all off by default (= v13 bit for bit): ``st_late`` (late-recovery escalation), ``st_hand_c_h``
(occupied cooling target ramp at the morning handover), ``st_swing`` (swing-zone preconditioning target) and
``st_msat`` (cost-model supply air for handlers with heating and cooling zones). See results/struct_r1/PLAN.md.

Round-10 switches (default off): ``wk_free`` (weekend free-float: from the last workday's end until the lead into
the workday that follows a non-workday, a set-back zone has no heating floor beyond ``wk_free_ext_k`` below the eco
heating setpoint), ``wk_lead_max_h`` (> 0: that recovery's lead may be up to this long) and ``wk_no_mbr`` (1 = that
recovery uses the learned-rate lead only, not the first-order model lead), ``ki_c_mult`` / ``kq_c_mult``
(multipliers on the zone cooling loop's integral / proportional gain), ``tw_hold_steps`` (> 0: the loop is kept
hot this long after the last heating demand, separately from the heating season that gates the night flush) and
``tw_hold_oat_k`` (> 0: that hold only while the outside air was below this temperature in the last 24 h),
``season_ext_steps`` / ``season_oat_k`` (> 0: the whole heating season is extended to this many steps after the last
heating demand, only while the outside air was below season_oat_k in the last 24 h).

Round-9 switches (default off): ``obr_on`` (occupied band riding: heat occupied zones in bursts at the
discharge cap and let handlers stop between bursts; reheat the forced air of heating-regime zones to the
cap while a handler runs) and ``tw_ramp_k`` (hot-water setpoint kept within this many K above the tank
temperature, never below it, while the pump runs).

v10:

v10 turns on the overrides of r7L_mbr2 in DEFAULT_PARAMS (sb_boost 1, sb_boost_h 8, sb_boost_cold 1, sb_boost_floor_k 2, lead_safety 1.15, night_low_ext 10, mbr_on 1, mbr_lead_cap 1); with ``mbr_lead_cap`` the model may only shorten the learned-rate lead and is ignored where it predicts the target is unreachable at the cap. Round-7 switches (all off reproduces the v9 code bit for bit): ``mbr_on`` (model-based
optimum start: a per-zone first-order thermal model dT/dt = a Q - k (T - Tout) + c fitted online by recursive
least squares on the zone's own unoccupied history; the lead is the model's recovery time at the request cap
times ``mbr_safety``) and ``mbr_floor`` (model-based setback depth: the floor that minimises the model's
predicted gas for holding it plus recovering from it, among floors reached in time and recoverable within the
lead limit, down to ``mbr_floor_ext_max`` below the eco heating setpoint).

v9 candidate code (round 6):

Round-6 switches (all off by default, which reproduces v8 bit for bit): ``frost_floor_k`` (absolute lower bound
of the setback floor, a frost-protection variant), ``sb_boost`` / ``sb_boost_h`` / ``sb_boost_gap`` (in heating
season, the forced minimum flow of a set-back zone on a running handler within ``sb_boost_h`` hours of a workday
start is reheated to the discharge limit instead of to the zone temperature: the outside-air share of that flow is
paid anyway, so the extra heat costs 1 J of gas per J, against about 1.9 J per J in the zone's own ventilated
recovery), ``sb_sync_k`` (a set-back zone within ``sb_sync_k`` of its floor joins a heating burst already running
on its handler, so the handler's zones share run time instead of each starting it), ``rate_oat_slope`` (optimum
start: the learned recovery rate is shifted by slope x (today's lift - the lift it was learned at), lift = target -
outside air), ``margin_occ_heat`` (occupied heating margin used while the outside air is below
``margin_occ_heat_oat``), ``sb_boost_floor_k`` (the same boost, at any time, for a set-back zone within this many K
of its setback floor: it would need a burst of its own soon).

v8 (round 5, r5_c12_comp):

v8 turns on the overrides of r5_c12_comp in DEFAULT_PARAMS (wk_ext_lo 6, wk_ext_hi 4, q_cap_w 20000); the round-5 switches (with them off the code reproduces v7 bit for bit) are:
``mon_lead_mult`` / ``mon_lead_max_h`` / ``mon_no_learn`` (longer optimum-start lead into a workday that follows a
non-workday, and no rate learning from it), ``wk_ext_lo`` / ``wk_ext_hi`` (deeper setback far from the next
workday start, ramped out ``wk_ramp_end_h`` .. ``wk_ramp_end_h + wk_ramp_h`` hours before it), ``lag_comp``
(one-step extrapolation of the zone temperature in the PI errors) and ``f_cap_pre_c`` (separate flow cap for
cooling while preconditioning) and ``pre_burst_h`` (a preconditioning zone far below its target requests
its full flow-capped heat at once instead of ramping up through the PI integral).

v7 (round 4, r4_c14_comp_a):

Round-4 switches (v7 turns on the five overrides of r4_c14_comp_a in DEFAULT_PARAMS:
disch_margin_k 0, occ_press_dyn with e0 0.1 K / span 0.5 K, flush_on + flush_heat_guard,
sb_neutral, f_cap_pre 0.316; with all of them off the code reproduces v6 bit for bit). Switches: ``f_cap_min`` (lower clip of the request flow cap), ``occ_press_dyn``
(occupied request cap that only allows handler pressure above the floor in proportion to
the zone's error), ``sat_mix_relief`` (on a handler that heats and cools, lower the supply
air below the mixed floor just enough to serve the cooling zones at floor pressure),
``flush_on`` (night flush with outside air before a workday when the outside air is
cooler than the zones), ``sb_neutral`` (in heating season, reheat the forced minimum flow
of set-back zones on a running handler to the zone temperature).

Drop-in replacement for ``smart_buildings/baselines/transfer_policy.py`` in the
official starter kit. Pure numpy + stdlib, deterministic, no file or network
access, shape-agnostic (any number of air handlers K and zones V). Every slot
of both action vectors is written explicitly on every step; every value is
clipped to the declared range, and non-finite inputs fall back to safe values.

The controller is organised around three facts about the plant and the score
that hold for every generated building:

  * The comfort term is charged only while a zone is occupied, and occupants
    are only present on weekday comfort hours.  Outside those hours the only
    cost is energy, so the plant is shut down (SETBACK) except to bound the
    fabric drift, and brought back just in time (PRECONDITION) with an
    optimum-start lead that is re-estimated from each morning's recovery.
    A weekday on which nobody has arrived by mid-morning is a holiday in the
    occupancy model, so it is treated as unoccupied.
  * Fan power is flow x pressure, and zone flow goes as damper x sqrt(pressure),
    so the cheapest way to move a given amount of air is wide dampers at the
    200 Pa ventilation floor.  Static pressure is therefore reset
    (trim-and-respond) so that the most demanding damper is fully open, and is
    only raised above the floor when a zone's error is large.
  * Heat comes only from the VAV reheat coils, whose discharge is capped at
    308.15 K, and gas is billed as heat delivered plus a small tank loss.  The
    hot water loop is therefore run hot whenever the building is in heating
    season (hotter water does not cost more gas per joule delivered), valves
    are sized from the heat each zone needs, and the loop differential
    pressure is reset so the most-open valve sits near full travel.

v6 (round 3) makes the zone loops scale with the building instead of with
constants tuned on 100 m2 zones.  Every VAV box has the same design flow
(2 m3/s), so a 500 m2 zone has a fifth of the heating authority per square
metre of a 100 m2 zone and several times the thermal mass; with fixed gains
and a fixed 12 kW request cap it recovers from a night setback at well under
1 K/h and sits below the band for the first hours of every workday.  v6:

  1. reads each zone's floor area from the metadata zone map (4-connected
     rooms in raster order reproduce the simulator's room numbering) and
     scales the PI gains, the request cap and the setback burst with
     area / 100 m2 (clipped);
  2. caps each zone's request at what its box can physically deliver at a
     bounded flow fraction (f_cap_pre while preconditioning or set back,
     f_cap_occ while occupied), i.e. at the discharge-air limit for heating
     and the coldest supply air for cooling;
  3. runs the optimum start per zone: each zone learns its own recovery rate
     and enters PRECONDITION at its own lead, so fast zones are not woken
     with the slowest one;
  4. raises the setback floor (lowers the ceiling) of a zone whose learned
     recovery rate could not bring it back within the maximum lead;
  5. caps a heating request at what the reheat coil can deliver with its valve
     fully open at the maximum loop differential pressure (the hot-water Kv
     is the same for every box): past that point more airflow only cools the
     discharge air and the zone receives less heat, not more.

With ``gain_area_scaling``, ``per_zone_start`` and ``adaptive_floor`` at 0 and
``f_cap_*`` at 1 the controller reproduces v5 up to the request cap.

Per step, vectorised over zones:

  1. Mode (OCCUPIED / PRECONDITION / SETBACK) per zone from the comfort-mode
     flags, the cyclic time features, the building occupancy count and the
     zone's own lead.
  2. Zone demands from each zone's temperature against the mode's band
     (sensor setpoints while occupied, comfort window while preconditioning,
     widened eco window while set back).
  3. Air handler: runs iff some served zone has demand (dwell hysteresis);
     supply-air temperature reset from cooling demand (raised when the same
     handler also has heating zones); static pressure from the maximum flow
     demand; recirculation from an economiser rule with the fan-power
     break-even implied by the simulator's exhaust-fan model.
  4. Zones: dampers from flow demand; reheat valves sized from the water
     capacity rate each heating zone needs, given the discharge-air limit.
  5. Hot water: pump iff some valve is open; differential pressure from the
     largest valve demand; supply water temperature hot in heating season,
     idle otherwise.

All gains and bounds live in ``DEFAULT_PARAMS``.  A JSON file named by the
environment variable ``SADR_PARAMS_JSON`` (used only by the local tuning
harness) overrides them; when the variable is unset the defaults are used.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import deque
from collections.abc import Mapping
from typing import Any

import numpy as np

# Native action ranges from the shipped action normalisers (identical in every
# generated config; read from metadata when present, these are the fallbacks).
SAT_MIN_K = 285.0
SAT_MAX_K = 305.0
P_MAX_PA = 2000.0          # design static pressure: 100% damper gives design flow
P_FLOOR_PA = 200.0         # simulator floor while a handler runs
DAMPER_FLOOR = 0.1         # simulator floor while a handler runs
RECIRC_MAX = 0.7           # simulator cap
TW_MIN_K = 310.0
TW_MAX_K = 350.0
DP_MAX_BAR = 20.0
DISCHARGE_MAX_K = 308.15   # VAV discharge-air limit in the simulator
AIR_RHO_CP = 1.2 * 1006.0  # J/m3/K
WATER_RHO_CP = 1000.0 * 4180.0
# Economiser break-even, K of outside-air advantage per Pa of static pressure:
# COP / (fan_eff * rho * cp) = 3.5 / (0.9 * 1.2 * 1006).
ECON_K_PER_PA = 0.00322
CV_AREA_M2 = 0.04          # 20 cm control volumes in the zone map

F_FLOOR100 = math.sqrt(P_FLOOR_PA / P_MAX_PA)   # flow fraction, 100% damper at floor pressure
F_MIN = DAMPER_FLOOR * F_FLOOR100               # flow fraction, 10% damper at floor pressure

DEFAULT_PARAMS: dict[str, float] = {
    # --- band handling (K) ---
    "margin_occ": 0.289468493362518,        # inside the sensor band at which control starts (occupied)
    "margin_pre": 0.8277288469370598,        # same, while preconditioning (target comfort band)
    "night_low_ext": 10.0,     # drift allowed below the eco heating setpoint while set back
    "night_high_ext": 6.0,    # drift allowed above the eco cooling setpoint while set back
    "sb_hyst_k": 1.0,         # setback burst: heat/cool a zone from the floor until this far inside
    "q_sb_w": 4350.239442077557,         # net heat (W) requested per 100 m2 zone during a setback burst
    "e_full_k": 1.0,          # (unused, kept for parameter-file compatibility)
    "e_full2_k": 4.0,         # (unused, kept for parameter-file compatibility)
    # --- optimum start ---
    "lead_base_h": 0.46092233188230813,       # fixed part of the preconditioning lead (h)
    "lead_min_h": 0.75,
    "lead_max_h": 8.0,
    "rate0_kph": 2.0,         # initial recovery-rate estimate (K/h), heating and cooling
    "rate_min_kph": 0.4,
    "rate_max_kph": 8.0,
    "rate_ema": 0.5,          # weight of the newest observed recovery rate
    "lead_safety": 1.0894479119580387,       # lead uses rate / safety
    "holiday_check_h": 7.0,   # weekday hour by which somebody must have arrived
    "per_zone_start": 1.0,    # v6: 1 = each zone learns its own rate and has its own lead
    "adaptive_floor": 1.0,    # v6: 1 = setback floor/ceiling limited to what the zone can recover
    "pre_latch": 0.0,         # v6: 1 = a zone that started preconditioning holds the target until 06:00 (tested: costlier)
    # --- zone PI loops (net heat into the zone air, W, per 100 m2 of zone area) ---
    "kq_w_per_k": 581.8664289382334,      # proportional gain: W of net heat per K of error
    "ki_w_per_k_step": 204.8034490847218,  # integral gain: W added per K of error per 5-min step
    "i_decay_mult": 1.0,      # integral unwinding speed when the zone is past its target
    "q_cap_w": 20000.0,       # cap on net heat (either sign) a zone may ask for, per 100 m2
    "q_on_w": 50.0,           # a zone with |Q| above this is "in demand"
    "area_ref_m2": 100.0,     # v6: zone area at which the gains above apply unscaled
    "gain_area_scaling": 1.0,  # v6: 1 = scale kq, ki, q_cap_w, q_sb_w with area / area_ref
    "area_scale_min": 1.0,    # v6: clip of the area scale factor
    "area_scale_max": 6.0,
    "f_cap_pre": 0.31622776601683794,         # v6: flow-fraction cap on a zone's request while preconditioning / set back
    "f_cap_occ": 1.0,         # v6: same while occupied (1 = the box's full design flow)
    "water_cap": 1.0,         # v6: 1 = cap heating requests at what the coil can deliver with the valve
                              #     fully open at the maximum loop dP (more airflow would cool the discharge)
    "water_cap_frac": 0.97,   # v6: fraction of that limit to request (keeps the valve off its stop)
    # --- air handler ---
    "min_on_steps": 3,        # dwell (5-min steps) after a start
    "min_off_steps": 2,       # dwell after a stop
    "sat_cool_hi": 285.0,     # SAT ceiling (K) while cooling; 285 = always the coldest supply (v5)
    "sat_mix_floor": 293.67185622344374,   # SAT floor while the same handler also has heating zones
    "disch_margin_k": 0.0,    # keep discharge air this far below the 308.15 K limit
    "econ_on": 1.0,
    # --- zones ---
    "kv_valve": 0.108,        # reheat valve Kv at full open (m3/h/sqrt(bar)), kit default
    "valve_target": 0.940431020597017,      # dP reset aims the most-open valve here
    # --- hot water ---
    "dp_lo_bar": 0.1,
    "dp_hi_bar": 20.0,
    "tw_idle_k": 310.0,
    "tw_hot_k": 350.0,
    "heat_season_steps": 288, # keep the loop hot this long after the last heating demand
    # --- round-4 candidate switches (defaults = v6) ---
    "f_cap_min": 0.31622776601683794,  # lower clip of f_cap_*; v6 = sqrt(200/2000), the floor-pressure flow
    "occ_press_dyn": 1.0,     # 1 = occupied request cap ramps from f_cap_occ_lo to f_cap_occ with the error
    "f_cap_occ_lo": 0.31622776601683794,  # occupied cap while the error is below occ_press_e0
    "occ_press_e0": 0.1,      # K beyond the occupied target before the cap starts to rise
    "occ_press_span": 0.5,    # K over which it rises to f_cap_occ
    "sat_mix_relief": 0.0,    # 1 = mixed handler SAT = max(need at floor pressure, relief_min), <= sat_mix_floor
    "sat_mix_relief_min": 285.0,
    "flush_on": 1.0,          # 1 = night flush with outside air before a workday
    "flush_dt_k": 3.0,        # outside air must be this much colder than the zone
    "flush_depth_k": 1.0,     # flush target = hi_pre - depth (not below lo_pre + flush_lo_margin_k)
    "flush_lo_margin_k": 0.5,
    "flush_window_h": 8.0,    # only within this many hours before a workday start
    "flush_heat_guard": 1.0,  # 1 = no flush in heating season (heat_timer > 0)
    "sb_neutral": 1.0,        # 1 = heating season: reheat set-back zones' forced air to zone temperature
    "sb_neutral_cold": 0.0,   # 1 = ... only while the outside air is below the zone's setback floor
    # --- round-5 candidate switches (defaults = v7) ---
    "mon_lead_mult": 1.0,     # lead multiplier for the recovery into a workday that follows a non-workday
    "mon_lead_max_h": 8.0,    # cap of that lead (h)
    "mon_no_learn": 0.0,      # 1 = recoveries into such a workday do not update the learned rates
    "wk_ext_lo": 11.888338660994712,         # extra setback depth (K, below the floor) far from the next workday start
    "wk_ext_hi": 4.0,         # extra setback height (K, above the ceiling) far from the next workday start
    "wk_ramp_h": 12.0,        # the extra depth ramps out over this many hours ...
    "wk_ramp_end_h": 10.0,    # ... ending this many hours before the next workday start
    "lag_comp": 0.0,          # zone PI uses T + lag_comp * (T - T_prev) (one-step sensor lag)
    "f_cap_pre_c": -1.0,      # >= 0: separate request flow-fraction cap for cooling while preconditioning
    "pre_burst_h": 0.0,       # 1 = a preconditioning zone more than pre_burst_e0 below target requests its full cap
    "pre_burst_e0": 0.3,
    # --- round-6 candidate switches (defaults = v8) ---
    "frost_floor_k": 0.0,     # > 0: the setback floor never goes below this temperature (K)
    "sb_boost": 1.0,          # 1 = reheat set-back zones' forced air to the discharge limit before a workday
    "sb_boost_h": 11.232239405726546,        # ... within this many hours of the workday start
    "sb_boost_gap": 0.5,      # ... while the zone is at least this far below its preconditioning target (K)
    "sb_boost_cold": 1.0,     # 1 = boost only while the outside air is at or below the zone's setback floor
    "sb_sync_k": 0.0,         # > 0: set-back zones this close to their floor join a heating burst on their handler
    "sb_boost_floor_k": 0.6792934058135757,  # > 0: in heating season, a set-back zone on a running handler within this many K of
                              #      its floor gets its forced air reheated to the discharge limit (any time)
    "rate_oat_slope": 0.0,    # K/h of recovery rate per K of lift (target - outside air) vs the learned lift
    "margin_occ_heat": -1.0,  # >= 0: occupied heating margin while the outside air is below margin_occ_heat_oat
    "margin_occ_heat_oat": 273.15,
    # --- round-7 candidate switches (defaults = the v9 code) ---
    "mbr_on": 1.0,            # 1 = model-based optimum start (per-zone first-order model, RLS)
    "mbr_floor": 0.0,         # 1 = model-based setback depth (min predicted hold + recovery gas)
    "mbr_safety": 1.1,        # multiplier on the model-predicted recovery time
    "mbr_min_n": 36.0,        # heated 30-min windows a zone needs before its model is used
    "mbr_lambda": 0.998,      # RLS forgetting factor per update
    "mbr_floor_ext_max": 16.0,  # deepest floor considered, K below the eco heating setpoint
    "mbr_lead_cap": 1.0,      # 1 = (post hoc, ledger) the model may only shorten the learned-rate lead
    # --- round-9 candidate switches (defaults = v10) ---
    "obr_on": 0.0,            # 1 = occupied band riding (bursts at the cap, handlers stop between bursts)
    "obr_start_k": 0.3,       # a heating-regime occupied zone starts a burst below lo_s + this (K)
    "obr_join_k": 0.6,        # ... joins a burst already running on its handler below lo_s + this
    "obr_stop_k": 0.9,        # ... ends its burst above lo_s + this
    "obr_boost_k": 0.9,       # forced air of heating-regime occupied zones on a running handler is reheated to
                              #     the discharge cap below lo_s + this
    "obr_cool_guard_k": 1.0,  # a zone within this of its cooling target is never in the heating regime
    "tw_ramp_k": 0.0,         # > 0: pump on -> setpoint within [tank, tank + tw_ramp_k]
    "dow_snap": 1.0,          # 1 = round the decoded day of week (fixes Saturday = workday, Sunday evening)
    # --- round-10 candidate switches (defaults = v11) ---
    "wk_free": 0.0,           # 1 = weekend free-float (no heating floor from Friday evening to Monday's lead)
    "wk_free_ext_k": 30.0,    # ... the floor is then this many K below the eco heating setpoint
    "wk_lead_max_h": 0.0,     # > 0: lead cap (h) of the recovery into a workday that follows a non-workday
    "wk_no_mbr": 0.0,         # 1 = that recovery does not use the model lead (mbr)
    "ki_c_mult": 1.0,         # multiplier on the zone cooling loop's integral gain
    "kq_c_mult": 1.0,         # multiplier on the zone cooling loop's proportional gain
    "tw_hold_steps": 0.0,     # > 0: loop kept hot this long after the last heating demand (heating season unchanged)
    "tw_hold_oat_k": 0.0,     # > 0: ... only while the outside air was below this (K) at some step of the last 24 h
    "season_ext_steps": 2016.0,  # > 0: heating season this long after the last heating demand (while cold, below)
    "season_oat_k": 288.15,      # > 0: ... only while the outside air was below this (K) at some step of the last 24 h
    "mpr_on": 1.0,            # 1 = model-predictive recovery start (second-order online zone model)
    "mpr_safety": 1.3,        # multiplier on the model's recovery time
    "mpr_min_n": 288.0,       # unoccupied 5-min samples a zone needs before its model is used
    "mpr_lambda": 0.9995,     # RLS forgetting factor per sample
    "mpr_burst": 0.0,         # 1 = (with mpr_on) a preconditioning zone below target heats at the request cap
    # --- struct_r1 switches (all off = v13 bit for bit) ---
    "st_late": 0.0,           # 1 = late-recovery escalation: a preconditioning zone whose observed rise rate cannot reach
    "st_late_h": 2.0,         #     its target by the workday start (checked within st_late_h hours of it) requests its
    "st_late_fcap": 1.0,      #     full heat at flow fraction st_late_fcap (occupied-style cap) until it reaches the target
    "st_hand_c_h": 1.0,       # > 0: occupied cooling target starts at the preconditioning target at the zone's first
                              #       occupied step and relaxes linearly to the occupied target over this many hours
    "st_swing": 0.0,          # 1 = swing zones (cooled >= st_swing_min steps in the last workday's comfort hours)
    "st_swing_k": 0.1,        #     precondition to comfort_lo + margin_occ + st_swing_k instead of comfort_lo + margin_pre
    "st_swing_min": 6.0,
    "st_msat": 0.0,           # 1 = supply air of a handler with heating and cooling zones chosen by a per-step cost model
    "st_msat_r": 2.1,         #     (chiller + fans + reheat gas); score cost of 1 J of electricity per J of gas
}


def _num_suffix(name: str) -> int:
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else 0


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def _room_areas(zone_map) -> np.ndarray | None:
    """Floor area (m2) of room_1..room_n from the metadata zone map, or None.

    The simulator numbers rooms by 4-connected components of the free cells
    (value 0) in raster order of their first cell; this reproduces that
    numbering with a breadth-first search, so index i-1 is room_i.
    """
    try:
        zm = np.asarray(zone_map)
    except Exception:  # noqa: BLE001
        return None
    if zm.ndim != 2 or zm.size == 0 or zm.size > 20_000_000:
        return None
    free = zm == 0
    if not free.any():
        return None
    H, W = free.shape
    lab = np.zeros((H, W), dtype=np.int32)
    counts: list[int] = []
    rows, cols = np.nonzero(free)
    for r0, c0 in zip(rows.tolist(), cols.tolist()):
        if lab[r0, c0]:
            continue
        n = len(counts) + 1
        lab[r0, c0] = n
        size = 1
        dq = deque([(r0, c0)])
        while dq:
            r, c = dq.popleft()
            if r + 1 < H and free[r + 1, c] and not lab[r + 1, c]:
                lab[r + 1, c] = n
                size += 1
                dq.append((r + 1, c))
            if r > 0 and free[r - 1, c] and not lab[r - 1, c]:
                lab[r - 1, c] = n
                size += 1
                dq.append((r - 1, c))
            if c + 1 < W and free[r, c + 1] and not lab[r, c + 1]:
                lab[r, c + 1] = n
                size += 1
                dq.append((r, c + 1))
            if c > 0 and free[r, c - 1] and not lab[r, c - 1]:
                lab[r, c - 1] = n
                size += 1
                dq.append((r, c - 1))
        counts.append(size)
    return np.asarray(counts, dtype=float) * CV_AREA_M2


class TransferPolicy:
    """Schedule-aware demand-based reset (SADR) rule-based controller."""

    name = "transfer"

    OCCUPIED, PRECOND, SETBACK = 0, 1, 2

    def __init__(
        self,
        building_meta_data: Mapping[str, Any],
        action_spec: Any,
        params: Mapping[str, float] | None = None,
    ) -> None:
        md = building_meta_data
        self.metadata = md
        self.action_spec = action_spec
        self.p = dict(DEFAULT_PARAMS)
        if params:
            self.p.update(params)
        env_path = os.environ.get("SADR_PARAMS_JSON")
        if env_path:
            try:
                with open(env_path) as f:
                    self.p.update(json.load(f))
            except (OSError, ValueError):
                pass
        p = self.p

        self.n_discrete = int(md["n_discrete_action"])
        self.n_continuous = int(md["n_continuous_action"])
        names = list(md["observation_field_names"])
        self.ix = {n: i for i, n in enumerate(names)}

        temp_indices = list(md["temp_indices"])
        pressure_indices = list(md["pressure_indices"])
        recirc_indices = list(md["recirculation_indices"])
        water_indices = list(md["water_indices"])
        zone_action_indices = md["zone_action_indices"]
        assignments = md.get("ahu_zone_assignments", {}) or {}

        # ---- air handlers, in the positional order of the action lists ----
        ahu_ids = sorted(assignments.keys(), key=_num_suffix)
        self.K = len(temp_indices)
        if len(ahu_ids) > self.K:
            ahu_ids = ahu_ids[: self.K]
        self.ahu_temp_idx = np.array(temp_indices[: self.K], dtype=int)
        self.ahu_press_idx = np.array(
            [pressure_indices[i] if i < len(pressure_indices) else -1 for i in range(self.K)], dtype=int)
        self.ahu_recirc_idx = np.array(
            [recirc_indices[i] if i < len(recirc_indices) else -1 for i in range(self.K)], dtype=int)
        self.ahu_oat_idx = np.array([
            self.ix.get(f"ahs_{aid}_outside_air_temperature_sensor", -1) for aid in ahu_ids
        ] + [-1] * (self.K - len(ahu_ids)), dtype=int)
        bare = [n for n in names if n.endswith("outside_air_temperature_sensor") and not n.startswith("ahs_")]
        self.bare_oat_idx = self.ix[bare[0]] if bare else -1
        onorm = md.get("observation_normalization_metadata", {}) or {}
        oat_n = onorm.get("outside_air_temperature_sensor", {})
        self.oat_mean = float(oat_n.get("sample_mean", 295.0))
        self.oat_std = float(oat_n.get("sample_std", 10.0)) or 10.0

        # ---- zones ----
        zt = onorm.get("zone_air_temperature_sensor", {})
        self.zt_mean = float(zt.get("sample_mean", 295.0))
        self.zt_std = float(zt.get("sample_std", 5.0)) or 5.0

        vav_to_ahu: dict[str, int] = {}
        for a, aid in enumerate(ahu_ids):
            for room in assignments[aid]:
                vav_to_ahu[room.replace("room_", "VAV_")] = a

        z_t, z_lo, z_hi, z_d, z_v, z_a, z_mf, z_num = [], [], [], [], [], [], [], []
        self.orphan_slots: list[int] = []  # action slots of zones we cannot see or place
        for vav_id, acts in zone_action_indices.items():
            d_idx = acts.get("supply_air_damper_percentage_command")
            v_idx = acts.get("reheat_valve_percentage_command")
            t_idx = self.ix.get(f"{vav_id}_zone_air_temperature_sensor")
            a_idx = vav_to_ahu.get(vav_id)
            if t_idx is None or a_idx is None:
                for s in (d_idx, v_idx):
                    if s is not None:
                        self.orphan_slots.append(int(s))
                continue
            z_t.append(t_idx)
            z_lo.append(self.ix.get(f"{vav_id}_zone_air_heating_temperature_setpoint_sensor", -1))
            z_hi.append(self.ix.get(f"{vav_id}_zone_air_cooling_temperature_setpoint_sensor", -1))
            z_d.append(-1 if d_idx is None else int(d_idx))
            z_v.append(-1 if v_idx is None else int(v_idx))
            z_a.append(a_idx)
            z_mf.append(self.ix.get(f"{vav_id}_supply_air_flowrate_setpoint", -1))
            z_num.append(_num_suffix(vav_id))
        self.V = len(z_t)
        self.z_t = np.array(z_t, dtype=int)
        self.z_lo = np.array(z_lo, dtype=int)
        self.z_hi = np.array(z_hi, dtype=int)
        self.z_d = np.array(z_d, dtype=int)
        self.z_v = np.array(z_v, dtype=int)
        self.z_a = np.array(z_a, dtype=int)
        self.z_mf = np.array(z_mf, dtype=int)
        self.member = np.zeros((self.K, max(self.V, 1)), dtype=bool)
        if self.V:
            self.member[self.z_a, np.arange(self.V)] = True
        self.ahu_has_zones = self.member.any(axis=1)
        self.ahu_zone_count = np.maximum(self.member.sum(axis=1), 1)

        # ---- zone floor areas -> gain scale (v6) ----
        area_ref = max(float(p.get("area_ref_m2", 100.0)), 1.0)
        self.area_m2 = np.full(max(self.V, 1), area_ref)
        self.area_known = False
        if self.V and float(p.get("gain_area_scaling", 1.0)) > 0.5:
            areas = _room_areas(md.get("zone_map"))
            if areas is not None and len(areas) >= max(z_num) >= 1 and min(z_num) >= 1:
                cand = areas[np.array(z_num, dtype=int) - 1]
                if np.all(np.isfinite(cand)) and np.all(cand > 0) and len(areas) == self.V:
                    self.area_m2 = cand.astype(float)
                    self.area_known = True
        if self.V and self.area_known:
            self.scale = np.clip(self.area_m2 / area_ref, float(p.get("area_scale_min", 0.3)),
                                 float(p.get("area_scale_max", 6.0)))
        else:
            self.scale = np.ones(max(self.V, 1))

        # ---- hot water system ----
        self.water_temp_idx = int(water_indices[0]) if water_indices else -1
        self.water_press_idx = int(pressure_indices[-1]) if len(pressure_indices) > self.K else -1
        self.water_disc_idx = self.n_discrete - 1
        self.hws_temp_obs_idx = self.ix.get("hws_supply_water_temperature_sensor", -1)

        # ---- schedule & time ----
        cw = md.get("comfort_temp_window", (294.0, 297.0))
        ew = md.get("eco_temp_window", (289.0, 298.0))
        self.comfort_lo, self.comfort_hi = float(cw[0]), float(cw[1])
        self.eco_lo, self.eco_hi = float(ew[0]), float(ew[1])
        self.morning_start = float(md.get("morning_start_hour", 6))
        self.evening_start = float(md.get("evening_start_hour", 19))
        self.hod_sin = self.ix.get("hod_sin_000", -1)
        self.hod_cos = self.ix.get("hod_cos_000", -1)
        self.dow_sin = self.ix.get("dow_sin_000", -1)
        self.dow_cos = self.ix.get("dow_cos_000", -1)
        self.now_idx = self.ix.get("comfort_mode_now", -1)
        self.soon_idx = self.ix.get("comfort_mode_soon", -1)
        self.occ_idx = self.ix.get("num_occupants", -1)

        # ---- action normalisation (fallbacks match every shipped config) ----
        anorm = md.get("action_normalization_metadata", {}) or {}

        def rng(key_sub, lo, hi):
            for k, v in anorm.items():
                if key_sub in k:
                    try:
                        return float(v["min_native_value"]), float(v["max_native_value"])
                    except (KeyError, TypeError, ValueError):
                        return lo, hi
            return lo, hi

        self.sat_rng = rng("supply_air_temperature_setpoint", SAT_MIN_K, SAT_MAX_K)
        self.p_rng = rng("static_pressure_setpoint", 0.0, P_MAX_PA)
        self.tw_rng = rng("supply_water_temperature_setpoint", TW_MIN_K, TW_MAX_K)
        self.dp_rng = rng("differential_pressure_setpoint", 0.0, DP_MAX_BAR)

        # Optional per-step trace for local development only (never set on Kaggle).
        self._dbg = None
        dbg_path = os.environ.get("SADR_DEBUG_LOG")
        if dbg_path:
            try:
                self._dbg = open(dbg_path, "w")
                self._dbg.write("step,hod,dow,now,soon,occupied,mode,holiday,lead_h,rate_h,rate_c,"
                                "deficit_h,deficit_c,Tmin,Tmax,Tmean,n_heat,n_cool,maxP,n_on,tw,dp,pump,"
                                "max_valve,max_damper,sat_min,recirc_min,n_pre\n")
            except OSError:
                self._dbg = None

        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        p = self.p
        Vn = max(self.V, 1)
        self.ahu_on = np.zeros(self.K, dtype=bool)
        self.dwell = np.zeros(self.K, dtype=int)
        self.recirc_prev = np.full(self.K, RECIRC_MAX)
        self.I_h = np.zeros(Vn)   # integral of heating demand, W
        self.I_c = np.zeros(Vn)   # integral of cooling demand, W
        self.sb_latch_h = np.zeros(Vn, dtype=bool)  # setback burst latches
        self.sb_latch_c = np.zeros(Vn, dtype=bool)
        self.heat_timer = 0
        self.tw_timer = 0          # round 10: loop-hold timer (tw_hold_steps)
        self.oat_min_hist = deque(maxlen=288)   # round 10: per-step minimum outside air, last 24 h (tw_hold_oat_k)
        self.season_timer = 0      # round 10: extended heating season (season_ext_steps)
        self.season_on = False     # round 10: extended season active (previous step's value at the flush guard)
        self.step_count = 0
        self.occ_min = math.inf
        self.last_hod = None
        self.seen_occ_today = False
        self.holiday = False
        self.mode = self.SETBACK
        self.zone_mode = np.full(Vn, self.SETBACK, dtype=int)
        self.flush_latch = np.zeros(Vn, dtype=bool)  # round 4: night flush per zone
        self.T_prev = None                           # round 5: previous zone temperatures (lag compensation)
        self.after_off = False                       # round 5: the coming workday follows a non-workday
        # optimum start, building level (fallback when per_zone_start == 0)
        self.rate_h = float(p["rate0_kph"])
        self.rate_c = float(p["rate0_kph"])
        self.pre_active = False
        self.pre_t0 = 0            # step index at which preconditioning started
        self.pre_deficit0 = 0.0
        self.pre_kind = 0          # +1 heating deficit, -1 cooling deficit
        self.pre_reached_step = -1
        # optimum start, per zone (v6)
        self.rate_h_z = np.full(Vn, float(p["rate0_kph"]))
        self.rate_c_z = np.full(Vn, float(p["rate0_kph"]))
        self.zpre_active = np.zeros(Vn, dtype=bool)
        self.zpre_t0 = np.zeros(Vn, dtype=int)
        self.zpre_def0 = np.zeros(Vn)
        self.zpre_kind = np.zeros(Vn, dtype=int)
        self.zpre_reached = np.full(Vn, -1, dtype=int)
        self.pre_latch = np.zeros(Vn, dtype=bool)   # zone stays in PRECONDITION until comfort hours
        self.lift_h_z = np.full(Vn, np.nan)          # round 6: lift (target - outside air) the heating rate was learned at
        self.lift_c_z = np.full(Vn, np.nan)
        self.zpre_lift0 = np.zeros(Vn)
        self._lead_dbg = -1.0
        # round 7: per-zone first-order model dT/dt [K/h] = a Q[kW] - k (T - Tout) + c, RLS state
        self.mbr_W = 6                                   # steps per window (30 min)
        self.mbr_th = np.tile(np.array([0.5, 0.1, 0.0]), (Vn, 1))
        self.mbr_P = np.tile(np.diag([10.0, 1.0, 1.0]), (Vn, 1, 1))
        self.mbr_n = np.zeros(Vn)                        # heated windows seen
        self.mbr_hT = np.full((self.mbr_W + 1, Vn), np.nan)
        self.mbr_hX = np.full((self.mbr_W + 1, Vn), np.nan)   # T - Tout
        self.mbr_hQ = np.full((self.mbr_W + 1, Vn), np.nan)   # kW delivered in the step that follows
        self.mbr_hU = np.zeros((self.mbr_W + 1, Vn), dtype=bool)  # unoccupied mode in that step
        self.mbr_q_prev = np.zeros(Vn)
        self.mbr_floor_cache = np.full(Vn, np.nan)
        self.mbr_floor_age = 10 ** 6
        self.obr_latch = np.zeros(Vn, dtype=bool)        # round 9: occupied burst latches
        self.obr_regime = np.zeros(Vn, dtype=bool)       # round 9: zone in the occupied heating regime
        # round 9 (mpr): second-order zone model dT(k+1) = a dT(k) + b Q(k)[kW] - c (T - Tout) + d, RLS state
        self.mpr_th = np.tile(np.array([0.5, 0.05, 0.002, 0.0]), (Vn, 1))
        self.mpr_P = np.tile(np.diag([1.0, 1.0, 0.01, 0.1]), (Vn, 1, 1))
        self.mpr_n = np.zeros(Vn)
        self.mpr_T1 = None                               # T at the previous step
        self.mpr_T2 = None                               # T two steps back
        self.mpr_To1 = None                              # outside air at the previous step
        self.mpr_U1 = np.zeros(Vn, dtype=bool)           # previous step was unoccupied (setback / precondition)
        # struct_r1 state
        self.st_Thist = deque(maxlen=7)                  # zone temperatures of the last 7 steps (st_late)
        self.st_late_latch = np.zeros(Vn, dtype=bool)
        self.st_pre_n = np.zeros(Vn, dtype=int)          # consecutive preconditioning steps
        self.st_occ_t0 = np.zeros(Vn, dtype=int)         # step of the zone's first occupied step today (st_hand)
        self.st_cool_cnt = np.zeros(Vn)                  # occupied cooling steps today (st_swing)
        self.st_swing_z = np.zeros(Vn, dtype=bool)
        self.st_prev_now = False

    # ---------------- round 7: first-order model helpers ----------------
    def _mbr_A(self, F_design, p):
        """kW/K: capacity rate of the preconditioning request cap (flow fraction f_cap_pre at the discharge limit)."""
        f = float(np.clip(p.get("f_cap_pre", F_FLOOR100), max(float(p.get("f_cap_min", F_FLOOR100)), 0.0), 1.0))
        return F_design * f / 1000.0

    def _mbr_trec(self, T0, T1, Tout, F_design, p):
        """Hours to heat from T0 to T1 at the cap, first-order model (inf where the cap cannot get there)."""
        a, k, c = self.mbr_th[:, 0], self.mbr_th[:, 1], self.mbr_th[:, 2]
        A = self._mbr_A(F_design, p)
        td = DISCHARGE_MAX_K - max(p["disch_margin_k"], 0.0)
        beta = np.maximum(a * A + k, 1e-6)
        tinf = (a * A * td + k * Tout + c) / beta
        T1 = np.broadcast_to(T1, np.shape(T0))
        ok = tinf > T1 + 0.05
        num = np.where(ok, tinf - T0, 1.0)
        den = np.where(ok, tinf - T1, 1.0)
        t = np.where(T0 >= T1, 0.0, np.log(np.maximum(num, 1e-6) / np.maximum(den, 1e-6)) / beta)
        return np.where(ok | (T0 >= T1), np.maximum(t, 0.0), np.inf)

    def _mbr_floor(self, T, base, lo_pre, Tout, F_design, hours_to_workday, p):
        """Per zone, the setback floor with the least predicted gas until the next workday start."""
        a, k, c = self.mbr_th[:, 0], self.mbr_th[:, 1], self.mbr_th[:, 2]
        A = self._mbr_A(F_design, p)
        td = DISCHARGE_MAX_K - max(p["disch_margin_k"], 0.0)
        safety = float(p.get("mbr_safety", 1.1))
        lowest = self.eco_lo - float(p.get("mbr_floor_ext_max", 16.0))
        H = max(hours_to_workday - p["lead_base_h"], 0.0)
        rec_max = max(p["lead_max_h"] - p["lead_base_h"], 0.5)
        best = np.array(base, dtype=float).copy()
        best_cost = np.full(len(best), np.inf)
        beta = np.maximum(a * A + k, 1e-6)
        tinf = (a * A * td + k * Tout + c) / beta
        teq = Tout + c / np.maximum(k, 1e-6)                # unheated equilibrium
        for j in range(0, 33):
            F = base - 0.5 * j
            if j > 0 and np.all(F < lowest):
                break
            t_rec = self._mbr_trec(F, lo_pre, Tout, F_design, p)
            ok = np.isfinite(t_rec) & (safety * t_rec <= min(rec_max, H)) & (F >= lowest - 1e-9)
            tr = np.where(ok, t_rec, 0.0)
            # recovery gas: integral of A (Td - 0.3 Tout - 0.7 T(t)) along T(t) = tinf - (tinf - F) e^(-beta t)
            g_rec = A * ((td - 0.3 * Tout - 0.7 * tinf) * tr
                         + 0.7 * (tinf - F) * (1.0 - np.exp(-beta * tr)) / beta)
            # free float from T down to F, then hold F until the recovery must start
            reach = (F > teq + 0.05) & (T > F)
            t_ff = np.where(reach, np.log(np.maximum(T - teq, 1e-6) / np.maximum(F - teq, 1e-6))
                            / np.maximum(k, 1e-6), np.where(T <= F, 0.0, np.inf))
            t_hold = np.clip(H - safety * tr - t_ff, 0.0, H)
            q_hold = np.maximum((k * (F - Tout) - c) / np.maximum(a, 1e-6), 0.0)
            g_hold = 1.0 + 0.3 * (F - Tout) / np.maximum(td - F, 0.5)
            cost = g_hold * q_hold * t_hold + g_rec
            if j > 0:
                ok &= t_hold > 0.0                           # a floor never reached changes nothing
            better = ok & (cost < best_cost - 1e-9)
            best = np.where(better, F, best)
            best_cost = np.where(better, cost, best_cost)
        return best

    @staticmethod
    def _norm(x, lo, hi):
        if hi <= lo:
            return 0.0
        return 2.0 * (x - lo) / (hi - lo) - 1.0

    def _cyclic(self, obs, si, ci, period):
        if si < 0 or ci < 0:
            return None
        s, c = float(obs[si]), float(obs[ci])
        if not (math.isfinite(s) and math.isfinite(c)):
            return None
        ang = math.atan2(s, c)
        if ang < 0:
            ang += 2.0 * math.pi
        return period * ang / (2.0 * math.pi)

    def _oat(self, obs) -> np.ndarray:
        """Outside air temperature (K) per AHU (NaN when unavailable)."""
        out = np.full(self.K, np.nan)
        m = self.ahu_oat_idx >= 0
        if m.any():
            out[m] = obs[self.ahu_oat_idx[m]]
        if self.bare_oat_idx >= 0:
            v = float(obs[self.bare_oat_idx]) * self.oat_std + self.oat_mean
            out[~m] = v
        out = np.where(np.isfinite(out) & (out > 150.0) & (out < 350.0), out, np.nan)
        return out

    # -------------------------------------------------------------- the step
    def get_action(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        p = self.p
        obs = np.asarray(obs, dtype=np.float64)
        cont = np.zeros(self.n_continuous, dtype=np.float32)
        disc = np.zeros(self.n_discrete, dtype=np.int32)
        K, V = self.K, self.V
        self.step_count += 1
        per_zone_start = V > 0 and float(p.get("per_zone_start", 1.0)) > 0.5

        # ---------------- time & flags ----------------
        hod = self._cyclic(obs, self.hod_sin, self.hod_cos, 24.0)
        dow = self._cyclic(obs, self.dow_sin, self.dow_cos, 7.0)
        if dow is not None and float(p.get("dow_snap", 0.0)) > 0.5:
            # round 9: the feature is an integer day sent as float32 sin/cos; Saturday decodes to 4.999999999
            dow = float(int(round(dow)) % 7)
        if self.now_idx >= 0 and math.isfinite(obs[self.now_idx]):
            now = float(obs[self.now_idx]) > 0.5
        else:  # reconstruct from the schedule
            now = (hod is not None and dow is not None and dow < 5.0
                   and self.morning_start <= hod < self.evening_start)
        soon = self.soon_idx >= 0 and math.isfinite(obs[self.soon_idx]) and float(obs[self.soon_idx]) > 0.5
        if float(p.get("st_swing", 0.0)) > 0.5 and self.st_prev_now and not now and V:
            # struct_r1: end of a comfort day; a day with occupants re-labels the swing zones
            if not self.holiday:
                self.st_swing_z[:V] = self.st_cool_cnt[:V] >= float(p.get("st_swing_min", 6.0))
            self.st_cool_cnt[:] = 0.0
        self.st_prev_now = now

        # day boundary bookkeeping
        if hod is not None:
            if self.last_hod is not None and hod < self.last_hod - 12.0:
                self.seen_occ_today = False
                self.holiday = False
            self.last_hod = hod

        # occupancy: the count is affinely normalised with an unknown constant,
        # so detect "somebody is here" relative to the running minimum.
        occupied = False
        if self.occ_idx >= 0 and math.isfinite(obs[self.occ_idx]):
            o = float(obs[self.occ_idx])
            if self.step_count > 1:
                self.occ_min = min(self.occ_min, o)
                occupied = o > self.occ_min + 2e-3
        if occupied:
            self.seen_occ_today = True
            self.holiday = False

        # ---------------- zone temperatures & sensor band ----------------
        if V:
            T = obs[self.z_t] * self.zt_std + self.zt_mean
            T = np.where(np.isfinite(T) & (T > 200.0) & (T < 350.0), T, self.zt_mean)
            sched_lo, sched_hi = (self.comfort_lo, self.comfort_hi) if now else (self.eco_lo, self.eco_hi)
            lo_s = np.where(self.z_lo >= 0, obs[np.maximum(self.z_lo, 0)], sched_lo)
            hi_s = np.where(self.z_hi >= 0, obs[np.maximum(self.z_hi, 0)], sched_hi)
            bad = ~np.isfinite(lo_s) | ~np.isfinite(hi_s) | (hi_s <= lo_s) | (lo_s < 250) | (hi_s > 330)
            lo_s = np.where(bad, sched_lo, lo_s)
            hi_s = np.where(bad, sched_hi, hi_s)
        else:
            T = lo_s = hi_s = np.zeros(0)

        # ---------------- design flow per zone (needed for the request caps) ----------------
        if V:
            design_flow = np.where(self.z_mf >= 0, obs[np.maximum(self.z_mf, 0)], 2.0)
            design_flow = np.where(np.isfinite(design_flow) & (design_flow > 0.01) & (design_flow < 100.0),
                                   design_flow, 2.0)
            F_design = AIR_RHO_CP * design_flow                 # W/K at 100% damper, design pressure
            F_floor = F_design * F_FLOOR100                     # W/K at 100% damper, floor pressure
            F_min = F_design * F_MIN                            # W/K at the damper floor
        else:
            design_flow = F_design = F_floor = F_min = np.zeros(0)

        # ---------------- per-AHU return air, outside air, hot water sensor ----------------
        if V:
            T_ret = (self.member * T[None, :]).sum(axis=1) / self.ahu_zone_count
        else:
            T_ret = np.full(K, self.zt_mean)
        oat = self._oat(obs)
        oat_f = np.where(np.isfinite(oat), oat, T_ret)
        tw_obs = float(obs[self.hws_temp_obs_idx]) if self.hws_temp_obs_idx >= 0 else float("nan")
        if not (math.isfinite(tw_obs) and 273.0 < tw_obs < 400.0):
            tw_obs = 340.0
        kv = max(p["kv_valve"], 1e-4)
        cap1 = WATER_RHO_CP * kv / 3600.0            # W/K of water capacity rate per unit valve per sqrt(bar)

        # ---------------- round 7: per-zone first-order model (RLS) ----------------
        mbr_active = bool(V) and (float(p.get("mbr_on", 0.0)) > 0.5 or float(p.get("mbr_floor", 0.0)) > 0.5)
        mbr_ok = np.zeros(V, dtype=bool)
        if mbr_active:
            Wn = self.mbr_W
            oat_zone0 = oat_f[self.z_a]
            # shift the history: row 0 oldest; the row appended now pairs this step's T with the heat and the
            # mode of the step that just ended (stored at the end of the previous call)
            self.mbr_hT = np.roll(self.mbr_hT, -1, axis=0)
            self.mbr_hX = np.roll(self.mbr_hX, -1, axis=0)
            self.mbr_hQ = np.roll(self.mbr_hQ, -1, axis=0)
            self.mbr_hU = np.roll(self.mbr_hU, -1, axis=0)
            self.mbr_hT[-1] = T
            self.mbr_hX[-1] = T - oat_zone0
            self.mbr_hQ[-2] = self.mbr_q_prev / 1000.0
            self.mbr_hU[-2] = self.zone_mode != self.OCCUPIED
            okw = (np.isfinite(self.mbr_hT).all(axis=0) & np.isfinite(self.mbr_hQ[:-1]).all(axis=0)
                   & self.mbr_hU[:-1].all(axis=0))
            if okw.any():
                y = (self.mbr_hT[-1] - self.mbr_hT[0]) / (Wn * 300.0 / 3600.0)
                x = np.stack([self.mbr_hQ[:-1].mean(axis=0), -self.mbr_hX[:-1].mean(axis=0), np.ones(V)], axis=1)
                lam = float(p.get("mbr_lambda", 0.998))
                Px = np.einsum("vij,vj->vi", self.mbr_P, x)
                den = lam + np.einsum("vi,vi->v", x, Px)
                Kg = Px / den[:, None]
                err = y - np.einsum("vi,vi->v", x, self.mbr_th)
                th_new = self.mbr_th + Kg * err[:, None]
                P_new = (self.mbr_P - np.einsum("vi,vj->vij", Kg, Px)) / lam
                upd = okw & np.isfinite(err) & (np.abs(err) < 20.0)
                self.mbr_th = np.where(upd[:, None], th_new, self.mbr_th)
                self.mbr_P = np.where(upd[:, None, None], P_new, self.mbr_P)
                self.mbr_n = self.mbr_n + (upd & (x[:, 0] > 0.2))
            a_m, k_m = self.mbr_th[:, 0], self.mbr_th[:, 1]
            mbr_ok = (self.mbr_n >= float(p.get("mbr_min_n", 36.0))) & (a_m > 1e-3) & (k_m > 1e-3) & (k_m < 3.0)

        # ---------------- round 9: second-order zone model (mpr) ----------------
        mpr_ok = np.zeros(V, dtype=bool)
        if V and float(p.get("mpr_on", 0.0)) > 0.5:
            oat_z9 = oat_f[self.z_a]
            if self.mpr_T1 is not None and self.mpr_T2 is not None and len(self.mpr_T1) == V:
                y9 = T - self.mpr_T1
                q9 = self.mbr_q_prev / 1000.0 if mbr_active else np.zeros(V)
                x9 = np.stack([self.mpr_T1 - self.mpr_T2, q9, -(self.mpr_T1 - self.mpr_To1), np.ones(V)], axis=1)
                lam9 = float(p.get("mpr_lambda", 0.9995))
                Px9 = np.einsum("vij,vj->vi", self.mpr_P, x9)
                den9 = lam9 + np.einsum("vi,vi->v", x9, Px9)
                K9 = Px9 / den9[:, None]
                err9 = y9 - np.einsum("vi,vi->v", x9, self.mpr_th)
                upd9 = self.mpr_U1 & np.isfinite(err9) & (np.abs(y9) < 2.0) & np.isfinite(x9).all(axis=1)
                th9 = self.mpr_th + K9 * err9[:, None]
                P9 = (self.mpr_P - np.einsum("vi,vj->vij", K9, Px9)) / lam9
                self.mpr_th = np.where(upd9[:, None], th9, self.mpr_th)
                self.mpr_P = np.where(upd9[:, None, None], P9, self.mpr_P)
                self.mpr_n = self.mpr_n + upd9
            a9, b9, c9 = self.mpr_th[:, 0], self.mpr_th[:, 1], self.mpr_th[:, 2]
            mpr_ok = ((self.mpr_n >= float(p.get("mpr_min_n", 288.0))) & (a9 >= 0.0) & (a9 < 0.98)
                      & (b9 > 1e-3) & (c9 > 1e-5) & (c9 < 0.05))
            self.mpr_T2 = self.mpr_T1
            self.mpr_T1 = T.copy()
            self.mpr_To1 = oat_z9.copy()

        # ---------------- mode ----------------
        lo_pre = self.comfort_lo + p["margin_pre"]
        hi_pre = self.comfort_hi - p["margin_pre"]
        safety = max(p["lead_safety"], 1e-3)
        lo_pre_z = lo_pre
        if V and float(p.get("st_swing", 0.0)) > 0.5 and self.st_swing_z[:V].any():
            # struct_r1: a zone that needed cooling in the last workday's comfort hours is only brought to just above
            # the occupied heating target before the workday (heat stored above it would be cooled away later)
            lo_sw = min(lo_pre, self.comfort_lo + p["margin_occ"] + float(p.get("st_swing_k", 0.1)))
            lo_pre_z = np.where(self.st_swing_z[:V], lo_sw, lo_pre)
        if V:
            def_h_z = np.maximum(0.0, lo_pre_z - T)
            def_c_z = np.maximum(0.0, T - hi_pre)
            deficit_h = float(def_h_z.max())
            deficit_c = float(def_c_z.max())
        else:
            def_h_z = def_c_z = np.zeros(0)
            deficit_h = deficit_c = 0.0
        prev_zone_mode = self.zone_mode
        # round 6: recovery rates shifted by how much harder today's lift is than the one they were learned at
        rate_h_eff, rate_c_eff = self.rate_h_z, self.rate_c_z
        slope = float(p.get("rate_oat_slope", 0.0))
        lift_h_now = lift_c_now = None
        if V:
            oat_zone = oat_f[self.z_a]
            lift_h_now = lo_pre - oat_zone
            lift_c_now = oat_zone - hi_pre
            if slope != 0.0:
                dh = np.where(np.isfinite(self.lift_h_z), lift_h_now - self.lift_h_z, 0.0)
                dc = np.where(np.isfinite(self.lift_c_z), lift_c_now - self.lift_c_z, 0.0)
                rate_h_eff = np.clip(self.rate_h_z - slope * dh, p["rate_min_kph"], p["rate_max_kph"])
                rate_c_eff = np.clip(self.rate_c_z - slope * dc, p["rate_min_kph"], p["rate_max_kph"])
        workday_ahead = False
        hours_to_start = 24.0
        hours_to_workday = 24.0
        if now:
            if (hod is not None and hod >= p["holiday_check_h"] and not self.seen_occ_today
                    and self.occ_idx >= 0):
                self.holiday = True
            mode = self.SETBACK if (self.holiday and not occupied) else self.OCCUPIED
            zone_mode = np.full(max(V, 1), mode, dtype=int)
            self.pre_latch[:] = False
        else:
            workday_ahead = False
            hours_to_start = 24.0
            if hod is not None and dow is not None:
                hours_to_start = (self.morning_start - hod) % 24.0
                day_next = dow if hod < self.morning_start else (dow + 1.0) % 7.0
                workday_ahead = day_next < 5.0
                # round 5: hours to the next workday start (skipping weekend days) and whether that
                # workday follows a non-workday (a Monday in the shipped schedule)
                n_skip = 0
                dn = int(round(day_next)) % 7          # the day-of-week feature is integral (Monday = 0)
                while dn >= 5 and n_skip < 7:
                    dn = (dn + 1) % 7
                    n_skip += 1
                hours_to_workday = hours_to_start + 24.0 * n_skip
                self.after_off = (dn - 1) % 7 >= 5
            if per_zone_start:
                lead_z = p["lead_base_h"] + np.maximum(
                    def_h_z / np.maximum(rate_h_eff / safety, 1e-3),
                    def_c_z / np.maximum(rate_c_eff / safety, 1e-3))
                lead_cap = p["lead_max_h"]
                wk_after = bool(self.after_off) and workday_ahead
                if float(p.get("wk_lead_max_h", 0.0)) > 0.0 and wk_after:
                    # round 10: the recovery into a workday that follows a non-workday may start earlier
                    lead_cap = max(lead_cap, float(p["wk_lead_max_h"]))
                lead_z = np.clip(lead_z, p["lead_min_h"], lead_cap)
                if (mbr_active and float(p.get("mbr_on", 0.0)) > 0.5 and mbr_ok.any()
                        and not (float(p.get("wk_no_mbr", 0.0)) > 0.5 and wk_after)):
                    # round 7: the model's recovery time to lo_pre at the preconditioning request cap
                    t_rec = self._mbr_trec(T, lo_pre_z, oat_f[self.z_a], F_design, p)
                    lead_m = np.clip(p["lead_base_h"] + float(p.get("mbr_safety", 1.1)) * t_rec,
                                     p["lead_min_h"], lead_cap)
                    use = mbr_ok & (def_h_z > 0.0) & (def_h_z >= def_c_z)
                    if float(p.get("mbr_lead_cap", 0.0)) > 0.5:
                        # post hoc (round 7 ledger): the latest feasible start -- the model may delay the start,
                        # never advance it past the learned-rate lead; an unreachable model target changes nothing
                        use = use & np.isfinite(t_rec)
                        lead_m = np.minimum(lead_m, lead_z)
                    lead_z = np.where(use, lead_m, lead_z)
                if float(p.get("mpr_on", 0.0)) > 0.5 and mpr_ok.any() and self.mpr_T2 is not None:
                    # round 9: simulate the zone heating at the preconditioning request cap from now on
                    a9, b9, c9, d9 = (self.mpr_th[:, i] for i in range(4))
                    tz = T.copy()
                    dz = T - self.mpr_T2
                    dz = np.where(np.isfinite(dz), np.clip(dz, -0.5, 0.5), 0.0)
                    sc9 = self.scale[:V]
                    steps9 = np.full(V, np.inf)
                    to9 = oat_f[self.z_a]
                    fcp9 = float(p.get("f_cap_pre", F_FLOOR100))
                    for k9 in range(int(p["lead_max_h"] * 12)):
                        qc9 = np.minimum(p["q_cap_w"] * sc9, F_design * fcp9 * np.maximum(DISCHARGE_MAX_K - tz, 0.5)) / 1000.0
                        dz = a9 * dz + b9 * qc9 - c9 * (tz - to9) + d9
                        tz = tz + dz
                        hit = ~np.isfinite(steps9) & (tz >= lo_pre_z)
                        steps9 = np.where(hit, k9 + 1.0, steps9)
                        if np.isfinite(steps9).all():
                            break
                    lead_p = float(p.get("mpr_safety", 1.1)) * steps9 / 12.0
                    use9 = mpr_ok & np.isfinite(lead_p) & (def_h_z > 0.0) & (def_h_z >= def_c_z)
                    lead_z = np.where(use9, np.clip(np.minimum(lead_p, lead_z), p["lead_min_h"], p["lead_max_h"]),
                                      lead_z)
                m_lead = float(p.get("mon_lead_mult", 1.0))
                if m_lead != 1.0 and self.after_off:
                    lead_z = np.clip(lead_z * m_lead, p["lead_min_h"],
                                     max(float(p.get("mon_lead_max_h", 8.0)), p["lead_max_h"]))
                self._lead_dbg = float(lead_z.max())
                lead_rule = workday_ahead & (hours_to_start <= lead_z)
                if float(p.get("pre_latch", 0.0)) > 0.5:
                    # once a zone has started its recovery for the coming workday it holds the
                    # preconditioning target until comfort hours: the air responds within minutes
                    # while the walls take hours, so letting it fall back to SETBACK as soon as the
                    # air is warm would cycle the handler and inflate the learned recovery rate
                    self.pre_latch = ((self.pre_latch | lead_rule | bool(soon)) & workday_ahead
                                      & (hours_to_start <= p["lead_max_h"]))
                    pre_z = self.pre_latch | occupied
                else:
                    pre_z = lead_rule | bool(soon) | occupied
                zone_mode = np.where(pre_z, self.PRECOND, self.SETBACK)
                mode = self.PRECOND if bool(pre_z.any()) else self.SETBACK
            else:
                precondition = bool(soon)
                if workday_ahead:
                    lead = p["lead_base_h"] + max(
                        deficit_h / max(self.rate_h / safety, 1e-3),
                        deficit_c / max(self.rate_c / safety, 1e-3))
                    lead = min(max(lead, p["lead_min_h"]), p["lead_max_h"])
                    self._lead_dbg = lead
                    if hours_to_start <= lead:
                        precondition = True
                mode = self.PRECOND if (precondition or occupied) else self.SETBACK
                zone_mode = np.full(max(V, 1), mode, dtype=int)

        # ---------------- optimum-start learning ----------------
        if per_zone_start:
            # one episode per zone: from entering PRECONDITION (not from OCCUPIED) until the
            # deficit it started with is gone or the mode changes
            start = (zone_mode == self.PRECOND) & ~self.zpre_active & (prev_zone_mode != self.OCCUPIED)
            if start.any():
                self.zpre_active[start] = True
                self.zpre_t0[start] = self.step_count
                self.zpre_def0[start] = np.maximum(def_h_z, def_c_z)[start]
                self.zpre_kind[start] = np.where(def_h_z >= def_c_z, 1, -1)[start]
                self.zpre_reached[start] = -1
                if lift_h_now is not None:
                    self.zpre_lift0[start] = np.where(self.zpre_kind > 0, lift_h_now, lift_c_now)[start]
            act = self.zpre_active
            if act.any():
                cur = np.where(self.zpre_kind > 0, def_h_z, def_c_z)
                newly = act & (self.zpre_reached < 0) & (cur <= 0.1)
                self.zpre_reached[newly] = self.step_count
                ended = act & (zone_mode != self.PRECOND)
                if ended.any():
                    end_step = np.where(self.zpre_reached > 0, self.zpre_reached, self.step_count)
                    hours = np.maximum((end_step - self.zpre_t0) / 12.0, 1.0 / 12.0)
                    recovered = self.zpre_def0 - np.where(self.zpre_reached > 0, 0.0, cur)
                    valid = ended & (self.zpre_def0 >= 0.5)
                    good = valid & (recovered > 0.0)
                    if float(p.get("mon_no_learn", 0.0)) > 0.5 and self.after_off:
                        valid = valid & False
                        good = good & False
                    r_obs = np.clip(recovered / hours, p["rate_min_kph"], p["rate_max_kph"])
                    a = p["rate_ema"]
                    upd_h = good & (self.zpre_kind > 0)
                    upd_c = good & (self.zpre_kind < 0)
                    self.rate_h_z = np.where(upd_h, (1 - a) * self.rate_h_z + a * r_obs, self.rate_h_z)
                    self.rate_c_z = np.where(upd_c, (1 - a) * self.rate_c_z + a * r_obs, self.rate_c_z)
                    # round 6: remember the lift the rates were learned at (EMA with the same weight)
                    self.lift_h_z = np.where(upd_h, np.where(np.isfinite(self.lift_h_z),
                                                             (1 - a) * self.lift_h_z + a * self.zpre_lift0,
                                                             self.zpre_lift0), self.lift_h_z)
                    self.lift_c_z = np.where(upd_c, np.where(np.isfinite(self.lift_c_z),
                                                             (1 - a) * self.lift_c_z + a * self.zpre_lift0,
                                                             self.zpre_lift0), self.lift_c_z)
                    # did not recover at all before comfort hours: assume a much slower zone
                    fail = valid & ~good & (zone_mode == self.OCCUPIED)
                    self.rate_h_z = np.where(fail & (self.zpre_kind > 0),
                                             np.maximum(self.rate_h_z * 0.5, p["rate_min_kph"]), self.rate_h_z)
                    self.rate_c_z = np.where(fail & (self.zpre_kind < 0),
                                             np.maximum(self.rate_c_z * 0.5, p["rate_min_kph"]), self.rate_c_z)
                    self.zpre_active[ended] = False
        else:
            if mode == self.PRECOND and not self.pre_active and self.mode != self.OCCUPIED:
                self.pre_active = True
                self.pre_t0 = self.step_count
                self.pre_deficit0 = max(deficit_h, deficit_c)
                self.pre_kind = 1 if deficit_h >= deficit_c else -1
                self.pre_reached_step = -1
            if self.pre_active:
                cur = deficit_h if self.pre_kind > 0 else deficit_c
                if self.pre_reached_step < 0 and cur <= 0.1:
                    self.pre_reached_step = self.step_count
                if mode != self.PRECOND:
                    if self.pre_deficit0 >= 0.5:
                        end_step = self.pre_reached_step if self.pre_reached_step > 0 else self.step_count
                        hours = max((end_step - self.pre_t0) / 12.0, 1.0 / 12.0)
                        recovered = self.pre_deficit0 - (0.0 if self.pre_reached_step > 0 else cur)
                        if recovered > 0.0:
                            r_obs = recovered / hours
                            r_obs = min(max(r_obs, p["rate_min_kph"]), p["rate_max_kph"])
                            a = p["rate_ema"]
                            if self.pre_kind > 0:
                                self.rate_h = (1 - a) * self.rate_h + a * r_obs
                            else:
                                self.rate_c = (1 - a) * self.rate_c + a * r_obs
                        elif mode == self.OCCUPIED:
                            if self.pre_kind > 0:
                                self.rate_h = max(self.rate_h * 0.5, p["rate_min_kph"])
                            else:
                                self.rate_c = max(self.rate_c * 0.5, p["rate_min_kph"])
                    self.pre_active = False
        self.mode = mode
        self.zone_mode = zone_mode
        if V and float(p.get("mpr_on", 0.0)) > 0.5:
            self.mpr_U1 = zone_mode[:V] != self.OCCUPIED

        # ---------------- band for this mode (per zone) ----------------
        if V:
            is_occ = zone_mode == self.OCCUPIED
            is_pre = zone_mode == self.PRECOND
            is_sb = zone_mode == self.SETBACK
            floor_z = np.full(V, self.eco_lo - p["night_low_ext"])
            ceil_z = np.full(V, self.eco_hi + p["night_high_ext"])
            if per_zone_start and float(p.get("adaptive_floor", 1.0)) > 0.5:
                recover_h = max(p["lead_max_h"] - p["lead_base_h"], 0.5)
                floor_z = np.maximum(floor_z, lo_pre - (rate_h_eff / safety) * recover_h)
                ceil_z = np.minimum(ceil_z, hi_pre + (rate_c_eff / safety) * recover_h)
            wk_lo, wk_hi = float(p.get("wk_ext_lo", 0.0)), float(p.get("wk_ext_hi", 0.0))
            if (wk_lo > 0.0 or wk_hi > 0.0) and not now:
                # round 5: far from the next workday start (weekends) the setback band widens; the extra
                # depth ramps out linearly so the zones are back at the normal floor before any lead
                ramp = np.clip((hours_to_workday - float(p.get("wk_ramp_end_h", 10.0)))
                               / max(float(p.get("wk_ramp_h", 12.0)), 1e-3), 0.0, 1.0)
                floor_z = floor_z - wk_lo * ramp
                ceil_z = ceil_z + wk_hi * ramp
            if (mbr_active and float(p.get("mbr_floor", 0.0)) > 0.5 and not now and workday_ahead
                    and mbr_ok.any()):
                self.mbr_floor_age += 1
                if self.mbr_floor_age >= self.mbr_W or not np.isfinite(self.mbr_floor_cache).any():
                    self.mbr_floor_cache = self._mbr_floor(T, floor_z, lo_pre, oat_f[self.z_a], F_design,
                                                           hours_to_workday, p)
                    self.mbr_floor_age = 0
                fc = self.mbr_floor_cache
                floor_z = np.where(mbr_ok & is_sb & np.isfinite(fc), np.minimum(floor_z, fc), floor_z)
            if float(p.get("wk_free", 0.0)) > 0.5 and not now and self.after_off:
                # round 10: weekend free-float (the LP bound heats almost nothing on Saturdays and Sundays)
                floor_z = np.minimum(floor_z, self.eco_lo - float(p.get("wk_free_ext_k", 30.0)))
            frost = float(p.get("frost_floor_k", 0.0))
            if frost > 0.0:
                # round 6: frost protection (sensible practice; the simulator does not charge for cold zones)
                floor_z = np.maximum(floor_z, frost)
            m_occ_h = p["margin_occ"]
            if float(p.get("margin_occ_heat", -1.0)) >= 0.0:
                cold = np.isfinite(oat[self.z_a]) & (oat[self.z_a] < float(p.get("margin_occ_heat_oat", 273.15)))
                m_occ_h = np.where(cold, float(p["margin_occ_heat"]), p["margin_occ"])
            lo = np.where(is_occ, lo_s + m_occ_h, np.where(is_pre, lo_pre_z, floor_z))
            hi_occ = hi_s - p["margin_occ"]
            new_occ = is_occ & (prev_zone_mode[:V] != self.OCCUPIED)
            self.st_occ_t0[:V] = np.where(new_occ, self.step_count, self.st_occ_t0[:V])
            hand_h = float(p.get("st_hand_c_h", 0.0))
            if hand_h > 0.0:
                # struct_r1: the occupied cooling target starts where preconditioning left the zone and relaxes over
                # hand_h hours, so cooling does not stop at the handover while the fabric and arrivals warm the zone
                r_h = np.clip((self.step_count - self.st_occ_t0[:V]) / (hand_h * 12.0), 0.0, 1.0)
                hi_occ = np.minimum(hi_occ, hi_pre + r_h * (hi_occ - hi_pre))
            hi = np.where(is_occ, hi_occ, np.where(is_pre, hi_pre, ceil_z))
            mid = 0.5 * (lo + hi)
            lo = np.minimum(lo, mid - 0.1)
            hi = np.maximum(hi, mid + 0.1)
        else:
            is_occ = is_pre = is_sb = np.zeros(0, dtype=bool)
            lo = hi = np.zeros(0)

        # ---------------- night flush (round 4 candidate) ----------------
        # Before a workday, a set-back zone that is warmer than the flush target is cooled with
        # outside air when the outside air is at least flush_dt_k colder than the zone: fan only
        # (floor pressure, 100% outside air, cooling coil off), which moves several times more
        # heat per unit of electricity than the chiller does in the afternoon.
        flush_z = np.zeros(V, dtype=bool)
        if V and float(p.get("flush_on", 0.0)) > 0.5 and not now and workday_ahead \
                and hours_to_start <= float(p.get("flush_window_h", 8.0)):
            oat_z = oat[self.z_a]
            tgt = max(hi_pre - float(p.get("flush_depth_k", 1.0)), lo_pre + float(p.get("flush_lo_margin_k", 0.5)))
            ok = is_sb & np.isfinite(oat_z) & (oat_z <= T - float(p.get("flush_dt_k", 3.0)))
            if float(p.get("flush_heat_guard", 0.0)) > 0.5 and (self.heat_timer > 0 or self.season_on):
                # heating season (some zone was heated in the last heat_season_steps): flushed heat
                # would have to be bought back with gas in the morning
                ok = np.zeros(V, dtype=bool)
            self.flush_latch = ok & (self.flush_latch | (T > tgt + 0.3)) & (T > tgt)
            flush_z = self.flush_latch.copy()
        elif V:
            self.flush_latch[:] = False

        # ---------------- zone PI loops on net heat (W into the zone air) ----------------
        # Positive e_h: zone below its heating target; positive e_c: above its cooling
        # target.  Each loop regulates to its own target, so inside the deadband both
        # integrals unwind and the zone free-floats.  Gains and caps scale with the zone
        # area (v6); the cap is also bounded by what the box can deliver at the mode's
        # flow fraction.
        T_pi = T
        lc = float(p.get("lag_comp", 0.0))
        if V and lc != 0.0 and self.T_prev is not None and len(self.T_prev) == V:
            # round 5: the zone sensor lags one step; extrapolate one step for the PI errors
            T_pi = T + lc * np.clip(T - self.T_prev, -1.0, 1.0)
        if V:
            self.T_prev = T.copy()
        e_h = lo - T_pi
        e_c = T_pi - hi
        kq_z = max(p["kq_w_per_k"], 0.0) * self.scale[:V]
        ki_z = max(p["ki_w_per_k_step"], 0.0) * self.scale[:V]
        t_disch_max = DISCHARGE_MAX_K - max(p["disch_margin_k"], 0.0)
        sat_lo, sat_hi_native = self.sat_rng
        if V:
            f_occ = float(p.get("f_cap_occ", 1.0))
            if float(p.get("occ_press_dyn", 0.0)) > 0.5:
                # handler pressure above the floor is expensive twice (fan power and the forced
                # minimum flow of every other zone grows with sqrt(P)), while a zone carries the
                # same one or two occupants whatever its size: allow a zone flow above the
                # floor-pressure flow only in proportion to how far it is past its target
                e_pos = np.maximum(np.maximum(e_h, e_c), 0.0)
                f_lo = float(p.get("f_cap_occ_lo", F_FLOOR100))
                ramp = np.clip((e_pos - float(p.get("occ_press_e0", 0.3)))
                               / max(float(p.get("occ_press_span", 1.0)), 1e-3), 0.0, 1.0)
                f_occ = f_lo + (max(f_occ, f_lo) - f_lo) * ramp
            f_cap = np.where(is_occ, f_occ, float(p.get("f_cap_pre", 0.7)))
            f_cap = np.clip(f_cap, max(float(p.get("f_cap_min", F_FLOOR100)), 0.0), 1.0)
            f_cap_c = f_cap
            if float(p.get("f_cap_pre_c", -1.0)) >= 0.0:
                f_cap_c = np.where(is_pre, np.clip(float(p["f_cap_pre_c"]), F_MIN, 1.0), f_cap)
            late = np.zeros(V, dtype=bool)
            if float(p.get("st_late", 0.0)) > 0.5:
                # struct_r1: late-recovery escalation.  The rise rate observed over the last (up to) 30 min of a
                # zone's preconditioning, extrapolated to the workday start, falls short of its target: let it draw
                # the flow it would get at 06:00 anyway, now, instead of starting the workday below the band.
                self.st_pre_n[:V] = np.where(is_pre, self.st_pre_n[:V] + 1, 0)
                kh = min(6, len(self.st_Thist))
                trig = np.zeros(V, dtype=bool)
                if (kh >= 2 and not now and workday_ahead
                        and hours_to_start <= float(p.get("st_late_h", 2.0))):
                    T_k = self.st_Thist[-kh]
                    if len(T_k) == V:
                        r_obs = (T - T_k) / (kh / 12.0)
                        trig = (is_pre & (self.st_pre_n[:V] > kh) & (e_h > 0.3)
                                & (np.maximum(r_obs, 0.0) * hours_to_start < e_h))
                self.st_late_latch[:V] = is_pre & (self.st_late_latch[:V] | trig) & (e_h > 0.0)
                late = self.st_late_latch[:V].copy()
                self.st_Thist.append(T.copy())
                if late.any():
                    f_cap = np.where(late, np.clip(float(p.get("st_late_fcap", 1.0)), F_MIN, 1.0), f_cap)
            cap_h = F_design * f_cap * np.maximum(t_disch_max - T, 0.5)
            cap_c = F_design * f_cap_c * np.maximum(T - sat_lo, 0.5)
            q_cap_h = np.minimum(max(p["q_cap_w"], 100.0) * self.scale[:V], cap_h)
            q_cap_c = np.maximum(np.minimum(max(p["q_cap_w"], 100.0) * self.scale[:V], cap_c), 100.0)
            if float(p.get("water_cap", 1.0)) > 0.5:
                mixed_est = (RECIRC_MAX * T_ret + (1.0 - RECIRC_MAX) * oat_f)[self.z_a]
                t_sa_est = np.minimum(mixed_est, T - 0.5)
                w_avail = cap1 * math.sqrt(max(min(p["dp_hi_bar"], self.dp_rng[1]), p["dp_lo_bar"], 1e-3))
                coil_max = w_avail * np.maximum(tw_obs - t_sa_est, 1.0)
                cap_h_water = coil_max * np.maximum(t_disch_max - T, 0.5) / np.maximum(t_disch_max - t_sa_est, 1.0)
                q_cap_h = np.minimum(q_cap_h, float(p.get("water_cap_frac", 0.97)) * cap_h_water)
            q_cap_h = np.maximum(q_cap_h, 100.0)
            # Burst heating/cooling against a wide floor for set-back zones: a running handler
            # always pays the ventilation load of every served zone, so the cheap way to bound
            # the drift is to stay off and, when a zone crosses the floor, heat it hard until
            # it is sb_hyst_k inside, then switch off again.  No steady load is held.
            hyst = max(p["sb_hyst_k"], 0.1)
            self.sb_latch_h = is_sb & (self.sb_latch_h | (e_h > 0.0)) & (e_h > -hyst)
            self.sb_latch_c = is_sb & (self.sb_latch_c | (e_c > 0.0)) & (e_c > -hyst)
            sync = float(p.get("sb_sync_k", 0.0))
            if sync > 0.0:
                # round 6: a set-back zone close to its floor joins a heating burst on its handler (the handler
                # and the ventilation of all its zones are paid for that burst anyway)
                burst_ahu = np.zeros(K, dtype=bool)
                np.logical_or.at(burst_ahu, self.z_a, self.sb_latch_h & ~self.sb_latch_c)
                join = is_sb & burst_ahu[self.z_a] & (e_h > -min(sync, hyst)) & ~self.sb_latch_c
                self.sb_latch_h = self.sb_latch_h | join
            q_sb_h = np.minimum(max(p["q_sb_w"], 100.0) * self.scale[:V], q_cap_h)
            q_sb_c = np.minimum(max(p["q_sb_w"], 100.0) * self.scale[:V], q_cap_c)
            heat_sb = self.sb_latch_h & ~self.sb_latch_c
            cool_sb = self.sb_latch_c & ~self.sb_latch_h
            # PI for occupied / preconditioning zones; integrals of set-back zones are reset
            dec = max(p["i_decay_mult"], 0.0)
            dI_h = np.where(e_h > 0.0, ki_z * e_h, dec * ki_z * e_h)
            ki_cz = ki_z * max(float(p.get("ki_c_mult", 1.0)), 0.0)   # round 10 (1.0 = v11 exactly)
            dI_c = np.where(e_c > 0.0, ki_cz * e_c, dec * ki_cz * e_c)
            self.I_h = np.where(is_sb, 0.0, np.clip(self.I_h + dI_h, 0.0, q_cap_h))
            self.I_c = np.where(is_sb, 0.0, np.clip(self.I_c + dI_c, 0.0, q_cap_c))
            q_h_pi = np.clip(kq_z * np.maximum(e_h, 0.0) + self.I_h, 0.0, q_cap_h)
            if float(p.get("mpr_on", 0.0)) > 0.5 and float(p.get("mpr_burst", 1.0)) > 0.5:
                q_h_pi = np.where(is_pre & (e_h > 0.3), q_cap_h, q_h_pi)
            if float(p.get("pre_burst_h", 0.0)) > 0.5:
                # round 5: recover at the floor-pressure cap from the start (the other zones' forced
                # ventilation is paid per minute of handler run time, so a faster recovery is cheaper)
                q_h_pi = np.where(is_pre & (e_h > float(p.get("pre_burst_e0", 0.3))), q_cap_h, q_h_pi)
            if late.any():
                q_h_pi = np.where(late, q_cap_h, q_h_pi)
            q_c_pi = np.clip(kq_z * max(float(p.get("kq_c_mult", 1.0)), 0.0) * np.maximum(e_c, 0.0) + self.I_c,
                             0.0, q_cap_c)
            obr = float(p.get("obr_on", 0.0)) > 0.5
            if obr:
                # round 9: occupied band riding.  Heating regime = occupied, not near the cooling target and
                # not cooling; there the PI is replaced by bursts at the cap (q_cap_h at the occupied flow cap).
                reg = is_occ & (T < hi - float(p.get("obr_cool_guard_k", 1.0))) & (self.I_c <= 0.0) & (e_c <= 0.0)
                start = reg & (T < lo_s + float(p.get("obr_start_k", 0.3)))
                stop = T > lo_s + float(p.get("obr_stop_k", 0.9))
                latch = reg & ~stop & (self.obr_latch | start)
                burst_ahu = np.zeros(K, dtype=bool)
                np.logical_or.at(burst_ahu, self.z_a, latch)
                latch = latch | (reg & burst_ahu[self.z_a] & (T < lo_s + float(p.get("obr_join_k", 0.6))))
                self.obr_latch = latch
                self.obr_regime = reg
                self.I_h = np.where(reg, 0.0, self.I_h)
                q_h_pi = np.where(reg, np.where(latch, q_cap_h, 0.0), q_h_pi)
            else:
                self.obr_regime = np.zeros(V, dtype=bool)
            # a zone cannot be heated and cooled at once: the larger request wins
            heat_pi = ~is_sb & (q_h_pi > q_c_pi) & (q_h_pi > p["q_on_w"])
            cool_pi = ~is_sb & (q_c_pi >= q_h_pi) & (q_c_pi > p["q_on_w"])
            heat_zone = heat_sb | heat_pi
            cool_zone = cool_sb | cool_pi
            q_h = np.where(heat_sb, q_sb_h, np.where(heat_pi, q_h_pi, 0.0))
            q_c = np.where(cool_sb, q_sb_c, np.where(cool_pi, q_c_pi, 0.0))
            if float(p.get("st_swing", 0.0)) > 0.5:
                self.st_cool_cnt[:V] += is_occ & cool_zone
        else:
            q_h = q_c = np.zeros(0)
            heat_zone = cool_zone = np.zeros(0, dtype=bool)

        # ---------------- per-AHU aggregates ----------------
        if V:
            H = np.where(self.ahu_has_zones, (self.member * q_h[None, :]).max(axis=1), 0.0)
            C = np.where(self.ahu_has_zones, (self.member * q_c[None, :]).max(axis=1), 0.0)
        else:
            H = C = np.zeros(K)
        D = np.maximum(H, C)
        # night flush: a handler flushes only when none of its zones asks for heat or cooling
        fl_ahu = np.zeros(K, dtype=bool)
        if V and flush_z.any():
            np.logical_or.at(fl_ahu, self.z_a, flush_z)
            fl_ahu &= D <= 0.0
        flush_eff = flush_z & fl_ahu[self.z_a] if V else np.zeros(0, dtype=bool)

        # run / stop with dwell hysteresis
        want_on = (D > p["q_on_w"]) | fl_ahu
        want_off = (D <= 0.0) & ~fl_ahu
        can_switch = self.dwell >= np.where(self.ahu_on, p["min_on_steps"], p["min_off_steps"])
        turn_on = ~self.ahu_on & want_on & can_switch
        turn_off = self.ahu_on & want_off & can_switch
        new_on = (self.ahu_on | turn_on) & ~turn_off
        self.dwell = np.where(new_on == self.ahu_on, self.dwell + 1, 0)
        self.ahu_on = new_on
        on = self.ahu_on

        # ---------------- supply air temperature reset (per AHU) ----------------
        # Cooling: the coldest supply any cooling zone needs to be served at 100%
        # damper and floor pressure, bounded to [285, sat_cool_hi]; floored when the
        # handler also has heating zones.  Heating only: coil off (SAT at maximum).
        sat_hi = min(max(p["sat_cool_hi"], sat_lo), sat_hi_native)
        sat = np.full(K, sat_hi_native)
        if V:
            t_need = np.where(cool_zone, T - q_c / np.maximum(F_floor, 1.0), np.inf)
            t_need_ahu = np.full(K, np.inf)
            np.minimum.at(t_need_ahu, self.z_a, t_need)
            has_cool = C > 0.0
            s = np.clip(t_need_ahu, sat_lo, sat_hi)
            # the mixed-handler floor is independent of the cooling ceiling, so a cold SAT for
            # cooling-only handlers (v5) does not remove the reheat protection on mixed handlers
            mix_floor = min(p["sat_mix_floor"], sat_hi_native)
            if float(p.get("sat_mix_relief", 0.0)) > 0.5:
                # round 4: on a handler that heats and cools, go below the mixed floor only as far
                # as the most demanding cooling zone needs to be served at floor pressure (raising
                # the pressure instead costs fan power and inflates every zone's minimum flow)
                relief_min = max(float(p.get("sat_mix_relief_min", sat_lo)), sat_lo)
                floor_k = np.clip(t_need_ahu, relief_min, max(mix_floor, relief_min))
            else:
                floor_k = np.full(K, mix_floor)
            s = np.where(H > 0.0, np.maximum(s, floor_k), s)
            sat = np.where(has_cool, s, sat_hi_native)
            if float(p.get("st_msat", 0.0)) > 0.5:
                # struct_r1: a handler with heating and cooling zones gets the supply air that minimises this step's
                # modelled cost: chiller (mixed -> supply, COP 3.5) + supply/exhaust fans, both priced at st_msat_r
                # per J of gas, + reheat gas of the heating zones (their flow is set by the discharge cap), with a
                # steep penalty on cooling a zone could not receive at its design flow
                r_e = float(p.get("st_msat_r", 2.1))
                mixed_a = RECIRC_MAX * T_ret + (1.0 - RECIRC_MAX) * oat_f
                for a in range(K):
                    if not (has_cool[a] and H[a] > 0.0):
                        continue
                    m_a = float(min(mixed_a[a], sat_hi_native))
                    if not math.isfinite(m_a) or m_a <= sat_lo + 0.25:
                        continue
                    mz = self.z_a == a
                    Tz, Fd, Fm = T[mz], F_design[mz], F_min[mz]
                    qc = np.where(cool_zone[mz], q_c[mz], 0.0)
                    qh = np.where(heat_zone[mz], q_h[mz], 0.0)
                    Fh = np.where(qh > 0.0, qh / np.maximum(t_disch_max - Tz, 0.5), 0.0)
                    best_s, best_c = float(sat[a]), math.inf
                    for s_ in list(np.arange(sat_lo, m_a, 0.5)) + [m_a]:
                        t = min(float(s_), m_a)
                        dTc = np.maximum(Tz - t, 0.5)
                        Fc = np.where(qc > 0.0, qc / dTc, 0.0)
                        unmet = float((np.maximum(Fc - Fd, 0.0) * dTc).sum())
                        F = np.clip(np.maximum(np.maximum(Fc, Fh), Fm), Fm, Fd)
                        gas = float(np.where(qh > 0.0, qh + F * np.maximum(Tz - t, 0.0), 0.0).sum())
                        chill = float(F.sum()) * max(m_a - t, 0.0) / 3.5
                        fmax = max(float((F / Fd).max()), F_FLOOR100)
                        P = min(max(P_MAX_PA * fmax ** 2, P_FLOOR_PA), P_MAX_PA)
                        fan = float(F.sum()) / AIR_RHO_CP * P * (2.0 - RECIRC_MAX) / 0.9
                        cost = r_e * (chill + fan + 10.0 * unmet) + gas
                        if cost < best_c - 1e-9:
                            best_s, best_c = t, cost
                    sat[a] = best_s
        sat = np.clip(sat, sat_lo, sat_hi_native)

        # ---------------- economiser ----------------
        recirc = np.full(K, RECIRC_MAX)
        if p["econ_on"] > 0.5:
            cooling = on & (C > 0.0) & (H <= 0.0) & np.isfinite(oat)
            adv = T_ret - oat_f
            econ = cooling & (adv > P_FLOOR_PA * ECON_K_PER_PA)
            with np.errstate(divide="ignore", invalid="ignore"):
                f_oa = np.where(econ, (T_ret - sat) / np.where(adv > 0, adv, 1.0), 0.0)
            f_oa = np.clip(f_oa, 1.0 - RECIRC_MAX, 1.0)
            recirc = np.where(econ, 1.0 - f_oa, RECIRC_MAX)
        recirc = np.where(on & fl_ahu, 0.0, recirc)   # night flush: all outside air
        self.recirc_prev = recirc

        # estimated coil-entering (supply) air temperature per AHU
        mixed = recirc * T_ret + (1.0 - recirc) * oat_f
        t_sa = np.minimum(mixed, sat)

        # ---------------- hot water temperature ----------------
        any_heat_now = bool(V) and bool((heat_zone & on[self.z_a]).any())
        if any_heat_now or (mode == self.PRECOND and deficit_h > 0.0):
            self.heat_timer = int(p["heat_season_steps"])
        else:
            self.heat_timer = max(self.heat_timer - 1, 0)
        hold = int(float(p.get("tw_hold_steps", 0.0)))
        if hold > 0:
            # round 10: keep the loop hot after the last heating demand without lengthening the heating season
            # (heat_timer also gates the night-flush guard and the set-back reheat)
            if any_heat_now or (mode == self.PRECOND and deficit_h > 0.0):
                self.tw_timer = hold
            else:
                self.tw_timer = max(self.tw_timer - 1, 0)
        hold_ok = hold > 0 and self.tw_timer > 0
        thr = float(p.get("tw_hold_oat_k", 0.0))
        ext = int(float(p.get("season_ext_steps", 0.0)))
        thr_s = float(p.get("season_oat_k", 0.0))
        if (hold > 0 and thr > 0.0) or (ext > 0 and thr_s > 0.0):
            # round 10: per-step minimum outside air over the last 24 h (one entry per step)
            fin = np.isfinite(oat)
            if fin.any():
                self.oat_min_hist.append(float(oat[fin].min()))
        oat24 = min(self.oat_min_hist) if self.oat_min_hist else float("inf")
        if hold > 0 and thr > 0.0:
            # round 10: no hold in a warm spell (no step of the last 24 h below thr)
            hold_ok = hold_ok and oat24 < thr
        if ext > 0:
            # round 10: round 9's long heating season, only while some step of the last 24 h was below season_oat_k
            if any_heat_now or (mode == self.PRECOND and deficit_h > 0.0):
                self.season_timer = ext
            else:
                self.season_timer = max(self.season_timer - 1, 0)
            self.season_on = self.season_timer > 0 and (thr_s <= 0.0 or oat24 < thr_s)
        tw = p["tw_hot_k"] if (self.heat_timer > 0 or self.season_on or hold_ok) else p["tw_idle_k"]
        tw = float(np.clip(tw, self.tw_rng[0], self.tw_rng[1]))

        # ---------------- zone flows, dampers, pressure, valves ----------------
        valve = np.zeros(V)
        dp_bar = p["dp_lo_bar"]
        pump_on = False
        if V:
            t_sa_z = t_sa[self.z_a]
            zone_on = on[self.z_a]
            # cooling: flow so that F * (T - t_sa) = q_c
            dT_c = np.maximum(T - t_sa_z, 0.5)
            F_cool = np.where(cool_zone, q_c / dT_c, 0.0)
            # heating: the discharge limit bounds the heat per unit flow
            dT_h = np.maximum(t_disch_max - T, 0.5)
            F_heat = np.where(heat_zone, q_h / dT_h, 0.0)
            F_req = np.maximum(np.maximum(F_cool, F_heat), F_min)
            F_req = np.where(flush_eff, np.maximum(F_req, F_floor), F_req)   # flush: 100% damper at the floor
            F_req = np.where(zone_on, F_req, F_min)
            f_req = np.clip(F_req / F_design, F_MIN, 1.0)   # fraction of design flow
            f_max = np.zeros(K)
            np.maximum.at(f_max, self.z_a, f_req)
            f_max = np.maximum(f_max, F_FLOOR100)
            p_cmd = np.clip(P_MAX_PA * f_max ** 2, P_FLOOR_PA, min(P_MAX_PA, self.p_rng[1]))
            p_cmd = np.where(on, p_cmd, P_FLOOR_PA)
            f_real = np.sqrt(p_cmd / P_MAX_PA)
            damper = np.clip(f_req / f_real[self.z_a], DAMPER_FLOOR, 1.0)
            damper = np.where(zone_on, damper, 0.0)
            # actual flow delivered at the commanded damper and pressure
            F_act = F_design * np.clip(damper, DAMPER_FLOOR, 1.0) * f_real[self.z_a]
            # reheat: discharge temperature that delivers q_h net, then the water
            # capacity rate that produces it, then the valve at the chosen dP
            # round 4: in heating season, a set-back zone below its preconditioning target that
            # sits on a running handler gets its forced minimum flow reheated to its own
            # temperature (1 J of gas per J now instead of ~2 J per J in the ventilated recovery)
            neutral = np.zeros(V, dtype=bool)
            if float(p.get("sb_neutral", 0.0)) > 0.5 and (self.heat_timer > 0 or self.season_on):
                neutral = zone_on & is_sb & ~heat_zone & ~cool_zone & (T < lo_pre_z) & (T > t_sa_z + 0.5)
                if float(p.get("sb_neutral_cold", 0.0)) > 0.5:
                    # only when the outside air is below the zone's setback floor, i.e. the zone will
                    # certainly need heat before the workday (in mild weather it warms by itself)
                    neutral &= np.isfinite(oat[self.z_a]) & (oat[self.z_a] <= floor_z)
            boost = np.zeros(V, dtype=bool)
            if (float(p.get("sb_boost", 0.0)) > 0.5 and neutral.any() and not now and workday_ahead
                    and hours_to_start <= float(p.get("sb_boost_h", 8.0))):
                boost = neutral & (T < lo_pre_z - float(p.get("sb_boost_gap", 0.5)))
                if float(p.get("sb_boost_cold", 0.0)) > 0.5:
                    # round 6 (added after c01's first screen shard): in mild weather a set-back zone may not
                    # need much recovery heat, and heat stored early leaks before 06:00
                    boost &= np.isfinite(oat[self.z_a]) & (oat[self.z_a] <= floor_z)
            if float(p.get("obr_on", 0.0)) > 0.5:
                # round 9: forced air of heating-regime occupied zones on a running handler goes in at the cap
                occ_boost = (zone_on & self.obr_regime & ~heat_zone & ~cool_zone & (T > t_sa_z + 0.5)
                             & (T < lo_s + float(p.get("obr_boost_k", 0.9))))
                neutral = neutral | occ_boost
                boost = boost | occ_boost
            bfk = float(p.get("sb_boost_floor_k", 0.0))
            if bfk > 0.0 and neutral.any():
                # round 6: a zone near its setback floor will need a burst soon; heat it now through the forced
                # air that this handler run already ventilates it with (1 J of gas per J delivered)
                boost = boost | (neutral & (T < floor_z + bfk))
            if any_heat_now or neutral.any():
                t_disch = np.clip(T + q_h / np.maximum(F_act, 1.0), t_sa_z, t_disch_max)
                t_disch = np.where(neutral, np.minimum(T, t_disch_max), t_disch)
                t_disch = np.where(boost, t_disch_max, t_disch)
                gap = np.maximum(tw_obs - t_sa_z, 1.0)
                w_need = np.where((heat_zone | neutral) & zone_on,
                                  F_act * np.clip((t_disch - t_sa_z) / gap, 0.0, 1.0), 0.0)
                w_max = float(w_need.max())
                target = max(p["valve_target"], 0.05)
                dp_bar = (w_max / (target * cap1)) ** 2
                dp_bar = float(np.clip(dp_bar, p["dp_lo_bar"], min(p["dp_hi_bar"], self.dp_rng[1])))
                valve = np.clip(w_need / (cap1 * math.sqrt(dp_bar)), 0.0, 1.0)
                pump_on = bool(valve.max() > 0.0)
            if mbr_active:
                # round 7: the policy's own estimate of the net heat each zone receives this step
                td_eff = t_sa_z
                if any_heat_now or neutral.any():
                    td_eff = np.where((heat_zone | neutral | boost) & zone_on, t_disch, t_sa_z)
                self.mbr_q_prev = np.where(zone_on, F_act * (td_eff - T), 0.0)
        else:
            damper = np.zeros(0)
            p_cmd = np.full(K, P_FLOOR_PA)

        # ---------------- scatter into the action vectors ----------------
        for k in range(K):
            cont[self.ahu_temp_idx[k]] = self._norm(float(sat[k]), *self.sat_rng)
            if self.ahu_press_idx[k] >= 0:
                cont[self.ahu_press_idx[k]] = self._norm(float(p_cmd[k]), *self.p_rng) if on[k] else -1.0
            if self.ahu_recirc_idx[k] >= 0:
                cont[self.ahu_recirc_idx[k]] = 2.0 * float(recirc[k]) - 1.0
            disc[k] = 1 if on[k] else 0
        if V:
            md_ = self.z_d >= 0
            cont[self.z_d[md_]] = 2.0 * damper[md_] - 1.0
            mv_ = self.z_v >= 0
            cont[self.z_v[mv_]] = 2.0 * valve[mv_] - 1.0
        for s_ in self.orphan_slots:
            cont[s_] = -1.0
        twr = float(p.get("tw_ramp_k", 0.0))
        if twr > 0.0 and pump_on:
            # round 9: never bill the loop for a setpoint the tank cannot reach this step, never go below the tank
            tw = float(np.clip(tw, tw_obs, tw_obs + twr))
            tw = float(np.clip(tw, self.tw_rng[0], self.tw_rng[1]))
        if self.water_temp_idx >= 0:
            cont[self.water_temp_idx] = self._norm(tw, *self.tw_rng)
        if self.water_press_idx >= 0:
            cont[self.water_press_idx] = self._norm(dp_bar, *self.dp_rng) if pump_on else -1.0
        if 0 <= self.water_disc_idx < self.n_discrete:
            disc[self.water_disc_idx] = 1 if pump_on else 0

        cont = np.nan_to_num(cont, nan=0.0, posinf=1.0, neginf=-1.0)
        np.clip(cont, -1.0, 1.0, out=cont)

        if self._dbg is not None:
            try:
                rh = float(self.rate_h_z.min()) if per_zone_start else self.rate_h
                rc = float(self.rate_c_z.min()) if per_zone_start else self.rate_c
                self._dbg.write(
                    f"{self.step_count},{-1 if hod is None else hod:.3f},{-1 if dow is None else dow:.2f},"
                    f"{int(now)},{int(soon)},{int(occupied)},{mode},{int(self.holiday)},"
                    f"{self._lead_dbg:.2f},{rh:.2f},{rc:.2f},"
                    f"{deficit_h:.2f},{deficit_c:.2f},"
                    f"{T.min() if V else 0:.2f},{T.max() if V else 0:.2f},{T.mean() if V else 0:.2f},"
                    f"{int(heat_zone.sum())},{int(cool_zone.sum())},{p_cmd.max():.0f},{int(on.sum())},"
                    f"{tw:.0f},{dp_bar:.1f},{int(pump_on)},{valve.max() if V else 0:.2f},"
                    f"{damper.max() if V else 0:.2f},{sat.min():.1f},{recirc.min():.2f},"
                    f"{int((zone_mode == self.PRECOND).sum())}\n")
            except (OSError, ValueError):
                pass
        return {"continuous_action": cont.astype(np.float32), "discrete_action": disc.astype(np.int32)}
