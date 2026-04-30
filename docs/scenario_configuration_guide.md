# Scenario Configuration Guide

This guide defines the simulator scenario JSON format consumed by
simulator/config.py.

## File Location

Default scenario:

- scenarios/phase1_default.json

## Top-Level Keys

- start_time: ISO timestamp, inclusive start.
- end_time: ISO timestamp, exclusive end.
- time_step_minutes: Must be 1 for Phase 1.
- random_seed: Integer seed for reproducibility.
- neighborhood: Network and line settings.
- adoption: Solar, battery, EV penetration ratios.
- load_profile: Demand and PV profile controls.
- transformer: Thermal model constants.
- faults: Array of fault events.

## Neighborhood Block

- num_homes: 10-500 supported.
- line_resistance_ohm_per_km: Line resistance basis.
- min_edge_length_m and max_edge_length_m: Randomized edge lengths.
- base_voltage_v: Nominal single-phase voltage.
- line_ampacity_a: Used for branch loading pu metric.

## Adoption Block

Each value must be in [0, 1].

- solar_ratio
- battery_ratio
- ev_ratio

## Load Profile Block

- target_daily_kwh_min and target_daily_kwh_max: Bounds for household scaling.
- gaussian_noise_sigma: Minute-level stochastic variation.
- afternoon_ac_gain: Controls AC-driven afternoon surge.
- afternoon_ac_threshold_c: AC surge only ramps once modeled ambient exceeds this temperature.
- evening_peak_gain: Controls cooking and lighting peak.
- festival_spike_gain: Extra demand on configured festival days.
- cricket_spike_gain: Extra demand during marked cricket match windows and weekend evening watch spikes.
- use_pecan_profiles: If true, use Pecan-derived household traces.
- pecan_profile_file: Path to wired Pecan data with required columns.
- replace_solar_with_nsrdb: If true, substitute PV with city NSRDB profile.
- nsrdb_data_root: Root path for NSRDB city/year files.
- city and year: NSRDB city and year selector plus city profile selector.
- festival_dates: List of YYYY-MM-DD dates for event spikes.
- cricket_match_dates: List of YYYY-MM-DD dates to explicitly trigger cricket-match evening demand.

## Transformer Block

IEEE C57.91-inspired parameters:

- rated_power_kw
- ambient_temp_c
- delta_theta_to_r
- delta_theta_hs_r
- tau_to_min
- tau_w_min
- r_loss_ratio
- n_exp
- m_exp

## Fault Definitions

Each fault entry includes:

- event_type: overload, solar_dropout, grid_outage, planned_maintenance
- name: Human-friendly identifier
- start and end: ISO timestamps
- target: all or random_cluster
- params: Type-specific parameters

### overload params

- load_multiplier (example: 1.45)
- target_ratio (example: 0.45)

### solar_dropout params

- drop_fraction (example: 0.8)

### grid_outage and planned_maintenance params

- Empty object allowed.

## Planned Maintenance Behavior

planned_maintenance intentionally disables grid import but sets maintenance mode,
preventing false automated outage response logic.

## Recommended Presets

- Small debug run: `scenarios/phase1_debug.json`
- Baseline run: `scenarios/phase1_default.json`
- Stress run: `scenarios/phase1_stress.json`
