# VayuGrid Grid Simulator

This package is the minute-resolution neighborhood simulator that the training,
evaluation, and demo stack should use as the source of truth.

## What It Models

- A radial residential feeder with a transformer root node, home nodes, and
  explicit battery nodes.
- Resistive line segments with per-edge voltage drop and branch loading.
- Household combinations of solar, behind-the-meter storage, and EV charging.
- A 1-minute simulation clock that emits node and transformer state every step.
- Transformer thermal stress using an IEEE C57.91-inspired top-oil and hottest
  spot model with accelerated aging.
- Fault injection for overload, solar dropout, grid outage, and planned
  maintenance.

## India-Specific Behavior

- Household demand is scaled to roughly 5-8 kWh/day/home.
- City profiles are built in for `bangalore`, `chennai`, `hyderabad`, `delhi`,
  `kochi`, and `mumbai`.
- Afternoon AC surge only ramps when modeled ambient temperature exceeds the
  configured threshold, default `35 C`.
- Evening cooking and lighting peak is explicit.
- Festival bursts and cricket-match demand spikes are explicit.
- EV charging follows Indian home-arrival behavior with strong overnight bias
  and Tata Nexon-class charging rates around `2.2-3.3 kW`.

## Data Sources and Profile Modes

- Synthetic India mode:
  - Generates native residential, PV, and EV traces from the city profile.
- Pecan-backed mode:
  - Starts from Pecan Street 1-minute traces.
  - Scales daily household energy down to Indian usage bands.
  - Reapplies Indian load shaping so the result still reflects local behavior.
- NSRDB replacement:
  - If enabled, PV is replaced with city-specific NSRDB/Himawari irradiance.

## Scenario Presets

- [scenarios/phase1_debug.json](/home/varshith/VayuGrid/scenarios/phase1_debug.json)
  for a fast 10-home smoke run.
- [scenarios/phase1_default.json](/home/varshith/VayuGrid/scenarios/phase1_default.json)
  for a baseline 30-home run.
- [scenarios/phase1_stress.json](/home/varshith/VayuGrid/scenarios/phase1_stress.json)
  for a 300-home stress case.

## Quick Start

Run the default example:

```bash
/home/varshith/VayuGrid/.venv/bin/python examples/example_instantiation.py
```

Run a chosen scenario directly:

```python
from simulator import load_simulator_config
from simulator.simulator import GridSimulator

config = load_simulator_config("scenarios/phase1_default.json")
result = GridSimulator(config).run()
result.save("outputs/phase1_default")
```

## What Charithra Needs

1. Pick a scenario JSON from `scenarios/`.
2. Instantiate `GridSimulator(config).run()`.
3. Read:
   - `node_timeseries.parquet`
   - `transformer_timeseries.parquet`
   - `event_log.parquet`
   - `metadata.json`

`metadata.json` now includes the resolved load-profile block, transformer block,
and per-home asset inventory so downstream training can reconstruct topology and
device mix without re-sampling.

## Output References

- [docs/output_schema_definitions.md](/home/varshith/VayuGrid/docs/output_schema_definitions.md)
- [docs/scenario_configuration_guide.md](/home/varshith/VayuGrid/docs/scenario_configuration_guide.md)
