# Output Schema Definitions

The simulator emits three parquet tables via SimulationResult.save().

## 1) node_timeseries.parquet

Granularity: one row per home or battery node per minute.

Columns:

- timestamp: Simulation timestamp.
- node_id: Integer node identifier.
- node_type: Graph node type for this record (`home` or `battery`).
- battery_node_id: Linked battery node id for a home row; null for battery rows.
- load_kw: Household non-EV demand after profile shaping. Zero for battery rows.
- pv_kw: PV generation (raw/synthetic or NSRDB-replaced). Zero for battery rows.
- ev_kw: EV charging demand in the current minute. Zero for battery rows.
- battery_power_kw: Positive=discharge, negative=charge. Repeated on both home and battery rows.
- battery_soc_kwh: Battery state of charge. Zero for homes without batteries.
- net_grid_kw: Net import/export request at the node. For battery rows, positive means charging import and negative means discharge export.
- voltage_pu: Per-unit voltage estimate after feeder drop. Battery rows inherit the attached home voltage.
- line_flow_kw: Downstream branch flow seen at node edge. Battery rows use battery edge flow.
- unserved_load_kw: Demand not served during outage/islanding. Zero for battery rows.
- islanding_active: True when outage-triggered island mode is active.
- maintenance_mode: True during planned maintenance windows.
- active_faults: Comma-separated list of active fault types.

## 2) transformer_timeseries.parquet

Granularity: one row per minute.

Columns:

- timestamp: Simulation timestamp.
- feeder_total_kw: Net feeder real power at transformer root.
- transformer_loading_pu: feeder_total_kw / rated_power_kw.
- max_branch_loading_pu: Max line current loading ratio across edges.
- top_oil_rise_c: Modeled top-oil rise above ambient.
- hotspot_rise_c: Modeled hotspot rise above top-oil.
- hottest_spot_temp_c: Ambient + top-oil + hotspot.
- aging_acceleration: Per-minute accelerated aging factor.
- cumulative_loss_of_life_hours: Integrated equivalent aging hours.
- overload_event_count: Cumulative count of overload events.
- grid_available: False during outage/maintenance faults.
- islanding_triggered: True only for real outage-triggered islanding.
- maintenance_mode: True during planned maintenance faults.

## 3) event_log.parquet

Event stream for active scenario conditions and overload detections.

Columns:

- timestamp: Event timestamp.
- event_type: Event identifier.
- details: JSON string with runtime context such as maintenance mode and islanding permissions.

## 4) metadata.json

Scenario and topology metadata saved alongside the parquet outputs.

Fields include:

- num_homes
- start_time
- end_time
- time_step_minutes
- city
- year
- adoption
- load_profile
- transformer
- graph_total_nodes
- graph_home_nodes
- graph_battery_nodes
- home_assets

## Overload Event Logic

A transformer overload event is counted when loading exceeds 1.2 pu for
5 consecutive minutes, with a 10-minute cooldown before counting another.

## Integration Notes

- Use node_timeseries for RL/GNN observation generation.
- Use transformer_timeseries for safety and KPI tracking.
- Use event_log for supervised labels and scenario diagnostics.
- Use `battery_node_id` to join each home record to its explicit battery node rows.
