# Baseline Protocol and Success Metrics

This document locks the baseline controller definitions, metric formulas, experiment matrix, and acceptance criteria used to evaluate model performance.

## Baseline Controllers

### Baseline-0 (B0): No Control

- Solar serves local load first.
- Excess solar is exported at fixed feed-in tariff.
- Battery is fully disabled.
- EV charges immediately at maximum allowed charging rate.
- No response to grid or neighborhood signals.

### Baseline-1 (B1): Rule-Based (Deterministic)

#### Battery rules

- Charge when `solar > load` and `SoC < 90%`.
- Discharge when `load > solar` and `SoC > 20%`.

#### EV rules

- Charge only between `22:00-06:00` OR when `price < ev_price_threshold`.

#### P2P rules

- Sell when `market_price > 0.8 x grid_tariff`.
- Buy when `market_price < 1.1 x grid_tariff`.

No adaptive or learned logic is allowed in B1.

### Baseline-2 (B2): MPC-lite

- Rolling horizon: `30 minutes`.
- Re-optimization frequency: every simulation timestep.

#### Objective

Minimize:

`grid_cost + lambda_1 * curtailment + lambda_2 * battery_degradation`

#### Constraints

- SoC lower and upper bounds.
- EV energy deadline constraint.
- Battery charge and discharge power limits.
- Grid import and export bounds.
- Device-level and feeder-level power limits.

Use linear programming where feasible for speed and stability.

## Success Metrics (Locked Definitions)

### 1. Solar Curtailment (%)

Per timestep:

`P_curtailed(t) = max(0, solar(t) - load(t) - battery_charge(t) - export(t))`

Aggregate:

`Curtailment(%) = (sum(P_curtailed) / sum(solar)) * 100`

### 2. Peak Demand Reduction (%)

- Peak is defined as `max(grid_import)` over the full simulation window.

`PeakReduction(%) = ((Peak_B0 - Peak_controller) / Peak_B0) * 100`

### 3. Transformer Overload Events

- Trigger event when `loading > 1.2 pu` for `5 consecutive timesteps`.
- Cooldown after event: `10 minutes` before counting a new event.

### 4. Cost Reduction (%)

Total cost:

`C = grid_import_cost - p2p_revenue`

Include:

- EV charging energy cost.
- Optional battery inefficiency penalty (if enabled, report explicitly).

`CostReduction(%) = ((C_B0 - C_controller) / C_B0) * 100`

### 5. Island Switchover Time (seconds)

- Outage detection condition: `voltage < 0.9 pu`.
- Island stable condition: all nodes disconnected from central grid and local frequency stable.

`SwitchoverTime = t_island_stable - t_outage_detect`

### 6. P2P Settlement Latency

Track timeline:

- `submit_time`
- `match_time`
- `settle_time`

Report:

- p50
- p95
- p99

## Experiment Matrix

Run each controller (B0, B1, B2) across:

- Cities: Bengaluru, Kochi, Delhi, Chennai, Hyderabad.
- Day types: weekday, weekend.
- Seasonal windows: summer, monsoon, winter representative weeks.
- Scales: 10, 50, 200 nodes.
- Penetration sets:
  - Default: solar 40%, battery 30%, EV 25%.
  - Stress: solar 60%, battery 20%, EV 40%.
- Fault scenarios:
  - Sudden solar drop event (cloud transient).
  - Transformer overload stress.
  - Grid outage and island transition.

Minimum random seeds per scenario: `5`.

## Reporting Format

| City | Controller | Curtailment (%) | Peak Reduction (%) | Cost Reduction (%) | Overload Events | Latency p95 |
|------|------------|-----------------|--------------------|--------------------|-----------------|-------------|
| BLR  | B0         | mean ± std      | mean ± std         | mean ± std         | count           | value       |

Also include:

- Overall aggregate mean across all cities and seeds.
- Standard deviation and coefficient of variation (`std/mean`) for key metrics.

## Acceptance Criteria

Baseline suite is considered locked when all conditions pass:

- Coefficient of variation for key metrics is `< 0.1`.
- B2 is not worse than B1 on any key metric.
- Controller ordering remains stable across seeds and cities.
