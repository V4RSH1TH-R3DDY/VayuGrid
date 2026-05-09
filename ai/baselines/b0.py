from __future__ import annotations

from simulator.config import SimulatorConfig
from simulator.simulator import GridSimulator


class B0Controller:
    """Baseline-0: No control.

    - Solar serves local load first.
    - Excess solar is exported at fixed feed-in tariff.
    - Battery is fully disabled.
    - EV charges immediately at max allowed rate.
    - No P2P trading.
    """

    def __init__(self, config: SimulatorConfig) -> None:
        self.config = config
        self.sim = GridSimulator(config)

    def run(self) -> dict:
        """Run the simulation and return results + computed KPIs."""
        result = self.sim.run()
        return self._compute_kpis(result)

    def _compute_kpis(self, result) -> dict:
        node = result.node_timeseries
        trans = result.transformer_timeseries

        total_solar = float(node["pv_kw"].sum())

        peak_import = float(node["net_grid_kw"].clip(0).max())
        total_cost = float((node["net_grid_kw"].clip(0) * 0.25 * 8.0).sum())

        overload_events = int(trans["overload_event_count"].max())

        return {
            "controller": "B0",
            "solar_kwh": total_solar * 0.25,
            "curtailment_pct": 100.0,
            "peak_demand_kw": peak_import,
            "peak_reduction_pct": 0.0,
            "total_cost_inr": total_cost,
            "cost_reduction_pct": 0.0,
            "overload_events": overload_events,
        }
