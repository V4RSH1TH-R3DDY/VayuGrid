from __future__ import annotations

import pandas as pd

from simulator.models import SimulationResult


def compute_curtailment(node_df: pd.DataFrame) -> float:
    """Solar curtailment percentage per the locked definition.

    P_curtailed(t) = max(0, solar - load - battery_charge - export)
    Curtailment(%) = (sum(P_curtailed) / sum(solar)) * 100
    """
    home = node_df[node_df["node_type"] == "home"].copy()
    total_solar = float(home["pv_kw"].sum())
    if total_solar < 1e-6:
        return 0.0
    battery_charge = home["battery_power_kw"].clip(upper=0).abs()
    grid_export = home["net_grid_kw"].clip(upper=0).abs()
    curtailed = (
        home["pv_kw"] - home["load_kw"] - home["ev_kw"]
        - battery_charge - grid_export
    ).clip(lower=0)
    return float((curtailed.sum() / total_solar) * 100.0)


def compute_peak_demand(node_df: pd.DataFrame) -> float:
    """Peak grid import in kW."""
    home = node_df[node_df["node_type"] == "home"]
    return float(home["net_grid_kw"].clip(lower=0).max())


def compute_peak_reduction(node_df: pd.DataFrame, b0_peak: float) -> float:
    """Peak demand reduction relative to B0."""
    if b0_peak < 1e-6:
        return 0.0
    peak = compute_peak_demand(node_df)
    return ((b0_peak - peak) / b0_peak) * 100.0


def compute_total_cost(node_df: pd.DataFrame, tariff: float = 8.0, feed_in: float = 6.0) -> float:
    """Total cost = grid_import_cost - p2p_revenue + export_revenue.

    For baselines without P2P this is simply import cost minus export revenue.
    """
    home = node_df[node_df["node_type"] == "home"]
    import_kwh = float(home["net_grid_kw"].clip(lower=0).sum() / 60.0)
    export_kwh = float(home["net_grid_kw"].clip(upper=0).abs().sum() / 60.0)
    return import_kwh * tariff - export_kwh * feed_in


def compute_cost_reduction(
    node_df: pd.DataFrame, b0_cost: float,
    tariff: float = 8.0, feed_in: float = 6.0,
) -> tuple[float, float]:
    """Cost reduction relative to B0.  Returns (cost, reduction_pct)."""
    cost = compute_total_cost(node_df, tariff, feed_in)
    if abs(b0_cost) < 1e-6:
        return cost, 0.0
    return cost, ((b0_cost - cost) / abs(b0_cost)) * 100.0


def compute_overload_events(transformer_df: pd.DataFrame) -> int:
    """Transformer overload events count from simulation output."""
    return int(transformer_df["overload_event_count"].max()) if not transformer_df.empty else 0


def compute_island_switchover(node_df: pd.DataFrame, event_df: pd.DataFrame) -> float:
    """Island switchover time in seconds.

    Outage detection: first timestep where grid_available becomes False with islanding.
    Island stable: voltage stabilises above 0.95 pu for all homes.
    """
    home = node_df[node_df["node_type"] == "home"].copy()
    if "islanding_active" not in home.columns:
        return 0.0
    island_starts = home[home["islanding_active"]]
    if island_starts.empty:
        return 0.0
    outage_detect = island_starts["timestamp"].iloc[0]
    island_home = island_starts[island_starts["voltage_pu"] >= 0.95]
    if island_home.empty:
        return 0.0
    island_stable = island_home["timestamp"].iloc[0]
    return float((pd.Timestamp(island_stable) - pd.Timestamp(outage_detect)).total_seconds())


def compute_kpis(
    result: SimulationResult,
    b0_result: SimulationResult | None = None,
    tariff: float = 8.0,
    feed_in: float = 6.0,
) -> dict[str, float | int]:
    """Compute all 6 baseline KPIs from a simulation result.

    When b0_result is provided, peak_reduction_pct and cost_reduction_pct
    are computed relative to B0.
    """
    node = result.node_timeseries
    trans = result.transformer_timeseries
    events = result.event_log

    curtailment = compute_curtailment(node)
    peak = compute_peak_demand(node)
    cost, cost_reduction = compute_cost_reduction(node, 0.0, tariff, feed_in)
    overloads = compute_overload_events(trans)
    island_time = compute_island_switchover(node, events)

    peak_reduction = 0.0
    if b0_result is not None:
        b0_peak = compute_peak_demand(b0_result.node_timeseries)
        peak_reduction = compute_peak_reduction(node, b0_peak)
        b0_cost = compute_total_cost(b0_result.node_timeseries, tariff, feed_in)
        cost, cost_reduction = compute_cost_reduction(node, b0_cost, tariff, feed_in)

    return {
        "curtailment_pct": round(curtailment, 2),
        "peak_demand_kw": round(peak, 2),
        "peak_reduction_pct": round(peak_reduction, 2),
        "total_cost_inr": round(cost, 2),
        "cost_reduction_pct": round(cost_reduction, 2),
        "overload_events": overloads,
        "island_switchover_seconds": round(island_time, 1),
    }
