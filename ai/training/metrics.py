from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class KPIReport:
    controller: str = ""
    curtailment_pct: float = 0.0
    peak_reduction_pct: float = 0.0
    cost_reduction_pct: float = 0.0
    overload_events: int = 0
    island_switchover_s: float = 0.0
    settlement_latency_p95_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


def compute_kpi(
    controller_label: str,
    node_df: pd.DataFrame,
    trans_df: pd.DataFrame,
    b0_cost: float = 0.0,
    b0_peak: float = 0.0,
    b0_curtailment: float = 0.0,
) -> KPIReport:
    total_solar = float(node_df["pv_kw"].sum() * 0.25)
    total_load = float(node_df["load_kw"].sum() * 0.25) + float(node_df["ev_kw"].sum() * 0.25)
    total_export = float((-node_df["net_grid_kw"]).clip(0).sum() * 0.25)
    total_import = float(node_df["net_grid_kw"].clip(0).sum() * 0.25)

    curtailed = max(0.0, total_solar - (total_load - total_import) - total_export)
    curtailment_pct = (curtailed / max(total_solar, 1e-6)) * 100.0

    peak_import = float(node_df["net_grid_kw"].clip(0).max())
    denominator = max(b0_peak, 1e-6)
    peak_reduction_pct = ((b0_peak - peak_import) / denominator) * 100.0 if b0_peak > 0 else 0.0

    grid_cost = total_import * 8.0
    denominator = max(b0_cost, 1e-6)
    cost_reduction_pct = ((b0_cost - grid_cost) / denominator) * 100.0 if b0_cost > 0 else 0.0

    overload_events = int(trans_df["overload_event_count"].max()) if not trans_df.empty else 0

    return KPIReport(
        controller=controller_label,
        curtailment_pct=round(curtailment_pct, 2),
        peak_reduction_pct=round(peak_reduction_pct, 2),
        cost_reduction_pct=round(cost_reduction_pct, 2),
        overload_events=overload_events,
    )


def kpi_summary(reports: list[KPIReport]) -> pd.DataFrame:
    rows = []
    for r in reports:
        rows.append({
            "controller": r.controller,
            "curtailment_pct": r.curtailment_pct,
            "peak_reduction_pct": r.peak_reduction_pct,
            "cost_reduction_pct": r.cost_reduction_pct,
            "overload_events": r.overload_events,
        })
    return pd.DataFrame(rows)
