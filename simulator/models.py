from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class HomeAsset:
    node_id: int
    battery_node_id: int | None
    has_solar: bool
    has_battery: bool
    has_ev: bool
    pv_capacity_kw: float
    battery_capacity_kwh: float
    battery_max_kw: float
    battery_soc_kwh: float
    battery_charge_efficiency: float
    battery_discharge_efficiency: float
    ev_max_charge_kw: float
    ev_daily_kwh: float


@dataclass
class SimulationResult:
    node_timeseries: pd.DataFrame
    transformer_timeseries: pd.DataFrame
    event_log: pd.DataFrame
    metadata: dict[str, Any]

    def save(self, output_dir: str) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.node_timeseries.to_parquet(output_path / "node_timeseries.parquet", index=False)
        self.transformer_timeseries.to_parquet(
            output_path / "transformer_timeseries.parquet", index=False
        )
        self.event_log.to_parquet(output_path / "event_log.parquet", index=False)

        metadata_path = output_path / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as metadata_file:
            json.dump(self.metadata, metadata_file, indent=2, sort_keys=True)
