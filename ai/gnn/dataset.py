from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch

from ai.gnn.vayu_gnn import (
    EDGE_FEAT_DIM,
    N_SNAPSHOTS,
    NODE_FEAT_DIM,
    OVERLOAD_STEPS,
    VOLTAGE_STEPS,
    XFMR_FEAT_DIM,
    GraphSnapshot,
)
from simulator.config import SimulatorConfig
from simulator.simulator import GridSimulator


@dataclass
class GNNSample:
    snapshots: list[GraphSnapshot]
    target_overload: torch.Tensor   # (30,)
    target_voltage: torch.Tensor    # (30,)
    target_risk: torch.Tensor       # (1,)
    target_duck: torch.Tensor       # (96,)
    metadata: dict[str, Any] = field(default_factory=dict)


class GraphDatasetGenerator:
    """Generates GNN training samples from the simulator.

    Each sample: N_SNAPSHOTS historical GraphSnapshots → 4 prediction targets
    (overload, voltage, risk, duck curve) for the next 30 minutes / 24 hours.
    """

    SNAPSHOT_WINDOW = N_SNAPSHOTS   # 12
    OVERLOAD_HORIZON = OVERLOAD_STEPS  # 30
    VOLTAGE_HORIZON = VOLTAGE_STEPS    # 30
    DUCK_HORIZON = 96                  # 96 × 15-min = 24 h

    def __init__(self, config: SimulatorConfig) -> None:
        self.config = config
        self.sim = GridSimulator(config)
        self.num_homes = config.neighborhood.num_homes
        # Single-transformer: all homes map to transformer 0
        self._node_to_xfmr = np.zeros(self.num_homes, dtype=np.int64)

    def _extract_node_features(self, node_df: pd.DataFrame, t_idx: int) -> np.ndarray:
        """Returns (N, NODE_FEAT_DIM) for homes at timestamp t_idx."""
        row = node_df[node_df["timestamp_idx"] == t_idx]
        if row.empty:
            return np.zeros((self.num_homes, NODE_FEAT_DIM), dtype=np.float32)
        features = np.column_stack([
            row["pv_kw"].values,
            row["load_kw"].values,
            row["battery_soc_kwh"].values,
            row["net_grid_kw"].values,
            row["voltage_pu"].values,
        ])
        return features.astype(np.float32)

    def _extract_xfmr_features(self, trans_df: pd.DataFrame, t_idx: int) -> np.ndarray:
        """Returns (1, XFMR_FEAT_DIM) — single transformer."""
        if t_idx >= len(trans_df):
            return np.zeros((1, XFMR_FEAT_DIM), dtype=np.float32)
        row = trans_df.iloc[t_idx]
        return np.array([[
            row["transformer_loading_pu"],
            row["hottest_spot_temp_c"],
            row["aging_acceleration"],
        ]], dtype=np.float32)

    def _build_snapshot(
        self,
        node_df: pd.DataFrame,
        trans_df: pd.DataFrame,
        t_idx: int,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> GraphSnapshot:
        node_feat = self._extract_node_features(node_df, t_idx)
        xfmr_feat = self._extract_xfmr_features(trans_df, t_idx)
        return GraphSnapshot(
            node_features=torch.from_numpy(node_feat),
            xfmr_features=torch.from_numpy(xfmr_feat),
            edge_index=edge_index,
            edge_features=edge_features,
            node_to_xfmr=torch.from_numpy(self._node_to_xfmr),
        )

    def _compute_targets(
        self,
        node_df: pd.DataFrame,
        trans_df: pd.DataFrame,
        start_idx: int,
    ) -> dict[str, torch.Tensor]:
        end_idx = min(start_idx + self.OVERLOAD_HORIZON, len(trans_df))
        n = end_idx - start_idx

        # Overload: binary indicator for transformer_loading_pu > 1.2
        overloads = trans_df["transformer_loading_pu"].iloc[start_idx:end_idx].values
        target_overload = (overloads > 1.2).astype(np.float32)
        if n < self.OVERLOAD_HORIZON:
            target_overload = np.pad(
                target_overload, (0, self.OVERLOAD_HORIZON - n)
            )

        # Voltage: mean home voltage across the next OVERLOAD_HORIZON minutes
        voltage_vals = []
        for offset in range(self.VOLTAGE_HORIZON):
            vt = start_idx + offset
            row = node_df[node_df["timestamp_idx"] == vt]
            if not row.empty:
                voltage_vals.append(float(row["voltage_pu"].mean()))
            else:
                voltage_vals.append(1.0)
        target_voltage = np.array(voltage_vals, dtype=np.float32)

        # Risk: max transformer loading in the next 30 min, normalized to [0,1]
        max_loading = float(trans_df["transformer_loading_pu"].iloc[start_idx:end_idx].max())
        target_risk = np.array([min(max_loading, 1.0)], dtype=np.float32)

        # Duck curve: net load (load - pv) at 15-min intervals for next 24h
        duck_vals = []
        for offset in range(0, self.DUCK_HORIZON * 15, 15):
            dt = start_idx + offset
            if dt < len(node_df["timestamp_idx"].unique()):
                row = node_df[node_df["timestamp_idx"] == dt]
                if not row.empty:
                    net = float((row["load_kw"].sum() - row["pv_kw"].sum()))
                    duck_vals.append(net)
                else:
                    duck_vals.append(0.0)
            else:
                duck_vals.append(0.0)
        target_duck = np.array(duck_vals, dtype=np.float32)

        return {
            "target_overload": torch.from_numpy(target_overload),
            "target_voltage": torch.from_numpy(target_voltage),
            "target_risk": torch.from_numpy(target_risk),
            "target_duck": torch.from_numpy(target_duck),
        }

    def generate(self) -> list[GNNSample]:
        result = self.sim.run()
        node_df = result.node_timeseries.copy()
        node_df = node_df[node_df["node_type"] == "home"].reset_index(drop=True)
        trans_df = result.transformer_timeseries

        timestamps = node_df["timestamp"].unique()
        timestamp_to_idx = {ts: i for i, ts in enumerate(timestamps)}
        node_df["timestamp_idx"] = node_df["timestamp"].map(timestamp_to_idx)

        # Build static edge data from the graph
        edges = getattr(self.sim.graph, "edges", [])
        if edges:
            edge_index = torch.tensor(
                [[u, v] for u, v in edges], dtype=torch.int64
            ).T.contiguous()
            edge_features = torch.zeros((edge_index.shape[1], EDGE_FEAT_DIM), dtype=torch.float32)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.int64)
            edge_features = torch.zeros((0, EDGE_FEAT_DIM), dtype=torch.float32)

        samples: list[GNNSample] = []
        max_start = len(timestamps) - self.SNAPSHOT_WINDOW - max(
            self.OVERLOAD_HORIZON, self.VOLTAGE_HORIZON
        )

        for t_idx in range(self.SNAPSHOT_WINDOW, max_start):
            # Build N_SNAPSHOTS consecutive GraphSnapshots
            snapshots = []
            for k in range(self.SNAPSHOT_WINDOW):
                snap = self._build_snapshot(
                    node_df, trans_df,
                    t_idx - self.SNAPSHOT_WINDOW + k,
                    edge_index, edge_features,
                )
                snapshots.append(snap)

            # Compute targets for the future horizon
            targets = self._compute_targets(node_df, trans_df, t_idx)

            samples.append(GNNSample(
                snapshots=snapshots,
                **targets,
                metadata={"t_idx": t_idx, "timestamp": str(timestamps[t_idx])},
            ))

        return samples

    def generate_dataset(
        self, num_episodes: int = 90,
    ) -> tuple[list[GNNSample], list[GNNSample], list[GNNSample]]:
        all_samples: list[GNNSample] = []
        for _ in range(num_episodes):
            samples = self.generate()
            all_samples.extend(samples)

        n = len(all_samples)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)

        return (
            all_samples[:train_end],
            all_samples[train_end:val_end],
            all_samples[val_end:],
        )
