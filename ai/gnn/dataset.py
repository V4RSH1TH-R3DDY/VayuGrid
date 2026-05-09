from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from simulator.config import SimulatorConfig
from simulator.simulator import GridSimulator


@dataclass
class GraphSample:
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    transformer_features: np.ndarray
    target: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


class GraphDatasetGenerator:
    """Generates GNN training samples from the simulator.

    Each sample: 12 historical graph snapshots → next 30-minute target
    of transformer overload probability and voltage forecasts.
    """

    SNAPSHOT_WINDOW = 12
    TARGET_HORIZON = 30

    def __init__(self, config: SimulatorConfig) -> None:
        self.config = config
        self.sim = GridSimulator(config)

    def _extract_node_features(self, node_df: pd.DataFrame, t_idx: int) -> np.ndarray:
        row = node_df[node_df["timestamp_idx"] == t_idx]
        if row.empty:
            return np.zeros((self.config.neighborhood.num_homes, 6), dtype=np.float32)
        features = np.column_stack([
            row["pv_kw"].values,
            row["load_kw"].values,
            row["battery_soc_kwh"].values,
            row["net_grid_kw"].values,
            row["voltage_pu"].values,
            np.ones(len(row)),
        ])
        return features.astype(np.float32)

    def generate(self) -> list[GraphSample]:
        result = self.sim.run()
        node_df = result.node_timeseries.copy()
        node_df = node_df[node_df["node_type"] == "home"].reset_index(drop=True)
        trans_df = result.transformer_timeseries

        timestamps = node_df["timestamp"].unique()
        timestamp_to_idx = {ts: i for i, ts in enumerate(timestamps)}
        node_df["timestamp_idx"] = node_df["timestamp"].map(timestamp_to_idx)

        edges = getattr(self.sim.graph, "edges", [])
        edge_index = (
            np.array([[u, v] for u, v in edges], dtype=np.int64).T
            if edges else np.zeros((2, 0), dtype=np.int64)
        )

        samples = []
        for t_idx in range(self.SNAPSHOT_WINDOW, len(timestamps) - self.TARGET_HORIZON):
            snapshots = []
            for k in range(self.SNAPSHOT_WINDOW):
                feat = self._extract_node_features(node_df, t_idx - self.SNAPSHOT_WINDOW + k)
                snapshots.append(feat)

            node_feats = np.stack(snapshots, axis=0)

            target_start = t_idx
            target_end = min(t_idx + self.TARGET_HORIZON, len(trans_df))
            overloads = trans_df["transformer_loading_pu"].iloc[target_start:target_end].values
            target = (overloads > 1.2).astype(np.float32)
            if len(target) < self.TARGET_HORIZON:
                target = np.pad(target, (0, self.TARGET_HORIZON - len(target)))

            tf = trans_df[["transformer_loading_pu", "hottest_spot_temp_c",
                           "aging_acceleration"]].iloc[t_idx].values.astype(np.float32)

            samples.append(GraphSample(
                node_features=node_feats,
                edge_index=edge_index,
                edge_features=np.zeros((edge_index.shape[1], 1), dtype=np.float32),
                transformer_features=tf,
                target=target,
                metadata={"t_idx": t_idx, "timestamp": str(timestamps[t_idx])},
            ))

        return samples

    def generate_dataset(
        self, num_episodes: int = 90,
    ) -> tuple[list[GraphSample], list[GraphSample], list[GraphSample]]:
        all_samples: list[GraphSample] = []
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
