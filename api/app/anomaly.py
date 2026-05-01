from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .db import fetch_all

logger = logging.getLogger(__name__)

FEATURES = [
    "battery_soc_kwh",
    "solar_output_kw",
    "household_load_kw",
    "ev_charge_kw",
    "net_grid_kw",
    "voltage_pu",
]
MIN_SAMPLES = 20


def _to_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array(
        [[float(r.get(f) or 0.0) for f in FEATURES] for r in rows],
        dtype=np.float64,
    )


class AnomalyDetector:
    """Isolation Forest anomaly detector for Vayu-Node telemetry.

    Maintains a global model trained on all nodes, plus per-node models
    when a node has at least MIN_SAMPLES readings in the last 7 days.
    """

    def __init__(self, contamination: float = 0.05, threshold: float = -0.1) -> None:
        self.contamination = contamination
        self.threshold = threshold
        self._global_model: Any = None
        self._node_models: dict[int, Any] = {}
        self.is_trained: bool = False

    def fit_from_db(self) -> None:
        """Train (or retrain) models from the last 7 days of telemetry."""
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            logger.warning("scikit-learn not installed; anomaly detection disabled.")
            return

        rows = fetch_all(
            """
            SELECT node_id, battery_soc_kwh, solar_output_kw, household_load_kw,
                   ev_charge_kw, net_grid_kw, voltage_pu
            FROM node_telemetry
            WHERE ts >= now() - INTERVAL '7 days'
            ORDER BY node_id, ts
            """
        )
        if len(rows) < MIN_SAMPLES:
            logger.info("Not enough telemetry data to train anomaly detector (%d rows).", len(rows))
            return

        X_global = _to_matrix(rows)
        self._global_model = IsolationForest(
            n_estimators=100, contamination=self.contamination, random_state=42
        )
        self._global_model.fit(X_global)

        # Per-node models
        from collections import defaultdict

        node_rows: dict[int, list] = defaultdict(list)
        for row in rows:
            node_rows[int(row["node_id"])].append(row)

        self._node_models = {}
        for node_id, n_rows in node_rows.items():
            if len(n_rows) >= MIN_SAMPLES:
                X_node = _to_matrix(n_rows)
                model = IsolationForest(
                    n_estimators=100, contamination=self.contamination, random_state=42
                )
                model.fit(X_node)
                self._node_models[node_id] = model

        self.is_trained = True
        logger.info(
            "Anomaly detector trained on %d rows; %d per-node models.",
            len(rows),
            len(self._node_models),
        )

    def score(self, reading: dict[str, Any], node_id: int | None = None) -> float:
        """Return the anomaly score (more negative = more anomalous). Returns 0.0 if not trained."""
        model = (
            self._node_models.get(node_id)  # type: ignore[arg-type]
            if node_id is not None
            else None
        ) or self._global_model
        if model is None:
            return 0.0
        X = _to_matrix([reading])
        return float(model.score_samples(X)[0])

    def is_anomalous(
        self, reading: dict[str, Any], node_id: int | None = None
    ) -> tuple[bool, float]:
        """Return (is_anomalous, score). Uses per-node model if available, else global."""
        s = self.score(reading, node_id=node_id)
        return s < self.threshold, s


# Module-level singleton — trained lazily at API startup
detector = AnomalyDetector()
