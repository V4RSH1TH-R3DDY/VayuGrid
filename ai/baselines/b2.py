from __future__ import annotations

import numpy as np
import pandas as pd

from simulator.config import SimulatorConfig
from simulator.models import HomeAsset

try:
    from scipy.optimize import linprog
except ImportError:
    linprog = None  # type: ignore[assignment]


class B2Controller:
    """Baseline-2: MPC-lite.

    Rolling horizon of 30 minutes, re-optimised every timestep.
    Uses linear programming to minimise grid cost + curtailment penalty
    + battery degradation.
    """

    HORIZON = 30
    GRID_TARIFF = 8.0
    CURTAILMENT_PENALTY = 2.0
    BATTERY_DEGRADATION_PENALTY = 0.5
    DT_HOURS = 1.0 / 60.0

    def __init__(self, config: SimulatorConfig) -> None:
        self.config = config

    def _optimise(
        self,
        asset: HomeAsset,
        pv_forecast: np.ndarray,
        load_forecast: np.ndarray,
        ev_forecast: np.ndarray,
    ) -> np.ndarray:
        if linprog is None:
            raise ImportError("scipy is required for B2 MPC-lite")

        n = len(pv_forecast)
        c = np.zeros(3 * n)
        c[:n] = self.GRID_TARIFF
        c[n : 2 * n] = self.CURTAILMENT_PENALTY
        c[2 * n :] = self.BATTERY_DEGRADATION_PENALTY

        A_list: list[np.ndarray] = []
        b_list: list[float] = []

        for t in range(n):
            row = np.zeros(3 * n)
            row[t] = 1.0
            A_list.append(row)
            b_list.append(asset.battery_max_kw)
            row2 = np.zeros(3 * n)
            row2[t] = -1.0
            A_list.append(row2)
            b_list.append(asset.battery_max_kw)

        A_ub = np.array(A_list)
        b_ub = np.array(b_list)

        bounds: list[tuple[float, float | None]] = [(0, asset.battery_max_kw) for _ in range(n)]
        bounds += [(0, None) for _ in range(2 * n)]

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not result.success:
            return np.zeros(n)
        return result.x[:n]

    def run(self, timeline: pd.DatetimeIndex, home_assets: list[HomeAsset],
            load_matrix: np.ndarray, pv_matrix: np.ndarray, ev_matrix: np.ndarray) -> dict:
        records = []
        for t_idx in range(len(timeline)):
            horizon_end = min(t_idx + self.HORIZON, len(timeline))
            for i, asset in enumerate(home_assets):
                pv_fc = pv_matrix[t_idx:horizon_end, i]
                load_fc = load_matrix[t_idx:horizon_end, i]
                ev_fc = ev_matrix[t_idx:horizon_end, i]

                opt_batt = self._optimise(asset, pv_fc, load_fc, ev_fc)

                net = float(load_matrix[t_idx, i] + ev_matrix[t_idx, i] - pv_matrix[t_idx, i])
                batt = float(opt_batt[0]) if len(opt_batt) > 0 else 0.0
                grid = net - batt

                records.append({
                    "timestamp": timeline[t_idx],
                    "node_id": asset.node_id,
                    "battery_power_kw": batt,
                    "net_grid_kw": grid,
                })

        df = pd.DataFrame(records)
        total_import = float(df["net_grid_kw"].clip(0).sum() * 0.25)
        total_cost = total_import * self.GRID_TARIFF

        return {
            "controller": "B2",
            "total_grid_import_kwh": total_import,
            "total_cost_inr": total_cost,
            "net_cost_inr": total_cost,
        }
