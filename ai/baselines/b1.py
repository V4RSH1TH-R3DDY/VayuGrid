from __future__ import annotations

import numpy as np
import pandas as pd

from simulator.config import SimulatorConfig
from simulator.models import HomeAsset


class B1Controller:
    """Baseline-1: Rule-based TOU control.

    Battery rules:
      - Charge when solar > load and SoC < 90%.
      - Discharge when load > solar and SoC > 20%.

    EV rules:
      - Charge only between 22:00-06:00 OR when market price < threshold.

    P2P rules:
      - Sell when market_price > 0.8 × grid_tariff.
      - Buy when market_price < 1.1 × grid_tariff.
    """

    GRID_TARIFF = 8.0
    EV_PRICE_THRESHOLD = 6.0
    PEAK_START_HOUR = 18
    PEAK_END_HOUR = 22

    def __init__(self, config: SimulatorConfig) -> None:
        self.config = config

    def dispatch(
        self,
        asset: HomeAsset,
        load_kw: float,
        pv_kw: float,
        ev_kw: float,
        hour: int,
        market_buy_price: float = GRID_TARIFF,
        market_sell_price: float = GRID_TARIFF,
        dt_hours: float = 1.0 / 60.0,
    ) -> dict[str, float]:
        soc = asset.battery_soc_kwh
        net = load_kw + ev_kw - pv_kw
        battery_power = 0.0
        p2p_import = 0.0
        p2p_export = 0.0

        if asset.has_battery:
            soc_norm = soc / max(asset.battery_capacity_kwh, 1e-6)
            if net < 0 and soc_norm < 0.90:
                charge_kw = min(asset.battery_max_kw, -net)
                soc_delta = charge_kw * dt_hours * asset.battery_charge_efficiency
                soc = min(asset.battery_capacity_kwh, soc + soc_delta)
                battery_power = -charge_kw
                net += charge_kw
            elif net > 0 and soc_norm > 0.20:
                discharge_kw = min(asset.battery_max_kw, net)
                eff = max(asset.battery_discharge_efficiency, 1e-6)
                soc_delta = (discharge_kw * dt_hours) / eff
                soc = max(0.0, soc - soc_delta)
                battery_power = discharge_kw
                net -= discharge_kw

        if asset.has_ev and ev_kw > 0:
            if not (hour >= 22 or hour < 6 or market_buy_price < self.EV_PRICE_THRESHOLD):
                ev_kw = 0.0
                net = load_kw + 0.0 - pv_kw - battery_power

        asset.battery_soc_kwh = float(soc)

        if net > 0:
            if market_sell_price > 0.8 * self.GRID_TARIFF:
                p2p_export = net
                net = 0.0
        elif net < 0:
            if market_buy_price < 1.1 * self.GRID_TARIFF:
                p2p_import = -net
                net = 0.0

        return {
            "battery_power_kw": battery_power,
            "battery_soc_kwh": float(asset.battery_soc_kwh),
            "ev_kw": ev_kw,
            "net_grid_kw": net,
            "p2p_import_kw": p2p_import,
            "p2p_export_kw": p2p_export,
        }

    def run(self, timeline: pd.DatetimeIndex, home_assets: list[HomeAsset],
            load_matrix: np.ndarray, pv_matrix: np.ndarray, ev_matrix: np.ndarray) -> dict:
        records = []
        for t_idx in range(len(timeline)):
            hour = timeline[t_idx].hour
            for i, asset in enumerate(home_assets):
                result = self.dispatch(
                    asset,
                    float(load_matrix[t_idx, i]),
                    float(pv_matrix[t_idx, i]),
                    float(ev_matrix[t_idx, i]),
                    hour,
                )
                records.append({
                    "timestamp": timeline[t_idx],
                    "node_id": asset.node_id,
                    **result,
                })

        df = pd.DataFrame(records)
        total_import = float(df["net_grid_kw"].clip(0).sum() * 0.25)
        total_export = float((-df["net_grid_kw"]).clip(0).sum() * 0.25)
        total_cost = total_import * self.GRID_TARIFF
        p2p_revenue = float(df["p2p_export_kw"].sum() * 0.25 * self.GRID_TARIFF * 0.9)
        net_cost = total_cost - p2p_revenue

        return {
            "controller": "B1",
            "total_grid_import_kwh": total_import,
            "total_grid_export_kwh": total_export,
            "total_cost_inr": total_cost,
            "p2p_revenue_inr": p2p_revenue,
            "net_cost_inr": net_cost,
        }
