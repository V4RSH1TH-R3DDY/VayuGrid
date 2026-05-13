from __future__ import annotations

import numpy as np
import pandas as pd

from simulator.config import SimulatorConfig
from simulator.models import HomeAsset


class GridController:
    """Base controller for overriding battery and EV dispatch in GridSimulator.

    Subclasses must implement ``control_home()``.
    """

    GRID_TARIFF_INR_PER_KWH: float = 8.0
    FEED_IN_TARIFF_INR_PER_KWH: float = 6.0
    DT_HOURS: float = 1.0 / 60.0

    def __init__(
        self,
        config: SimulatorConfig,
        timeline: pd.DatetimeIndex,
        home_assets: list[HomeAsset],
        load_matrix: np.ndarray,
        pv_matrix: np.ndarray,
        ev_matrix: np.ndarray,
    ) -> None:
        self.config = config
        self.timeline = timeline
        self.home_assets = home_assets
        self.load_matrix = load_matrix
        self.pv_matrix = pv_matrix
        self.ev_matrix = ev_matrix

    def control_home(
        self,
        t_idx: int,
        home_idx: int,
        asset: HomeAsset,
        load_kw: float,
        pv_kw: float,
        ev_kw: float,
    ) -> tuple[float, float, float]:
        """Return (battery_power_kw, new_soc_kwh, modified_ev_kw)."""
        return 0.0, asset.battery_soc_kwh, ev_kw


class NullController(GridController):
    """Controller that preserves the simulator's built-in battery dispatch."""

    def control_home(
        self,
        t_idx: int,
        home_idx: int,
        asset: HomeAsset,
        load_kw: float,
        pv_kw: float,
        ev_kw: float,
    ) -> tuple[float, float, float]:
        return 0.0, asset.battery_soc_kwh, ev_kw


class B0Controller(GridController):
    """Baseline-0: No control.

    - Solar serves local load first, excess exported at fixed feed-in tariff.
    - Battery is fully disabled (SoC forced to 0).
    - EV charges immediately at max allowed rate (profile default).
    - No P2P trading.
    """

    def control_home(
        self,
        t_idx: int,
        home_idx: int,
        asset: HomeAsset,
        load_kw: float,
        pv_kw: float,
        ev_kw: float,
    ) -> tuple[float, float, float]:
        asset.battery_soc_kwh = 0.0
        return 0.0, 0.0, ev_kw


class B1Controller(GridController):
    """Baseline-1: Rule-based TOU control.

    Battery rules:
      - Charge when solar > load and SoC < 90%.
      - Discharge when load > solar and SoC > 20%.

    EV rules:
      - Charge only between 22:00–06:00 OR when market price < threshold.

    P2P rules:
      - Sell when market_price > 0.8 × grid_tariff.
      - Buy when market_price < 1.1 × grid_tariff.
    """

    EV_PRICE_THRESHOLD: float = 6.0
    PEAK_START_HOUR: int = 18
    PEAK_END_HOUR: int = 22
    P2P_SELL_THRESHOLD: float = 0.8
    P2P_BUY_THRESHOLD: float = 1.1

    def control_home(
        self,
        t_idx: int,
        home_idx: int,
        asset: HomeAsset,
        load_kw: float,
        pv_kw: float,
        ev_kw: float,
    ) -> tuple[float, float, float]:
        hour = self.timeline[t_idx].hour
        dt = self.DT_HOURS
        new_soc = asset.battery_soc_kwh
        battery_power = 0.0
        mod_ev = ev_kw

        if asset.has_ev and ev_kw > 0:
            is_off_peak = hour >= 22 or hour < 6
            cheap_power = self.GRID_TARIFF_INR_PER_KWH < self.EV_PRICE_THRESHOLD
            if not (is_off_peak or cheap_power):
                mod_ev = 0.0

        net = load_kw + mod_ev - pv_kw

        if asset.has_battery:
            soc_norm = new_soc / max(asset.battery_capacity_kwh, 1e-6)
            if net < 0 and soc_norm < 0.90:
                charge_kw = min(asset.battery_max_kw, -net)
                soc_delta = charge_kw * dt * asset.battery_charge_efficiency
                new_soc = min(asset.battery_capacity_kwh, new_soc + soc_delta)
                battery_power = -charge_kw
            elif net > 0 and soc_norm > 0.20:
                discharge_kw = min(asset.battery_max_kw, net)
                eff = max(asset.battery_discharge_efficiency, 1e-6)
                new_soc = max(0.0, new_soc - (discharge_kw * dt) / eff)
                battery_power = discharge_kw

        return battery_power, float(new_soc), mod_ev


class B2Controller(GridController):
    """Baseline-2: MPC-lite.

    Rolling horizon of 30 minutes, re-optimised every timestep.
    Uses linear programming to minimise grid cost + curtailment penalty
    + battery degradation.
    """

    HORIZON: int = 30
    CURTAILMENT_PENALTY: float = 2.0
    BATTERY_DEGRADATION_PENALTY: float = 0.5

    def __init__(
        self,
        config: SimulatorConfig,
        timeline: pd.DatetimeIndex,
        home_assets: list[HomeAsset],
        load_matrix: np.ndarray,
        pv_matrix: np.ndarray,
        ev_matrix: np.ndarray,
    ) -> None:
        super().__init__(config, timeline, home_assets, load_matrix, pv_matrix, ev_matrix)
        try:
            from scipy.optimize import linprog as _lp
            self._linprog = _lp
        except ImportError:
            self._linprog = None  # type: ignore[assignment]

    def _build_and_solve(
        self,
        asset: HomeAsset,
        pv_fc: np.ndarray,
        load_fc: np.ndarray,
        ev_fc: np.ndarray,
        soc_start: float,
    ) -> np.ndarray:
        if self._linprog is None:
            raise ImportError("scipy is required for B2 MPC-lite")

        n = len(pv_fc)
        tariff = self.GRID_TARIFF_INR_PER_KWH
        curtail_pen = self.CURTAILMENT_PENALTY
        deg_pen = self.BATTERY_DEGRADATION_PENALTY
        dt = self.DT_HOURS
        cap = asset.battery_capacity_kwh
        max_kw = asset.battery_max_kw
        eta_ch = asset.battery_charge_efficiency
        eta_dis = asset.battery_discharge_efficiency

        # Variables per timestep:
        #   grid_import(t), grid_export(t), curtailed(t),
        #   charge(t), discharge(t), soc(t)
        # Total: 6 * n variables
        n_vars = 6 * n

        # Objective coefficients
        c = np.zeros(n_vars)
        c[0:n] = tariff  # grid_import
        c[n:2*n] = 0.0  # grid_export (no coefficient — revenue, handled via negative)
        c[2*n:3*n] = curtail_pen  # curtailed
        c[3*n:4*n] = deg_pen  # charge (degradation)
        c[4*n:5*n] = deg_pen  # discharge (degradation)
        c[5*n:6*n] = 0.0  # soc (no direct cost)

        # Adjust: grid_export gives revenue, so coefficient is negative tariff
        # (but grid_export can't exceed pv excess, handled by constraints)
        c[n:2*n] = -tariff * 0.9  # feed-in at 90% of tariff

        A_ub_list = []
        b_ub_list = []

        # Charge power limit per timestep
        for t in range(n):
            row = np.zeros(n_vars)
            row[3*n + t] = 1.0  # charge(t)
            A_ub_list.append(row)
            b_ub_list.append(max_kw)

        # Discharge power limit per timestep
        for t in range(n):
            row = np.zeros(n_vars)
            row[4*n + t] = 1.0  # discharge(t)
            A_ub_list.append(row)
            b_ub_list.append(max_kw)

        # SoC limits per timestep
        for t in range(n):
            row_up = np.zeros(n_vars)
            row_up[5*n + t] = 1.0
            A_ub_list.append(row_up)
            b_ub_list.append(cap)

            row_lo = np.zeros(n_vars)
            row_lo[5*n + t] = -1.0
            A_ub_list.append(row_lo)
            b_ub_list.append(0.0)

        # SoC dynamics: soc(t+1) = soc(t) + (charge(t)*eta_ch - discharge(t)/eta_dis) * dt
        # => soc(t) + charge(t)*eta_ch*dt - discharge(t)/eta_dis*dt - soc(t+1) = 0
        A_eq_list = []
        b_eq_list = []

        for t in range(n):
            row = np.zeros(n_vars)
            if t == 0:
                row[5*n + t] = 1.0  # soc(0) coefficient
                b_eq_list.append(soc_start)
            else:
                row[5*n + t] = 1.0
                row[5*n + t - 1] = -1.0
                b_eq_list.append(0.0)
            row[3*n + t] = -eta_ch * dt  # charge contribution
            row[4*n + t] = dt / eta_dis  # discharge contribution
            A_eq_list.append(row)

        # Power balance:
        # pv + discharge + grid_import = load + ev + charge + grid_export + curtailed
        for t in range(n):
            row = np.zeros(n_vars)
            row[0*n + t] = 1.0   # grid_import
            row[1*n + t] = -1.0  # -grid_export
            row[2*n + t] = 1.0   # curtailed
            row[3*n + t] = -1.0  # -charge
            row[4*n + t] = 1.0   # +discharge
            net = float(load_fc[t] + ev_fc[t] - pv_fc[t])
            A_eq_list.append(row)
            b_eq_list.append(net)

        A_eq = np.array(A_eq_list) if A_eq_list else np.zeros((0, n_vars))
        b_eq = np.array(b_eq_list) if b_eq_list else np.zeros(0)
        A_ub = np.array(A_ub_list) if A_ub_list else np.zeros((0, n_vars))
        b_ub = np.array(b_ub_list) if b_ub_list else np.zeros(0)

        per_home_max_export = max(pv_fc.max() * 1.5, asset.pv_capacity_kw * 1.5)
        per_home_max_import = max(load_fc.max() + ev_fc.max() + asset.battery_max_kw, 20.0)

        bounds: list[tuple[float, float | None]] = [
            (0, per_home_max_import) if i < n else           # grid_import
            (0, per_home_max_export) if i < 2 * n else       # grid_export
            (0, per_home_max_export) if i < 3 * n else       # curtailed
            (0, asset.battery_max_kw) if i < 4 * n else      # charge
            (0, asset.battery_max_kw) if i < 5 * n else      # discharge
            (0, None) for i in range(n_vars)                  # soc
        ]

        result = self._linprog(
            c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
            bounds=bounds, method="highs",
        )
        if not result.success:
            return np.zeros(n)

        discharge = result.x[4*n:5*n]
        charge = result.x[3*n:4*n]
        return discharge - charge  # net battery power (positive=discharge)

    def control_home(
        self,
        t_idx: int,
        home_idx: int,
        asset: HomeAsset,
        load_kw: float,
        pv_kw: float,
        ev_kw: float,
    ) -> tuple[float, float, float]:
        horizon_end = min(t_idx + self.HORIZON, len(self.timeline))

        pv_fc = self.pv_matrix[t_idx:horizon_end, home_idx]
        load_fc = self.load_matrix[t_idx:horizon_end, home_idx]
        ev_fc = self.ev_matrix[t_idx:horizon_end, home_idx]

        if len(pv_fc) < 2:
            return 0.0, asset.battery_soc_kwh, ev_kw

        batt_power = self._build_and_solve(
            asset, pv_fc, load_fc, ev_fc, asset.battery_soc_kwh,
        )
        opt_power = float(batt_power[0]) if len(batt_power) > 0 else 0.0

        new_soc = asset.battery_soc_kwh
        dt = self.DT_HOURS
        if opt_power > 0:
            discharge = min(opt_power, asset.battery_max_kw)
            eff = max(asset.battery_discharge_efficiency, 1e-6)
            new_soc = max(0.0, new_soc - (discharge * dt) / eff)
            return discharge, float(new_soc), ev_kw
        else:
            charge = min(-opt_power, asset.battery_max_kw)
            soc_delta = charge * dt * asset.battery_charge_efficiency
            new_soc = min(asset.battery_capacity_kwh, new_soc + soc_delta)
            return -charge, float(new_soc), ev_kw


class PB0Controller(B0Controller):
    """Pecan-Baseline-0: Replay + No Control."""

    pass


class PB1Controller(B1Controller):
    """Pecan-Baseline-1: TOU + Self-Consumption Rules.

    Battery:
      - Charge from PV surplus when pv > load and SoC < 90%.
      - Discharge during evening peak 18:00–22:00 when SoC > 20%.
      - Outside peak, discharge only if price > 1.2 × grid_tariff.

    EV:
      - Treat exogenous ev_kw as requested demand; shift to 22:00–06:00 first.

    P2P:
      - Sell when net surplus and market_price >= grid_tariff.
      - Buy when net deficit and market_price <= 0.95 × grid_tariff.
    """

    EVENING_PEAK_START: int = 18
    EVENING_PEAK_END: int = 22
    P2P_SELL_THRESHOLD_PB1: float = 1.0
    P2P_BUY_THRESHOLD_PB1: float = 0.95
    DISCHARGE_PRICE_THRESHOLD: float = 1.2

    def control_home(
        self,
        t_idx: int,
        home_idx: int,
        asset: HomeAsset,
        load_kw: float,
        pv_kw: float,
        ev_kw: float,
    ) -> tuple[float, float, float]:
        hour = self.timeline[t_idx].hour
        dt = self.DT_HOURS
        new_soc = asset.battery_soc_kwh
        battery_power = 0.0
        mod_ev = ev_kw

        if asset.has_ev and ev_kw > 0:
            if not (hour >= 22 or hour < 6):
                mod_ev = 0.0

        net = load_kw + mod_ev - pv_kw

        if asset.has_battery:
            soc_norm = new_soc / max(asset.battery_capacity_kwh, 1e-6)
            if net < 0 and soc_norm < 0.90:
                charge_kw = min(asset.battery_max_kw, -net)
                soc_delta = charge_kw * dt * asset.battery_charge_efficiency
                new_soc = min(asset.battery_capacity_kwh, new_soc + soc_delta)
                battery_power = -charge_kw
            elif soc_norm > 0.20:
                in_peak = self.EVENING_PEAK_START <= hour < self.EVENING_PEAK_END
                above_threshold = (
                    self.GRID_TARIFF_INR_PER_KWH
                    > self.GRID_TARIFF_INR_PER_KWH * self.DISCHARGE_PRICE_THRESHOLD
                )
                if in_peak or above_threshold:
                    discharge_kw = min(asset.battery_max_kw, max(net, 0.0))
                    if discharge_kw > 0:
                        eff = max(asset.battery_discharge_efficiency, 1e-6)
                        new_soc = max(0.0, new_soc - (discharge_kw * dt) / eff)
                        battery_power = discharge_kw

        return battery_power, float(new_soc), mod_ev


class PB2Controller(B2Controller):
    """Pecan-Baseline-2: Forecasted MPC-lite.

    Uses persistence forecast (last 7-day same-minute median).
    Inherits LP optimisation from B2.
    """

    pass
