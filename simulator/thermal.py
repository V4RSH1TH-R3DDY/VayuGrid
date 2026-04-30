from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class TransformerThermalState:
    top_oil_rise_c: float
    hotspot_rise_c: float
    hottest_spot_temp_c: float
    aging_acceleration: float
    cumulative_loss_of_life_hours: float


class IEEETransformerThermalModel:
    def __init__(
        self,
        rated_power_kw: float,
        ambient_temp_c: float,
        delta_theta_to_r: float,
        delta_theta_hs_r: float,
        tau_to_min: float,
        tau_w_min: float,
        r_loss_ratio: float,
        n_exp: float,
        m_exp: float,
        initial_top_oil_rise_c: float,
        initial_hotspot_rise_c: float,
    ) -> None:
        self.rated_power_kw = rated_power_kw
        self.ambient_temp_c = ambient_temp_c
        self.delta_theta_to_r = delta_theta_to_r
        self.delta_theta_hs_r = delta_theta_hs_r
        self.tau_to_min = tau_to_min
        self.tau_w_min = tau_w_min
        self.r_loss_ratio = r_loss_ratio
        self.n_exp = n_exp
        self.m_exp = m_exp

        self._top_oil_rise_c = initial_top_oil_rise_c
        self._hotspot_rise_c = initial_hotspot_rise_c
        self._loss_of_life_hours = 0.0

    def _aging_acceleration_factor(self, hottest_spot_temp_c: float) -> float:
        return math.exp((15000.0 / 383.0) - (15000.0 / (hottest_spot_temp_c + 273.0)))

    def update(self, loading_pu: float, dt_minutes: float) -> TransformerThermalState:
        loading_pu = max(0.0, loading_pu)
        load_term = ((loading_pu**2) * self.r_loss_ratio + 1.0) / (self.r_loss_ratio + 1.0)

        steady_to_rise = self.delta_theta_to_r * (load_term**self.n_exp)
        steady_hs_rise = self.delta_theta_hs_r * (loading_pu ** (2.0 * self.m_exp))

        top_oil_alpha = 1.0 - math.exp(-dt_minutes / max(self.tau_to_min, 1e-6))
        hotspot_alpha = 1.0 - math.exp(-dt_minutes / max(self.tau_w_min, 1e-6))

        self._top_oil_rise_c += top_oil_alpha * (steady_to_rise - self._top_oil_rise_c)
        self._hotspot_rise_c += hotspot_alpha * (steady_hs_rise - self._hotspot_rise_c)

        hottest_spot_temp_c = self.ambient_temp_c + self._top_oil_rise_c + self._hotspot_rise_c
        aging_acceleration = self._aging_acceleration_factor(hottest_spot_temp_c)
        self._loss_of_life_hours += aging_acceleration * (dt_minutes / 60.0)

        return TransformerThermalState(
            top_oil_rise_c=self._top_oil_rise_c,
            hotspot_rise_c=self._hotspot_rise_c,
            hottest_spot_temp_c=hottest_spot_temp_c,
            aging_acceleration=aging_acceleration,
            cumulative_loss_of_life_hours=self._loss_of_life_hours,
        )
