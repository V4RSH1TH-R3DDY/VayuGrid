from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from simulator.config import load_simulator_config
from simulator.models import HomeAsset
from simulator.simulator import GridSimulator

if TYPE_CHECKING:
    import gymnasium as gym
    from gymnasium import spaces

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]


OBS_DIM = 12  # matches CortexCore ObsIdx layout


@dataclass
class EnvConfig:
    scenario_path: str | Path = "scenarios/phase1_default.json"
    episode_minutes: int = 1440
    node_id: int = 1
    seed: int = 42
    use_pecan: bool = False
    city: str = "bangalore"


class VayuGridEnv:
    """Gymnasium environment wrapping the VayuGrid GridSimulator.

    Each environment steps one home (``node_id``) through the simulation.
    The observation is the home's ``NodeState`` vector; the action controls
    battery charge/discharge, EV charge rate, and P2P bid/ask prices.
    """

    def __init__(self, config: EnvConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        self._sim_cfg = load_simulator_config(config.scenario_path)
        if config.use_pecan:
            pecan_path = (
                f"data/processed/pecan_india/{config.city}/2019/"
                f"pecan_wired_{config.city}_2019.csv"
            )
            self._sim_cfg.load_profile.use_pecan_profiles = True
            self._sim_cfg.load_profile.pecan_profile_file = pecan_path
            self._sim_cfg.load_profile.city = config.city
        self._sim = GridSimulator(self._sim_cfg)
        self._node_id = config.node_id
        self._asset: HomeAsset | None = None
        self._t_idx = 0

        self._load_kw: float = 0.0
        self._pv_kw: float = 0.0
        self._ev_kw: float = 0.0

        low = np.full(OBS_DIM, -1e3, dtype=np.float32)
        high = np.full(OBS_DIM, 1e3, dtype=np.float32)
        obs_space = spaces.Box(low=low, high=high, dtype=np.float32) if spaces else None
        act_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32) if spaces else None
        self.observation_space = obs_space
        self.action_space = act_space

    def _locate_asset(self) -> HomeAsset:
        for a in self._sim.home_assets:
            if a.node_id == self._node_id:
                return a
        raise ValueError(f"Node {self._node_id} not found in simulator assets")

    def _observe(self) -> np.ndarray:
        assert self._asset is not None
        a = self._asset
        obs = np.array(
            [
                a.battery_soc_kwh / max(a.battery_capacity_kwh, 1e-6),   # SOC_NORM = 0
                self._pv_kw,                                               # SOLAR_KW = 1
                self._load_kw,                                             # LOAD_KW = 2
                self._ev_kw,                                               # EV_KW = 3
                a.battery_max_kw,                                          # BATT_MAX_KW = 4
                self._pv_kw - self._load_kw,                               # NET_KW = 5
                float(np.sin(2 * np.pi * self._t_idx / 1440)),              # TIME_SIN = 6
                float(np.cos(2 * np.pi * self._t_idx / 1440)),              # TIME_COS = 7
                0.0,                                                       # FCST_SOLAR_KW = 8
                0.0,                                                       # FCST_LOAD_KW = 9
                0.0,                                                       # FCST_PRICE = 10
                a.battery_soc_kwh,                                         # SOC_RAW_KWH = 11
            ],
            dtype=np.float32,
        )
        return obs

    def reset(
        self, *, seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.config.seed = seed
            self.rng = np.random.default_rng(seed)

        self._sim_cfg = load_simulator_config(self.config.scenario_path)
        if self.config.use_pecan:
            pecan_path = (
                f"data/processed/pecan_india/{self.config.city}/2019/"
                f"pecan_wired_{self.config.city}_2019.csv"
            )
            self._sim_cfg.load_profile.use_pecan_profiles = True
            self._sim_cfg.load_profile.pecan_profile_file = pecan_path
            self._sim_cfg.load_profile.city = self.config.city
        self._sim = GridSimulator(self._sim_cfg)
        self._asset = self._locate_asset()
        self._t_idx = 0

        self._load_kw = float(self._sim.load_kw_matrix[0, self._node_id - 1])
        self._pv_kw = float(self._sim.pv_kw_matrix[0, self._node_id - 1])
        self._ev_kw = float(self._sim.ev_kw_matrix[0, self._node_id - 1])

        return self._observe(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self._asset is not None
        dt = self._sim.dt_hours

        batt_setpoint = float(np.clip(action[0], -1.0, 1.0)) * self._asset.battery_max_kw
        _ = float(np.clip(action[1], 0.0, 1.0)) * self._asset.ev_max_charge_kw
        ask_price = float(np.clip(action[3], 0.0, 1.0)) * 20.0

        net = self._load_kw + self._ev_kw - self._pv_kw

        if batt_setpoint < 0 and net > 0:
            discharge_kw = min(-batt_setpoint, net, self._asset.battery_max_kw)
            soc_delta = (discharge_kw * dt) / max(self._asset.battery_discharge_efficiency, 1e-6)
            self._asset.battery_soc_kwh = max(0.0, self._asset.battery_soc_kwh - soc_delta)
            batt_power_kw = -discharge_kw
        elif batt_setpoint > 0 and net < 0:
            charge_kw = min(batt_setpoint, -net, self._asset.battery_max_kw)
            soc_delta = charge_kw * dt * self._asset.battery_charge_efficiency
            self._asset.battery_soc_kwh = min(
                self._asset.battery_capacity_kwh,
                self._asset.battery_soc_kwh + soc_delta,
            )
            batt_power_kw = charge_kw
        else:
            batt_power_kw = 0.0

        grid_kw = net + batt_power_kw

        grid_cost = max(0.0, grid_kw * dt * 8.0)
        p2p_revenue = 0.0
        if grid_kw < 0:
            if ask_price >= 8.0:
                p2p_revenue = -grid_kw * dt * ask_price

        ev_charge_cost = self._ev_kw * dt * 8.0
        total_cost = grid_cost + ev_charge_cost - p2p_revenue

        battery_low_penalty = 0.0
        soc_norm = self._asset.battery_soc_kwh / max(self._asset.battery_capacity_kwh, 1e-6)
        if soc_norm < 0.1:
            battery_low_penalty = 50.0 * (0.1 - soc_norm)

        reward = -(total_cost + battery_low_penalty)

        self._t_idx += 1
        truncated = self._t_idx >= self.config.episode_minutes
        terminated = False

        if not truncated:
            idx = min(self._t_idx, len(self._sim.timeline) - 1)
            self._load_kw = float(self._sim.load_kw_matrix[idx, self._node_id - 1])
            self._pv_kw = float(self._sim.pv_kw_matrix[idx, self._node_id - 1])
            self._ev_kw = float(self._sim.ev_kw_matrix[idx, self._node_id - 1])

        info = {"grid_cost": grid_cost, "p2p_revenue": p2p_revenue}
        return self._observe(), reward, terminated, truncated, info

    def close(self) -> None:
        pass
