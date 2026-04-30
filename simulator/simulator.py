from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import json

import numpy as np
import pandas as pd

from simulator.config import FaultConfig, SimulatorConfig, load_simulator_config
from simulator.faults import (
    FAULT_GRID_OUTAGE,
    FAULT_OVERLOAD,
    FAULT_PLANNED_MAINTENANCE,
    FAULT_SOLAR_DROPOUT,
    FaultEngine,
    FaultEvent,
)
from simulator.graph import ResidentialFeederGraph
from simulator.load_profiles import LoadProfileLibrary
from simulator.models import HomeAsset, SimulationResult
from simulator.thermal import IEEETransformerThermalModel


class GridSimulator:
    def __init__(self, config: SimulatorConfig) -> None:
        self.config = config
        self.config.validate()
        self.rng = np.random.default_rng(config.random_seed)

        self.start_time = datetime.fromisoformat(config.start_time)
        self.end_time = datetime.fromisoformat(config.end_time)
        self.dt_minutes = config.time_step_minutes
        self.dt_hours = self.dt_minutes / 60.0

        self.timeline = pd.date_range(
            start=self.start_time,
            end=self.end_time,
            freq=f"{self.dt_minutes}min",
            inclusive="left",
        )

        self.home_assets = self._build_home_assets()
        home_has_battery = np.array([asset.has_battery for asset in self.home_assets], dtype=bool)

        self.graph = ResidentialFeederGraph.build_random_radial(
            num_homes=config.neighborhood.num_homes,
            home_has_battery=home_has_battery,
            resistance_ohm_per_km=config.neighborhood.line_resistance_ohm_per_km,
            min_edge_length_m=config.neighborhood.min_edge_length_m,
            max_edge_length_m=config.neighborhood.max_edge_length_m,
            base_voltage_v=config.neighborhood.base_voltage_v,
            line_ampacity_a=config.neighborhood.line_ampacity_a,
            random_seed=config.random_seed,
        )
        self._bind_graph_nodes_to_assets()

        profile_library = LoadProfileLibrary(
            config=config.load_profile,
            random_seed=config.random_seed,
        )
        profiles = profile_library.generate_profiles(self.timeline, self.home_assets)
        self.load_kw_matrix = profiles.load_kw
        self.pv_kw_matrix = profiles.pv_kw
        self.ev_kw_matrix = profiles.ev_kw

        self.fault_engine = FaultEngine(
            events=self._fault_events_from_config(config.faults),
            num_homes=config.neighborhood.num_homes,
            random_seed=config.random_seed,
        )

        tr = config.transformer
        self.transformer_model = IEEETransformerThermalModel(
            rated_power_kw=tr.rated_power_kw,
            ambient_temp_c=tr.ambient_temp_c,
            delta_theta_to_r=tr.delta_theta_to_r,
            delta_theta_hs_r=tr.delta_theta_hs_r,
            tau_to_min=tr.tau_to_min,
            tau_w_min=tr.tau_w_min,
            r_loss_ratio=tr.r_loss_ratio,
            n_exp=tr.n_exp,
            m_exp=tr.m_exp,
            initial_top_oil_rise_c=tr.initial_top_oil_rise_c,
            initial_hotspot_rise_c=tr.initial_hotspot_rise_c,
        )

        self._overload_streak = 0
        self._overload_event_count = 0
        self._cooldown_remaining = 0

    def _build_home_assets(self) -> list[HomeAsset]:
        n = self.config.neighborhood.num_homes
        adoption = self.config.adoption

        has_solar = self.rng.random(n) < adoption.solar_ratio
        has_battery = self.rng.random(n) < adoption.battery_ratio
        has_ev = self.rng.random(n) < adoption.ev_ratio

        assets: list[HomeAsset] = []
        for idx in range(n):
            pv_capacity = float(self.rng.uniform(2.0, 5.5)) if has_solar[idx] else 0.0
            battery_capacity = float(self.rng.uniform(4.0, 12.0)) if has_battery[idx] else 0.0
            battery_max_kw = float(self.rng.uniform(2.0, 5.0)) if has_battery[idx] else 0.0
            battery_soc_init = (
                battery_capacity * float(self.rng.uniform(0.35, 0.75)) if has_battery[idx] else 0.0
            )
            ev_max_kw = float(self.rng.uniform(2.2, 3.3)) if has_ev[idx] else 0.0
            ev_daily_kwh = float(self.rng.uniform(3.0, 9.0)) if has_ev[idx] else 0.0

            assets.append(
                HomeAsset(
                    node_id=idx + 1,
                    battery_node_id=None,
                    has_solar=bool(has_solar[idx]),
                    has_battery=bool(has_battery[idx]),
                    has_ev=bool(has_ev[idx]),
                    pv_capacity_kw=pv_capacity,
                    battery_capacity_kwh=battery_capacity,
                    battery_max_kw=battery_max_kw,
                    battery_soc_kwh=battery_soc_init,
                    battery_charge_efficiency=0.96,
                    battery_discharge_efficiency=0.96,
                    ev_max_charge_kw=ev_max_kw,
                    ev_daily_kwh=ev_daily_kwh,
                )
            )

        return assets

    def _bind_graph_nodes_to_assets(self) -> None:
        for idx, asset in enumerate(self.home_assets):
            battery_node = int(self.graph.battery_node_id_by_home[idx])
            asset.battery_node_id = battery_node if battery_node >= 0 else None

    def _fault_events_from_config(self, fault_configs: list[FaultConfig]) -> list[FaultEvent]:
        events: list[FaultEvent] = []
        for item in fault_configs:
            if item.event_type not in {
                FAULT_OVERLOAD,
                FAULT_SOLAR_DROPOUT,
                FAULT_GRID_OUTAGE,
                FAULT_PLANNED_MAINTENANCE,
            }:
                raise ValueError(f"Unsupported fault type: {item.event_type}")

            events.append(
                FaultEvent(
                    name=item.name or item.event_type,
                    event_type=item.event_type,
                    start=datetime.fromisoformat(item.start),
                    end=datetime.fromisoformat(item.end),
                    target=item.target,
                    params=item.params,
                )
            )

        return events

    def _dispatch_battery(
        self,
        asset: HomeAsset,
        load_kw: float,
        pv_kw: float,
        ev_kw: float,
    ) -> tuple[float, float, float]:
        if not asset.has_battery:
            return 0.0, asset.battery_soc_kwh, 0.0

        net_without_battery = load_kw + ev_kw - pv_kw
        soc = asset.battery_soc_kwh
        battery_power_kw = 0.0

        if net_without_battery > 0:
            max_discharge_by_soc = (soc * asset.battery_discharge_efficiency) / max(
                self.dt_hours,
                1e-9,
            )
            discharge_kw = min(asset.battery_max_kw, max_discharge_by_soc, net_without_battery)
            soc -= (discharge_kw * self.dt_hours) / max(
                asset.battery_discharge_efficiency,
                1e-9,
            )
            battery_power_kw = discharge_kw
        elif net_without_battery < 0:
            room_kwh = max(0.0, asset.battery_capacity_kwh - soc)
            max_charge_by_room = room_kwh / max(
                asset.battery_charge_efficiency * self.dt_hours,
                1e-9,
            )
            charge_kw = min(asset.battery_max_kw, max_charge_by_room, -net_without_battery)
            soc += charge_kw * self.dt_hours * asset.battery_charge_efficiency
            battery_power_kw = -charge_kw

        soc = float(np.clip(soc, 0.0, asset.battery_capacity_kwh))
        throughput_kwh = abs(battery_power_kw) * self.dt_hours
        return battery_power_kw, soc, throughput_kwh

    def _append_home_and_battery_rows(
        self,
        node_rows: list[dict[str, object]],
        timestamp: pd.Timestamp,
        load_kw: np.ndarray,
        pv_kw: np.ndarray,
        ev_kw: np.ndarray,
        battery_power: np.ndarray,
        battery_soc: np.ndarray,
        net_grid_kw: np.ndarray,
        voltage_pu: np.ndarray,
        branch_flow_kw: np.ndarray,
        unserved_load_kw: np.ndarray,
        islanding_triggered: bool,
        maintenance_mode: bool,
        active_faults: list[str],
    ) -> None:
        active_faults_str = ",".join(active_faults)

        for idx, asset in enumerate(self.home_assets):
            node_rows.append(
                {
                    "timestamp": timestamp,
                    "node_id": asset.node_id,
                    "node_type": "home",
                    "battery_node_id": asset.battery_node_id,
                    "load_kw": float(load_kw[idx]),
                    "pv_kw": float(pv_kw[idx]),
                    "ev_kw": float(ev_kw[idx]),
                    "battery_power_kw": float(battery_power[idx]),
                    "battery_soc_kwh": float(battery_soc[idx]),
                    "net_grid_kw": float(net_grid_kw[idx]),
                    "voltage_pu": float(voltage_pu[idx]),
                    "line_flow_kw": float(branch_flow_kw[idx]),
                    "unserved_load_kw": float(unserved_load_kw[idx]),
                    "islanding_active": bool(islanding_triggered),
                    "maintenance_mode": bool(maintenance_mode),
                    "active_faults": active_faults_str,
                }
            )

            if asset.battery_node_id is None:
                continue

            battery_net_grid_kw = -battery_power[idx]
            node_rows.append(
                {
                    "timestamp": timestamp,
                    "node_id": asset.battery_node_id,
                    "node_type": "battery",
                    "battery_node_id": None,
                    "load_kw": 0.0,
                    "pv_kw": 0.0,
                    "ev_kw": 0.0,
                    "battery_power_kw": float(battery_power[idx]),
                    "battery_soc_kwh": float(battery_soc[idx]),
                    "net_grid_kw": float(battery_net_grid_kw),
                    "voltage_pu": float(voltage_pu[idx]),
                    "line_flow_kw": float(battery_net_grid_kw),
                    "unserved_load_kw": 0.0,
                    "islanding_active": bool(islanding_triggered),
                    "maintenance_mode": bool(maintenance_mode),
                    "active_faults": active_faults_str,
                }
            )

    def run(self) -> SimulationResult:
        node_rows: list[dict[str, object]] = []
        transformer_rows: list[dict[str, object]] = []
        event_rows: list[dict[str, object]] = []

        for t_idx, timestamp in enumerate(self.timeline):
            base_load = self.load_kw_matrix[t_idx, :]
            base_pv = self.pv_kw_matrix[t_idx, :]
            base_ev = self.ev_kw_matrix[t_idx, :]

            fault_state = self.fault_engine.apply(timestamp.to_pydatetime(), base_load, base_pv)
            load_kw = fault_state.load_kw
            pv_kw = fault_state.pv_kw
            ev_kw = base_ev

            battery_power = np.zeros(len(self.home_assets), dtype=float)
            battery_soc = np.zeros(len(self.home_assets), dtype=float)
            for idx, asset in enumerate(self.home_assets):
                batt_kw, soc, _ = self._dispatch_battery(
                    asset,
                    load_kw[idx],
                    pv_kw[idx],
                    ev_kw[idx],
                )
                battery_power[idx] = batt_kw
                battery_soc[idx] = soc
                asset.battery_soc_kwh = soc

            net_grid_kw = load_kw + ev_kw - pv_kw - battery_power

            islanding_triggered = False
            unserved_load_kw = np.zeros(len(self.home_assets), dtype=float)

            if not fault_state.grid_available:
                if fault_state.islanding_allowed and not fault_state.maintenance_mode:
                    islanding_triggered = True
                    local_supply = pv_kw + np.clip(battery_power, 0.0, None)
                    local_demand = load_kw + ev_kw
                    unserved_load_kw = np.clip(local_demand - local_supply, 0.0, None)
                net_grid_kw[:] = 0.0

            if fault_state.grid_available:
                (
                    voltage_pu,
                    branch_flow_kw,
                    feeder_kw,
                    max_branch_loading_pu,
                ) = self.graph.compute_network_state(net_grid_kw)
                voltage_pu = np.clip(voltage_pu, 0.85, 1.08)
            else:
                branch_flow_kw = np.zeros_like(net_grid_kw)
                feeder_kw = 0.0
                max_branch_loading_pu = 0.0
                voltage_pu = np.full(len(self.home_assets), 0.99)
                if islanding_triggered:
                    stress = np.divide(
                        unserved_load_kw,
                        np.maximum(load_kw + ev_kw, 1e-6),
                        out=np.zeros_like(unserved_load_kw),
                        where=(load_kw + ev_kw) > 1e-6,
                    )
                    voltage_pu = np.clip(voltage_pu - 0.10 * np.clip(stress, 0.0, 1.0), 0.85, 1.05)

            rated_kw = self.config.transformer.rated_power_kw
            loading_pu = max(0.0, feeder_kw / max(rated_kw, 1e-9))

            if loading_pu > 1.2:
                self._overload_streak += 1
            else:
                self._overload_streak = 0

            if self._cooldown_remaining > 0:
                self._cooldown_remaining -= 1

            if self._overload_streak >= 5 and self._cooldown_remaining == 0:
                self._overload_event_count += 1
                self._cooldown_remaining = 10
                event_rows.append(
                    {
                        "timestamp": timestamp,
                        "event_type": "transformer_overload_detected",
                        "details": "loading > 1.2 pu for 5 consecutive minutes",
                    }
                )

            thermal_state = self.transformer_model.update(
                loading_pu=loading_pu,
                dt_minutes=self.dt_minutes,
            )

            self._append_home_and_battery_rows(
                node_rows=node_rows,
                timestamp=timestamp,
                load_kw=load_kw,
                pv_kw=pv_kw,
                ev_kw=ev_kw,
                battery_power=battery_power,
                battery_soc=battery_soc,
                net_grid_kw=net_grid_kw,
                voltage_pu=voltage_pu,
                branch_flow_kw=branch_flow_kw,
                unserved_load_kw=unserved_load_kw,
                islanding_triggered=islanding_triggered,
                maintenance_mode=fault_state.maintenance_mode,
                active_faults=fault_state.active_faults,
            )

            transformer_rows.append(
                {
                    "timestamp": timestamp,
                    "feeder_total_kw": float(feeder_kw),
                    "transformer_loading_pu": float(loading_pu),
                    "max_branch_loading_pu": float(max_branch_loading_pu),
                    "top_oil_rise_c": float(thermal_state.top_oil_rise_c),
                    "hotspot_rise_c": float(thermal_state.hotspot_rise_c),
                    "hottest_spot_temp_c": float(thermal_state.hottest_spot_temp_c),
                    "aging_acceleration": float(thermal_state.aging_acceleration),
                    "cumulative_loss_of_life_hours": float(
                        thermal_state.cumulative_loss_of_life_hours
                    ),
                    "overload_event_count": int(self._overload_event_count),
                    "grid_available": bool(fault_state.grid_available),
                    "islanding_triggered": bool(islanding_triggered),
                    "maintenance_mode": bool(fault_state.maintenance_mode),
                }
            )

            for fault_type in fault_state.active_faults:
                event_rows.append(
                    {
                        "timestamp": timestamp,
                        "event_type": fault_type,
                        "details": json.dumps(
                            {
                                "grid_available": fault_state.grid_available,
                                "islanding_allowed": fault_state.islanding_allowed,
                                "maintenance_mode": fault_state.maintenance_mode,
                            },
                            sort_keys=True,
                        ),
                    }
                )

        node_df = pd.DataFrame(node_rows)
        transformer_df = pd.DataFrame(transformer_rows)
        event_df = pd.DataFrame(event_rows).drop_duplicates()

        metadata = {
            "num_homes": self.config.neighborhood.num_homes,
            "start_time": self.config.start_time,
            "end_time": self.config.end_time,
            "time_step_minutes": self.config.time_step_minutes,
            "city": self.config.load_profile.city,
            "year": self.config.load_profile.year,
            "adoption": asdict(self.config.adoption),
            "load_profile": asdict(self.config.load_profile),
            "transformer": asdict(self.config.transformer),
            "graph_total_nodes": self.graph.num_nodes,
            "graph_home_nodes": self.graph.num_homes,
            "graph_battery_nodes": int(np.sum(self.graph.battery_node_id_by_home >= 0)),
            "home_assets": [
                {
                    "node_id": asset.node_id,
                    "battery_node_id": asset.battery_node_id,
                    "has_solar": asset.has_solar,
                    "has_battery": asset.has_battery,
                    "has_ev": asset.has_ev,
                    "pv_capacity_kw": asset.pv_capacity_kw,
                    "battery_capacity_kwh": asset.battery_capacity_kwh,
                    "battery_max_kw": asset.battery_max_kw,
                    "ev_max_charge_kw": asset.ev_max_charge_kw,
                    "ev_daily_kwh": asset.ev_daily_kwh,
                }
                for asset in self.home_assets
            ],
        }

        return SimulationResult(
            node_timeseries=node_df,
            transformer_timeseries=transformer_df,
            event_log=event_df,
            metadata=metadata,
        )


def run_simulation_from_config(config_path: str | Path) -> SimulationResult:
    config = load_simulator_config(config_path)
    simulator = GridSimulator(config)
    return simulator.run()
