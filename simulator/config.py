from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class AdoptionConfig:
    solar_ratio: float = 0.40
    battery_ratio: float = 0.30
    ev_ratio: float = 0.25


@dataclass
class NeighborhoodConfig:
    num_homes: int = 30
    max_supported_nodes: int = 500
    line_resistance_ohm_per_km: float = 0.64
    min_edge_length_m: float = 40.0
    max_edge_length_m: float = 180.0
    base_voltage_v: float = 230.0
    line_ampacity_a: float = 140.0


@dataclass
class LoadProfileConfig:
    target_daily_kwh: float = 6.5
    target_daily_kwh_min: float = 5.0
    target_daily_kwh_max: float = 8.0
    gaussian_noise_sigma: float = 0.08
    afternoon_ac_gain: float = 0.35
    evening_peak_gain: float = 0.55
    festival_spike_gain: float = 0.45
    cricket_spike_gain: float = 0.18
    afternoon_ac_threshold_c: float = 35.0
    use_pecan_profiles: bool = False
    pecan_profile_file: str | None = None
    replace_solar_with_nsrdb: bool = True
    nsrdb_data_root: str = "data/nsrdb_himawari"
    city: str = "bangalore"
    year: int = 2019
    festival_dates: list[str] = field(default_factory=list)
    cricket_match_dates: list[str] = field(default_factory=list)


@dataclass
class TransformerConfig:
    rated_power_kw: float = 250.0
    ambient_temp_c: float = 35.0
    delta_theta_to_r: float = 55.0
    delta_theta_hs_r: float = 30.0
    tau_to_min: float = 180.0
    tau_w_min: float = 10.0
    r_loss_ratio: float = 5.0
    n_exp: float = 0.8
    m_exp: float = 0.8
    initial_top_oil_rise_c: float = 20.0
    initial_hotspot_rise_c: float = 10.0


@dataclass
class FaultConfig:
    event_type: str
    start: str
    end: str
    name: str = ""
    target: str = "all"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulatorConfig:
    start_time: str
    end_time: str
    time_step_minutes: int = 1
    random_seed: int = 42
    neighborhood: NeighborhoodConfig = field(default_factory=NeighborhoodConfig)
    adoption: AdoptionConfig = field(default_factory=AdoptionConfig)
    load_profile: LoadProfileConfig = field(default_factory=LoadProfileConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    faults: list[FaultConfig] = field(default_factory=list)

    def validate(self) -> None:
        if self.time_step_minutes != 1:
            raise ValueError("Simulator is locked to 1-minute resolution for Phase 1")
        if self.neighborhood.num_homes < 10:
            raise ValueError("num_homes must be at least 10")
        if self.neighborhood.num_homes > self.neighborhood.max_supported_nodes:
            raise ValueError(
                f"num_homes exceeds max_supported_nodes={self.neighborhood.max_supported_nodes}"
            )
        if self.neighborhood.min_edge_length_m <= 0 or self.neighborhood.max_edge_length_m <= 0:
            raise ValueError("Edge lengths must be positive")
        if self.neighborhood.min_edge_length_m > self.neighborhood.max_edge_length_m:
            raise ValueError("min_edge_length_m cannot exceed max_edge_length_m")

        for ratio_name, ratio_value in {
            "solar_ratio": self.adoption.solar_ratio,
            "battery_ratio": self.adoption.battery_ratio,
            "ev_ratio": self.adoption.ev_ratio,
        }.items():
            if not 0.0 <= ratio_value <= 1.0:
                raise ValueError(f"{ratio_name} must be within [0, 1]")

        start_dt = datetime.fromisoformat(self.start_time)
        end_dt = datetime.fromisoformat(self.end_time)
        if end_dt <= start_dt:
            raise ValueError("end_time must be after start_time")


def _fault_from_dict(raw: dict[str, Any]) -> FaultConfig:
    return FaultConfig(
        event_type=str(raw["event_type"]),
        start=str(raw["start"]),
        end=str(raw["end"]),
        name=str(raw.get("name", "")),
        target=str(raw.get("target", "all")),
        params=dict(raw.get("params", {})),
    )


def simulator_config_from_dict(raw: dict[str, Any]) -> SimulatorConfig:
    neighborhood = NeighborhoodConfig(**raw.get("neighborhood", {}))
    adoption = AdoptionConfig(**raw.get("adoption", {}))
    load_profile = LoadProfileConfig(**raw.get("load_profile", {}))
    transformer = TransformerConfig(**raw.get("transformer", {}))
    faults = [_fault_from_dict(item) for item in raw.get("faults", [])]

    config = SimulatorConfig(
        start_time=str(raw["start_time"]),
        end_time=str(raw["end_time"]),
        time_step_minutes=int(raw.get("time_step_minutes", 1)),
        random_seed=int(raw.get("random_seed", 42)),
        neighborhood=neighborhood,
        adoption=adoption,
        load_profile=load_profile,
        transformer=transformer,
        faults=faults,
    )
    config.validate()
    return config


def load_simulator_config(config_path: str | Path) -> SimulatorConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as config_file:
        raw = json.load(config_file)
    return simulator_config_from_dict(raw)
