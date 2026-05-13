from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai.baselines.controller import (
    B0Controller,
    B1Controller,
    B2Controller,
    GridController,
    PB0Controller,
    PB1Controller,
    PB2Controller,
)
from ai.baselines.metrics import compute_kpis
from simulator.config import (
    FaultConfig,
    SimulatorConfig,
    simulator_config_from_dict,
)
from simulator.faults import FAULT_GRID_OUTAGE, FAULT_OVERLOAD, FAULT_SOLAR_DROPOUT
from simulator.simulator import GridSimulator


@dataclass
class ExperimentSpec:
    """One cell in the experiment matrix."""

    city: str = "bangalore"
    num_homes: int = 30
    random_seed: int = 42
    day_type: str = "weekday"
    season: str = "summer"
    penetration: str = "default"
    fault_scenario: str = "none"
    solar_ratio: float = 0.40
    battery_ratio: float = 0.30
    ev_ratio: float = 0.25
    pecan_mode: bool = False
    pecan_profile_file: str | None = None


CITIES = ["bangalore", "kochi", "delhi", "chennai", "hyderabad"]
SEASONS: dict[str, tuple[str, str]] = {
    "summer": ("2019-05-15", "2019-05-22"),
    "monsoon": ("2019-07-15", "2019-07-22"),
    "winter": ("2019-12-01", "2019-12-08"),
}
PENETRATION_SETS: dict[str, dict[str, float]] = {
    "default": {"solar_ratio": 0.40, "battery_ratio": 0.30, "ev_ratio": 0.25},
    "stress": {"solar_ratio": 0.60, "battery_ratio": 0.20, "ev_ratio": 0.40},
}
FAULT_CONFIGS: dict[str, list[FaultConfig]] = {
    "none": [],
    "solar_dropout": [
        FaultConfig(
            event_type=FAULT_SOLAR_DROPOUT, name="cloud_transient",
            start="2019-05-17T13:00:00", end="2019-05-17T14:00:00",
            target="all", params={"drop_fraction": 0.8},
        ),
    ],
    "overload": [
        FaultConfig(
            event_type=FAULT_OVERLOAD, name="overload_stress",
            start="2019-05-17T19:00:00", end="2019-05-17T19:45:00",
            target="random_cluster", params={"load_multiplier": 1.45, "target_ratio": 0.45},
        ),
    ],
    "outage": [
        FaultConfig(
            event_type=FAULT_GRID_OUTAGE, name="grid_failure",
            start="2019-05-17T17:15:00", end="2019-05-17T18:00:00",
            target="all", params={},
        ),
    ],
}
SCALES = [10, 30, 100]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def _build_config(spec: ExperimentSpec) -> SimulatorConfig:
    season_dates = SEASONS.get(spec.season)
    if season_dates is None:
        season_dates = ("2019-05-15", "2019-05-22")
    start, end = season_dates
    raw: dict[str, Any] = {
        "start_time": f"{start}T00:00:00",
        "end_time": f"{end}T00:00:00",
        "time_step_minutes": 1,
        "random_seed": spec.random_seed,
        "neighborhood": {
            "num_homes": spec.num_homes,
            "max_supported_nodes": 500,
            "line_resistance_ohm_per_km": 0.64,
            "min_edge_length_m": 40.0,
            "max_edge_length_m": 180.0,
            "base_voltage_v": 230.0,
            "line_ampacity_a": 140.0,
        },
        "adoption": {
            "solar_ratio": spec.solar_ratio,
            "battery_ratio": spec.battery_ratio,
            "ev_ratio": spec.ev_ratio,
        },
        "load_profile": {
            "target_daily_kwh": 6.5,
            "target_daily_kwh_min": 5.0,
            "target_daily_kwh_max": 8.0,
            "gaussian_noise_sigma": 0.08,
            "afternoon_ac_gain": 0.35,
            "evening_peak_gain": 0.55,
            "festival_spike_gain": 0.45,
            "use_pecan_profiles": spec.pecan_mode,
            "pecan_profile_file": spec.pecan_profile_file or "",
            "replace_solar_with_nsrdb": spec.pecan_mode,
            "nsrdb_data_root": "data/nsrdb_himawari",
            "city": spec.city,
            "year": 2019,
        },
        "transformer": {
            "rated_power_kw": 250.0 if spec.num_homes < 100 else 500.0,
            "ambient_temp_c": 35.0,
            "delta_theta_to_r": 55.0,
            "delta_theta_hs_r": 30.0,
            "tau_to_min": 180.0,
            "tau_w_min": 10.0,
            "r_loss_ratio": 5.0,
            "n_exp": 0.8,
            "m_exp": 0.8,
            "initial_top_oil_rise_c": 20.0,
            "initial_hotspot_rise_c": 10.0,
        },
        "faults": [
            {"event_type": f.event_type, "start": f.start, "end": f.end,
             "name": f.name, "target": f.target, "params": f.params}
            for f in FAULT_CONFIGS.get(spec.fault_scenario, [])
        ],
    }
    return simulator_config_from_dict(raw)


def _build_controller(
    controller_name: str,
    config: SimulatorConfig,
    sim: GridSimulator,
) -> GridController:
    mapping = {
        "B0": B0Controller,
        "B1": B1Controller,
        "B2": B2Controller,
        "PB0": PB0Controller,
        "PB1": PB1Controller,
        "PB2": PB2Controller,
    }
    cls = mapping.get(controller_name, B0Controller)
    return cls(
        config, sim.timeline, sim.home_assets,
        sim.load_kw_matrix, sim.pv_kw_matrix, sim.ev_kw_matrix,
    )


def run_experiment(
    spec: ExperimentSpec,
    controllers: list[str] | None = None,
) -> pd.DataFrame:
    """Run a single experiment spec across specified controllers.

    Returns a DataFrame with one row per controller.
    """
    if controllers is None:
        controllers = ["B0", "B1", "B2"]

    config = _build_config(spec)
    sim = GridSimulator(config)

    rows: list[dict[str, Any]] = []
    b0_result = None

    for ctrl_name in controllers:
        if ctrl_name in ("B0", "PB0"):
            ctrl = _build_controller(ctrl_name, config, sim)
            sim.controller = ctrl
            result = sim.run()
            b0_result = result
        else:
            sim2 = GridSimulator(config)
            ctrl = _build_controller(ctrl_name, config, sim2)
            sim2.controller = ctrl
            result = sim2.run()

        kpis = compute_kpis(result, b0_result if ctrl_name != "B0" else None)

        rows.append({
            "city": spec.city.upper(),
            "controller": ctrl_name,
            "seed": spec.random_seed,
            "num_homes": spec.num_homes,
            "season": spec.season,
            "penetration": spec.penetration,
            "fault": spec.fault_scenario,
            **kpis,
        })

    return pd.DataFrame(rows)


def generate_experiment_matrix(
    cities: list[str] | None = None,
    seeds: list[int] | None = None,
    seasons: list[str] | None = None,
    scales: list[int] | None = None,
    penetrations: list[str] | None = None,
    faults: list[str] | None = None,
    controllers: list[str] | None = None,
) -> pd.DataFrame:
    """Run the full (or partial) experiment matrix.

    Returns a combined DataFrame with all results.
    """
    cities = cities or CITIES[:1]
    seeds = seeds or DEFAULT_SEEDS[:1]
    seasons = seasons or ["summer"]
    scales = scales or [30]
    penetrations = penetrations or ["default"]
    faults = faults or ["none"]
    controllers = controllers or ["B0", "B1"]

    all_rows: list[pd.DataFrame] = []

    for city, seed, season, n_homes, pen, fault in itertools.product(
        cities, seeds, seasons, scales, penetrations, faults,
    ):
        pen_set = PENETRATION_SETS.get(pen, PENETRATION_SETS["default"])
        spec = ExperimentSpec(
            city=city,
            num_homes=n_homes,
            random_seed=seed,
            season=season,
            penetration=pen,
            fault_scenario=fault,
            solar_ratio=pen_set["solar_ratio"],
            battery_ratio=pen_set["battery_ratio"],
            ev_ratio=pen_set["ev_ratio"],
        )
        df = run_experiment(spec, controllers)
        all_rows.append(df)

    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results by city and controller with mean ± std."""
    required = {"city", "controller"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    numeric_cols = [
        "curtailment_pct", "peak_demand_kw", "peak_reduction_pct",
        "total_cost_inr", "cost_reduction_pct", "overload_events",
        "island_switchover_seconds",
    ]
    available = [c for c in numeric_cols if c in df.columns]

    grouped = df.groupby(["city", "controller"])[available]
    result = grouped.agg(["mean", "std"]).round(2)
    result.columns = [f"{col}_{stat}" for col, stat in result.columns]
    return result.reset_index()


def print_summary(df: pd.DataFrame) -> None:
    """Print formatted summary table."""
    summary = summary_table(df)
    if summary.empty:
        print("No results to summarise.")
        return

    print("\n" + "=" * 90)
    print("  BASELINE BENCHMARK SUMMARY")
    print("=" * 90)

    def _fmt(row: pd.Series, col_base: str) -> str:  # noqa: ANN001
        m = row.get(f"{col_base}_mean")
        s = row.get(f"{col_base}_std")
        if m is None:
            return "N/A"
        return f"{float(m):>6.1f} ± {float(s):>4.1f}" if s else f"{float(m):>6.1f}"

    for _, row in summary.iterrows():
        city = row["city"]
        ctrl = row["controller"]
        curt = _fmt(row, "curtailment_pct")
        peak_red = _fmt(row, "peak_reduction_pct")
        cost_red = _fmt(row, "cost_reduction_pct")
        ov = int(row.get("overload_events_mean", 0)) if "overload_events_mean" in row else 0
        line = f"  {city:<10} {ctrl:<5} | Curtail: {curt}% | "
        line += f"Peak Red: {peak_red}% | Cost Red: {cost_red}% | Overloads: {ov}"
        print(line)

    print("=" * 90)

    cv_cols = [c for c in summary.columns if c.endswith("_mean") and c != "overload_events_mean"]
    for col in cv_cols:
        vals = summary[col].dropna()
        if len(vals) > 1 and abs(np.mean(vals)) > 1e-6:
            cv = np.std(vals) / abs(np.mean(vals))
            print(f"  CV({col.replace('_mean', '')}) = {cv:.3f}")
    print()


def export_results(df: pd.DataFrame, path: str = "outputs/baseline_results.csv") -> None:
    """Save results to CSV."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Results saved to {out}")


class BaselineRunner:
    """Legacy wrapper for backward compatibility."""

    CITIES = CITIES
    SEEDS = DEFAULT_SEEDS

    def __init__(self, scenario_path: str | Path = "scenarios/phase1_default.json") -> None:
        self.scenario_path = Path(scenario_path)

    def run_all(self) -> pd.DataFrame:
        return generate_experiment_matrix(
            cities=self.CITIES[:1],
            seeds=self.SEEDS[:1],
            scales=[30],
            controllers=["B0", "B1", "B2"],
        )

    def summary_table(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        if df is None:
            df = self.run_all()
        return summary_table(df)

    def print_summary(self, df: pd.DataFrame | None = None) -> None:
        if df is None:
            df = self.run_all()
        print_summary(df)
