from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ai.baselines.b0 import B0Controller
from ai.baselines.b1 import B1Controller
from ai.baselines.b2 import B2Controller
from ai.baselines.controller import PB0Controller, PB1Controller, PB2Controller
from ai.baselines.metrics import (
    compute_cost_reduction,
    compute_curtailment,
    compute_overload_events,
    compute_peak_demand,
    compute_peak_reduction,
    compute_total_cost,
)
from ai.baselines.runner import (
    ExperimentSpec,
    generate_experiment_matrix,
    run_experiment,
    summary_table,
)
from simulator.config import simulator_config_from_dict
from simulator.models import SimulationResult
from simulator.simulator import GridSimulator


def _quick_config(num_homes: int = 10, seed: int = 7) -> dict:
    return {
        "start_time": "2019-07-01T00:00:00",
        "end_time": "2019-07-01T01:00:00",
        "random_seed": seed,
        "neighborhood": {"num_homes": num_homes},
        "adoption": {"solar_ratio": 0.5, "battery_ratio": 0.5, "ev_ratio": 0.3},
        "load_profile": {
            "replace_solar_with_nsrdb": False,
            "city": "delhi",
        },
        "faults": [],
    }


class B0ControllerTest(unittest.TestCase):
    def test_b0_disables_battery(self) -> None:
        raw = _quick_config()
        raw["adoption"]["battery_ratio"] = 0.5
        config = simulator_config_from_dict(raw)

        sim = GridSimulator(config)
        ctrl = B0Controller(config, sim.timeline, sim.home_assets,
                            sim.load_kw_matrix, sim.pv_kw_matrix, sim.ev_kw_matrix)
        sim.controller = ctrl
        result = sim.run()

        home = result.node_timeseries[result.node_timeseries["node_type"] == "home"]
        soc_max = float(home["battery_soc_kwh"].max())
        self.assertEqual(soc_max, 0.0, "B0 should keep battery at 0")

    def test_b0_runs_successfully(self) -> None:
        raw = _quick_config()
        config = simulator_config_from_dict(raw)
        sim = GridSimulator(config)
        ctrl = B0Controller(config, sim.timeline, sim.home_assets,
                            sim.load_kw_matrix, sim.pv_kw_matrix, sim.ev_kw_matrix)
        sim.controller = ctrl
        result = sim.run()
        self.assertIsInstance(result, SimulationResult)
        self.assertFalse(result.node_timeseries.empty)


class B1ControllerTest(unittest.TestCase):
    def test_b1_charges_battery_from_solar_surplus(self) -> None:
        raw = _quick_config(num_homes=10, seed=42)
        config = simulator_config_from_dict(raw)
        sim = GridSimulator(config)
        ctrl = B1Controller(config, sim.timeline, sim.home_assets,
                            sim.load_kw_matrix, sim.pv_kw_matrix, sim.ev_kw_matrix)
        sim.controller = ctrl
        result = sim.run()

        home = result.node_timeseries[result.node_timeseries["node_type"] == "home"]
        soc_values = home["battery_soc_kwh"].values
        self.assertTrue(np.any(soc_values > 0), "B1 should charge battery from solar surplus")

    def test_b1_blocks_ev_outside_window(self) -> None:
        raw = _quick_config(num_homes=10, seed=42)
        config = simulator_config_from_dict(raw)
        sim = GridSimulator(config)
        ctrl = B1Controller(config, sim.timeline, sim.home_assets,
                            sim.load_kw_matrix, sim.pv_kw_matrix, sim.ev_kw_matrix)
        sim.controller = ctrl
        result = sim.run()

        home = result.node_timeseries[result.node_timeseries["node_type"] == "home"]
        home["hour"] = pd.to_datetime(home["timestamp"]).dt.hour
        blocked = home[(home["hour"] >= 7) & (home["hour"] < 22)]
        self.assertTrue((blocked["ev_kw"] == 0.0).all(), "B1 should block EV outside 22-06 window")

    def test_b1_runs_successfully(self) -> None:
        raw = _quick_config()
        config = simulator_config_from_dict(raw)
        sim = GridSimulator(config)
        ctrl = B1Controller(config, sim.timeline, sim.home_assets,
                            sim.load_kw_matrix, sim.pv_kw_matrix, sim.ev_kw_matrix)
        sim.controller = ctrl
        result = sim.run()
        self.assertIsInstance(result, SimulationResult)
        self.assertFalse(result.node_timeseries.empty)


class B2ControllerTest(unittest.TestCase):
    def test_b2_runs_successfully(self) -> None:
        raw = _quick_config(num_homes=10, seed=7)
        config = simulator_config_from_dict(raw)
        sim = GridSimulator(config)
        ctrl = B2Controller(config, sim.timeline, sim.home_assets,
                            sim.load_kw_matrix, sim.pv_kw_matrix, sim.ev_kw_matrix)
        sim.controller = ctrl
        result = sim.run()
        self.assertIsInstance(result, SimulationResult)

    def test_b2_battery_action_nonzero(self) -> None:
        raw = _quick_config(num_homes=10, seed=7)
        raw["end_time"] = "2019-07-01T08:00:00"
        raw["adoption"]["battery_ratio"] = 1.0
        config = simulator_config_from_dict(raw)
        sim = GridSimulator(config)
        ctrl = B2Controller(config, sim.timeline, sim.home_assets,
                            sim.load_kw_matrix, sim.pv_kw_matrix, sim.ev_kw_matrix)
        sim.controller = ctrl
        result = sim.run()

        home = result.node_timeseries[result.node_timeseries["node_type"] == "home"]
        max_batt = float(abs(home["battery_power_kw"]).max())
        self.assertGreater(max_batt, 0.0, "B2 should dispatch battery")


class PBControllerTest(unittest.TestCase):
    def test_pb0_runs_successfully(self) -> None:
        raw = _quick_config()
        config = simulator_config_from_dict(raw)
        sim = GridSimulator(config)
        ctrl = PB0Controller(config, sim.timeline, sim.home_assets,
                             sim.load_kw_matrix, sim.pv_kw_matrix, sim.ev_kw_matrix)
        sim.controller = ctrl
        result = sim.run()
        self.assertIsInstance(result, SimulationResult)

    def test_pb1_runs_successfully(self) -> None:
        raw = _quick_config()
        config = simulator_config_from_dict(raw)
        sim = GridSimulator(config)
        ctrl = PB1Controller(config, sim.timeline, sim.home_assets,
                             sim.load_kw_matrix, sim.pv_kw_matrix, sim.ev_kw_matrix)
        sim.controller = ctrl
        result = sim.run()
        self.assertIsInstance(result, SimulationResult)

    def test_pb2_runs_successfully(self) -> None:
        raw = _quick_config(num_homes=10, seed=7)
        config = simulator_config_from_dict(raw)
        sim = GridSimulator(config)
        ctrl = PB2Controller(config, sim.timeline, sim.home_assets,
                             sim.load_kw_matrix, sim.pv_kw_matrix, sim.ev_kw_matrix)
        sim.controller = ctrl
        result = sim.run()
        self.assertIsInstance(result, SimulationResult)


class MetricsTest(unittest.TestCase):
    def test_curtailment_returns_float(self) -> None:
        raw = _quick_config()
        config = simulator_config_from_dict(raw)
        result = GridSimulator(config).run()
        val = compute_curtailment(result.node_timeseries)
        self.assertIsInstance(val, float)

    def test_peak_demand_returns_positive(self) -> None:
        raw = _quick_config()
        config = simulator_config_from_dict(raw)
        result = GridSimulator(config).run()
        peak = compute_peak_demand(result.node_timeseries)
        self.assertGreaterEqual(peak, 0.0)

    def test_total_cost_non_negative(self) -> None:
        raw = _quick_config()
        config = simulator_config_from_dict(raw)
        result = GridSimulator(config).run()
        cost = compute_total_cost(result.node_timeseries)
        self.assertGreaterEqual(cost, 0.0)

    def test_cost_reduction_with_baseline(self) -> None:
        raw = _quick_config()
        config = simulator_config_from_dict(raw)
        b0_result = GridSimulator(config).run()
        config2 = simulator_config_from_dict(raw)
        sim2 = GridSimulator(config2)
        ctrl = B0Controller(config2, sim2.timeline, sim2.home_assets,
                            sim2.load_kw_matrix, sim2.pv_kw_matrix, sim2.ev_kw_matrix)
        sim2.controller = ctrl
        b1_result = sim2.run()
        cost_red = compute_cost_reduction(b1_result.node_timeseries,
                                          compute_total_cost(b0_result.node_timeseries))
        self.assertIsInstance(cost_red[0], float)

    def test_overload_events_returns_int(self) -> None:
        raw = _quick_config()
        config = simulator_config_from_dict(raw)
        result = GridSimulator(config).run()
        events = compute_overload_events(result.transformer_timeseries)
        self.assertIsInstance(events, int)

    def test_peak_reduction_with_baseline(self) -> None:
        raw = _quick_config()
        config = simulator_config_from_dict(raw)
        b0_result = GridSimulator(config).run()
        b0_peak = compute_peak_demand(b0_result.node_timeseries)
        red = compute_peak_reduction(b0_result.node_timeseries, b0_peak)
        self.assertIsInstance(red, float)


class RunnerTest(unittest.TestCase):
    def test_run_experiment_single_spec(self) -> None:
        spec = ExperimentSpec(
            city="delhi", num_homes=10, random_seed=7,
            season="summer", penetration="default", fault_scenario="none",
        )
        df = run_experiment(spec, controllers=["B0", "B1"])
        self.assertEqual(len(df), 2)
        self.assertIn("controller", df.columns)
        self.assertIn("curtailment_pct", df.columns)

    def test_generate_experiment_matrix_small(self) -> None:
        df = generate_experiment_matrix(
            cities=["delhi"],
            seeds=[7],
            scales=[10],
            controllers=["B0", "B1"],
        )
        self.assertGreater(len(df), 0)

    def test_summary_table_shape(self) -> None:
        spec = ExperimentSpec(city="delhi", num_homes=10, random_seed=7)
        df = run_experiment(spec, controllers=["B0", "B1"])
        summary = summary_table(df)
        self.assertIn("city", summary.columns)
        self.assertIn("controller", summary.columns)

    def test_b2_in_experiment_matrix(self) -> None:
        spec = ExperimentSpec(city="delhi", num_homes=10, random_seed=7)
        df = run_experiment(spec, controllers=["B0", "B2"])
        self.assertEqual(len(df), 2)
        b2_row = df[df["controller"] == "B2"]
        self.assertEqual(len(b2_row), 1)


if __name__ == "__main__":
    unittest.main()
