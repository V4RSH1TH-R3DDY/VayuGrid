from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ai.env.gym_env import EnvConfig, VayuGridEnv
from ai.gnn.dataset import GNNSample, GraphDatasetGenerator
from simulator.config import LoadProfileConfig, SimulatorConfig, simulator_config_from_dict
from simulator.load_profiles import LoadProfileLibrary
from simulator.models import HomeAsset


def _make_minimal_pecan_csv(path: Path, num_homes: int = 10, num_rows: int = 60) -> Path:
    """Create a minimal Pecan-style CSV for testing."""
    rows = []
    for h in range(num_homes):
        for t in range(num_rows):
            rows.append({
                "timestamp_ist": f"2019-07-01 {t // 60:02d}:{t % 60:02d}:00",
                "home_id": 1000 + h,
                "source_region": "newyork",
                "load_kw": 0.5 + 0.1 * np.sin(t),
                "pv_kw": 1.0 + 0.2 * np.sin(t),
                "ev_kw": 0.3 + 0.05 * np.cos(t),
                "battery_kw": 0.0,
                "grid_kw": 0.0,
                "target_city": "bangalore",
            })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


def _make_asset(node_id: int, **kwargs: bool) -> HomeAsset:
    defaults: dict = dict(
        battery_node_id=None, has_solar=True, has_battery=True,
        has_ev=True, pv_capacity_kw=3.0, battery_capacity_kwh=10.0,
        battery_max_kw=2.5, battery_soc_kwh=5.0, battery_charge_efficiency=0.93,
        battery_discharge_efficiency=0.93, ev_max_charge_kw=3.3, ev_daily_kwh=9.0,
    )
    defaults.update(kwargs)
    return HomeAsset(node_id=node_id, **defaults)


class PecanDataTest(unittest.TestCase):
    """Tests for the Pecan Street real-profile data path."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.csv_path = _make_minimal_pecan_csv(self.temp_dir / "test_pecan.csv")

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_generate_from_pecan_returns_correct_shapes(self) -> None:
        cfg = LoadProfileConfig(
            use_pecan_profiles=True,
            pecan_profile_file=str(self.csv_path),
            replace_solar_with_nsrdb=False,
        )
        library = LoadProfileLibrary(cfg, random_seed=42)
        time_index = pd.date_range("2019-07-01", periods=10, freq="h")
        assets = [_make_asset(i + 1) for i in range(5)]
        mats = library.generate_profiles(time_index, assets)
        expected_shape = (len(time_index), len(assets))
        self.assertEqual(mats.load_kw.shape, expected_shape)
        self.assertEqual(mats.pv_kw.shape, expected_shape)
        self.assertEqual(mats.ev_kw.shape, expected_shape)
        self.assertTrue(np.all(mats.load_kw >= 0.0))
        self.assertTrue(np.all(mats.pv_kw >= 0.0))
        self.assertTrue(np.all(mats.ev_kw >= 0.0))

    def test_generate_from_pecan_raises_on_missing_file(self) -> None:
        cfg = LoadProfileConfig(
            use_pecan_profiles=True,
            pecan_profile_file="/nonexistent/path.csv",
        )
        library = LoadProfileLibrary(cfg, random_seed=42)
        with self.assertRaises(FileNotFoundError):
            library._generate_from_pecan(
                pd.date_range("2019-07-01", periods=5, freq="h"),
                [_make_asset(1)],
            )

    def test_generate_from_pecan_raises_on_too_few_homes(self) -> None:
        path = self.temp_dir / "few_homes.csv"
        _make_minimal_pecan_csv(path, num_homes=2)
        cfg = LoadProfileConfig(
            use_pecan_profiles=True,
            pecan_profile_file=str(path),
            replace_solar_with_nsrdb=False,
        )
        library = LoadProfileLibrary(cfg, random_seed=42)
        with self.assertRaises(RuntimeError):
            library._generate_from_pecan(
                pd.date_range("2019-07-01", periods=5, freq="h"),
                [_make_asset(1), _make_asset(2), _make_asset(3)],
            )


class GraphDatasetGeneratorTest(unittest.TestCase):
    """Tests for GraphDatasetGenerator with default (synthetic) data."""

    def _build_config(self, num_homes: int = 10, end_time: str | None = None) -> SimulatorConfig:
        raw: dict = {
            "start_time": "2019-07-01T00:00:00",
            "end_time": end_time or "2019-07-01T02:00:00",
            "random_seed": 7,
            "neighborhood": {"num_homes": num_homes},
            "adoption": {"solar_ratio": 0.5, "battery_ratio": 0.3, "ev_ratio": 0.2},
            "load_profile": {"replace_solar_with_nsrdb": False},
            "faults": [],
        }
        return simulator_config_from_dict(raw)

    def test_generate_returns_gnn_samples(self) -> None:
        config = self._build_config()
        generator = GraphDatasetGenerator(config)
        samples = generator.generate()
        self.assertIsInstance(samples, list)
        self.assertGreater(len(samples), 0)
        sample = samples[0]
        self.assertIsInstance(sample, GNNSample)
        self.assertEqual(len(sample.snapshots), 12)
        self.assertEqual(sample.target_overload.shape, (30,))
        self.assertEqual(sample.target_voltage.shape, (30,))
        self.assertEqual(sample.target_risk.shape, (1,))
        self.assertEqual(sample.target_duck.shape, (96,))

    def test_generate_dataset_splits_correctly(self) -> None:
        config = self._build_config(num_homes=10)
        generator = GraphDatasetGenerator(config)
        train, val, test = generator.generate_dataset(num_episodes=1)
        total = len(train) + len(val) + len(test)
        self.assertGreater(total, 0)
        self.assertAlmostEqual(len(train) / total, 0.7, delta=0.05)
        self.assertAlmostEqual(len(val) / total, 0.15, delta=0.05)
        self.assertAlmostEqual(len(test) / total, 0.15, delta=0.05)


class EnvConfigPecanTest(unittest.TestCase):
    """Tests for EnvConfig.use_pecan propagation to simulator config."""

    def test_env_config_use_pecan_overrides_sim_config(self) -> None:
        env = VayuGridEnv(EnvConfig(
            scenario_path="scenarios/phase1_default.json",
            use_pecan=True,
            city="bangalore",
        ))
        self.assertTrue(env._sim_cfg.load_profile.use_pecan_profiles)
        self.assertIn("bangalore", env._sim_cfg.load_profile.pecan_profile_file or "")
        self.assertEqual(env._sim_cfg.load_profile.city, "bangalore")

    def test_env_config_use_pecan_false_keeps_scenario(self) -> None:
        env = VayuGridEnv(EnvConfig(scenario_path="scenarios/phase1_default.json"))
        self.assertTrue(env._sim_cfg.load_profile.use_pecan_profiles)


class PecanProfileMatrixTest(unittest.TestCase):
    """Test full generate_profiles flow with Pecan enabled (no NSRDB)."""

    def test_profile_matrices_with_pecan_and_nsrdb_off(self) -> None:
        temp_dir = Path(tempfile.mkdtemp())
        try:
            csv_path = _make_minimal_pecan_csv(temp_dir / "test.csv")
            cfg = LoadProfileConfig(
                use_pecan_profiles=True,
                pecan_profile_file=str(csv_path),
                replace_solar_with_nsrdb=False,
            )
            library = LoadProfileLibrary(cfg, random_seed=42)
            time_index = pd.date_range("2019-07-01", periods=10, freq="h")
            assets = [
                _make_asset(1, has_solar=True),
                _make_asset(2, has_solar=False, has_ev=True),
            ]
            mats = library.generate_profiles(time_index, assets)
            self.assertEqual(mats.load_kw.shape, (10, 2))
            self.assertEqual(mats.pv_kw.shape, (10, 2))
            self.assertTrue(np.all(mats.pv_kw[:, 1] == 0.0))
            self.assertGreater(mats.pv_kw[:, 0].sum(), 0.0)
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
