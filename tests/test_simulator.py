from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from simulator.config import simulator_config_from_dict
from simulator.simulator import GridSimulator


class GridSimulatorTest(unittest.TestCase):
    def _build_config(self) -> dict[str, object]:
        return {
            "start_time": "2019-07-01T00:00:00",
            "end_time": "2019-07-01T00:10:00",
            "random_seed": 7,
            "neighborhood": {"num_homes": 10},
            "adoption": {
                "solar_ratio": 0.5,
                "battery_ratio": 1.0,
                "ev_ratio": 0.2,
            },
            "load_profile": {
                "replace_solar_with_nsrdb": False,
                "city": "delhi",
                "cricket_match_dates": ["2019-07-01"],
            },
            "faults": [],
        }

    def test_run_emits_home_and_battery_rows(self) -> None:
        config = simulator_config_from_dict(self._build_config())
        result = GridSimulator(config).run()

        node_types = set(result.node_timeseries["node_type"].unique().tolist())
        self.assertEqual(node_types, {"home", "battery"})

        battery_node_count = result.metadata["graph_battery_nodes"]
        expected_rows_per_step = config.neighborhood.num_homes + battery_node_count
        actual_rows_per_step = result.node_timeseries.groupby("timestamp").size()
        self.assertTrue((actual_rows_per_step == expected_rows_per_step).all())

        battery_rows = result.node_timeseries[result.node_timeseries["node_type"] == "battery"]
        self.assertEqual(battery_rows["node_id"].nunique(), battery_node_count)
        self.assertTrue((battery_rows["load_kw"] == 0.0).all())
        self.assertTrue((battery_rows["pv_kw"] == 0.0).all())
        self.assertTrue((battery_rows["ev_kw"] == 0.0).all())

    def test_metadata_includes_asset_and_profile_context(self) -> None:
        config = simulator_config_from_dict(self._build_config())
        result = GridSimulator(config).run()

        self.assertIn("load_profile", result.metadata)
        self.assertIn("transformer", result.metadata)
        self.assertIn("home_assets", result.metadata)
        self.assertEqual(len(result.metadata["home_assets"]), config.neighborhood.num_homes)
        self.assertEqual(result.metadata["load_profile"]["city"], "delhi")

    def test_planned_maintenance_does_not_trigger_islanding(self) -> None:
        raw = self._build_config()
        raw["faults"] = [
            {
                "event_type": "planned_maintenance",
                "name": "scheduled_window",
                "start": "2019-07-01T00:02:00",
                "end": "2019-07-01T00:05:00",
                "target": "all",
                "params": {},
            }
        ]
        config = simulator_config_from_dict(raw)
        result = GridSimulator(config).run()

        maintenance_rows = result.transformer_timeseries[
            result.transformer_timeseries["maintenance_mode"]
        ]
        self.assertFalse(maintenance_rows.empty)
        self.assertTrue((maintenance_rows["grid_available"] == False).all())
        self.assertTrue((maintenance_rows["islanding_triggered"] == False).all())

    def test_grid_outage_triggers_islanding(self) -> None:
        raw = self._build_config()
        raw["faults"] = [
            {
                "event_type": "grid_outage",
                "name": "real_outage",
                "start": "2019-07-01T00:02:00",
                "end": "2019-07-01T00:05:00",
                "target": "all",
                "params": {},
            }
        ]
        config = simulator_config_from_dict(raw)
        result = GridSimulator(config).run()

        outage_rows = result.transformer_timeseries[
            result.transformer_timeseries["grid_available"] == False
        ]
        self.assertFalse(outage_rows.empty)
        self.assertTrue((outage_rows["islanding_triggered"] == True).all())

    def test_overload_fault_counts_transformer_overload_event(self) -> None:
        raw = self._build_config()
        raw["end_time"] = "2019-07-01T00:12:00"
        raw["transformer"] = {"rated_power_kw": 8.0}
        raw["faults"] = [
            {
                "event_type": "overload",
                "name": "forced_overload",
                "start": "2019-07-01T00:00:00",
                "end": "2019-07-01T00:10:00",
                "target": "random_cluster",
                "params": {"load_multiplier": 5.0, "target_ratio": 1.0},
            }
        ]
        config = simulator_config_from_dict(raw)
        result = GridSimulator(config).run()

        self.assertGreater(
            int(result.transformer_timeseries["overload_event_count"].max()),
            0,
        )
        overload_events = result.event_log[
            result.event_log["event_type"] == "transformer_overload_detected"
        ]
        self.assertFalse(overload_events.empty)

    def test_save_persists_metadata_json(self) -> None:
        config = simulator_config_from_dict(self._build_config())
        result = GridSimulator(config).run()

        with tempfile.TemporaryDirectory() as temp_dir:
            result.save(temp_dir)

            metadata_path = Path(temp_dir) / "metadata.json"
            self.assertTrue(metadata_path.exists())

            saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_metadata, result.metadata)


if __name__ == "__main__":
    unittest.main()
