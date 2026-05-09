from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai.baselines.b0 import B0Controller
from ai.baselines.b1 import B1Controller
from ai.baselines.b2 import B2Controller
from simulator.config import load_simulator_config


class BaselineRunner:
    """Runs all baselines (B0, B1, B2) across a scenario and returns KPI table."""

    CITIES = ["bangalore", "kochi", "delhi", "chennai", "hyderabad"]
    SEEDS = [42, 43, 44, 45, 46]

    def __init__(self, scenario_path: str | Path = "scenarios/phase1_default.json") -> None:
        self.scenario_path = Path(scenario_path)

    def run_all(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []

        for city in self.CITIES:
            for seed in self.SEEDS:
                cfg = load_simulator_config(str(self.scenario_path))
                cfg.load_profile.city = city
                cfg.random_seed = seed

                b0 = B0Controller(cfg)
                b0_result = b0.run()

                b1 = B1Controller(cfg)
                sim = getattr(b1, "sim", None)
                if sim:
                    b1_result = b1.run(
                        sim.timeline, sim.home_assets,
                        sim.load_kw_matrix, sim.pv_kw_matrix, sim.ev_kw_matrix,
                    )

                b2 = B2Controller(cfg)
                if sim:
                    b2_result = b2.run(
                        sim.timeline, sim.home_assets,
                        sim.load_kw_matrix, sim.pv_kw_matrix, sim.ev_kw_matrix,
                    )

                for label, result in [("B0", b0_result), ("B1", b1_result), ("B2", b2_result)]:
                    rows.append({
                        "city": city.upper(),
                        "controller": label,
                        "seed": seed,
                        **{k: v for k, v in result.items() if k != "controller"},
                    })

        return pd.DataFrame(rows)

    def summary_table(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        if df is None:
            df = self.run_all()
        group_cols = ["city", "controller"]
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                        if c not in {"seed"} and not c.startswith("overload")]
        return df.groupby(group_cols)[numeric_cols].agg(["mean", "std"]).round(2)

    def print_summary(self, df: pd.DataFrame | None = None) -> None:
        if df is None:
            df = self.run_all()
        summary = self.summary_table(df)
        print("\n=== Baseline Benchmark Summary ===\n")
        print(summary.to_string())
        print()
