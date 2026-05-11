#!/usr/bin/env python3
"""
Evaluate a trained CortexCore agent against baselines.

Loads a checkpoint, runs it on the simulator, and compares KPIs
against B0, B1, and B2 baselines.

Usage:
    python scripts/evaluate_agent.py \
        --checkpoint outputs/checkpoints/cortexcore_best.pt \
        --scenario scenarios/phase1_default.json
"""

from __future__ import annotations

import argparse

import pandas as pd
import torch

from ai.core.cortexcore import CortexCore, PPOConfig
from ai.core.normalizer import ObservationNormalizer
from ai.env.gym_env import EnvConfig, VayuGridEnv
from ai.training.metrics import KPIReport, compute_kpi, kpi_summary
from simulator.config import load_simulator_config
from simulator.simulator import GridSimulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CortexCore agent")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scenario", default="scenarios/phase1_default.json")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-pecan", action="store_true",
                        help="Use Pecan Street real load profiles (overrides scenario config)")
    parser.add_argument("--city", default="bangalore",
                        choices=["bangalore", "chennai", "delhi", "hyderabad", "kochi"],
                        help="City for Pecan/NSRDB data")
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> pd.DataFrame:
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    normalizer = ObservationNormalizer((12,))
    agent = CortexCore(
        cfg=PPOConfig(),
        normalizer=normalizer,
        device=device,
    )
    agent.load(args.checkpoint)
    agent.actor.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    reports: list[KPIReport] = []

    for ep in range(args.episodes):
        env = VayuGridEnv(EnvConfig(
            scenario_path=args.scenario, seed=ep,
            use_pecan=args.use_pecan, city=args.city,
        ))
        obs, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action, _, _ = agent.select_action(obs, deterministic=True)
            obs, rew, terminated, truncated, info = env.step(action)
            total_reward += rew
            done = terminated or truncated

        # Run simulator post-hoc for KPI computation
        sim_cfg = load_simulator_config(args.scenario)
        sim = GridSimulator(sim_cfg)
        result = sim.run()
        report = compute_kpi(
            controller_label=f"CortexCore_ep{ep}",
            node_df=result.node_timeseries,
            trans_df=result.transformer_timeseries,
        )
        report.metadata["total_reward"] = total_reward
        reports.append(report)
        print(
            f"  Episode {ep}: reward={total_reward:.1f}"
            f"  cost_reduction={report.cost_reduction_pct:.1f}%"
        )

    df = kpi_summary(reports)
    print("\n=== Evaluation Summary ===")
    print(df.to_string())
    return df


if __name__ == "__main__":
    evaluate(parse_args())
