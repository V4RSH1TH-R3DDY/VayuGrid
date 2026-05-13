#!/usr/bin/env python3
"""
VayuGrid Demo — multi-scenario demonstration with automatic model fallback.

Modes::

    basic       Minimal demo (10 steps, synthetic data)
    pecan       Real Pecan Street data demo (10 steps)
    fault       Fault-injection scenario demo (10 steps)
    compare     Compare PPO vs B1 rule-based over a full day
    full        Full day simulation with metrics

Usage::

    python scripts/demo.py basic
    python scripts/demo.py pecan --city bangalore
    python scripts/demo.py fault
    python scripts/demo.py compare
    python scripts/demo.py full
    python scripts/demo.py all          # run all modes sequentially
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from ai.core.cortexcore import CortexCore, PPOConfig, RuntimeWithFallback
from ai.core.normalizer import ObservationNormalizer
from ai.env.gym_env import EnvConfig, VayuGridEnv


def _make_env(
    scenario: str,
    use_pecan: bool = False,
    city: str = "bangalore",
) -> VayuGridEnv:
    return VayuGridEnv(EnvConfig(
        scenario_path=scenario,
        seed=42,
        use_pecan=use_pecan,
        city=city,
    ))


def _load_model(checkpoint: str) -> RuntimeWithFallback:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalizer = ObservationNormalizer((12,))
    agent = CortexCore(cfg=PPOConfig(), normalizer=normalizer, device=device)
    agent.load(checkpoint)
    return RuntimeWithFallback(agent=agent)


# ── Scenario runners ──────────────────────────────────────────────

def run_basic(n_steps: int = 10) -> None:
    print("=" * 56)
    print("  Demo: Basic (synthetic data, 10 homes)")
    print("=" * 56)
    env = _make_env("scenarios/phase1_debug.json")
    runtime = _load_model("outputs/checkpoints/cortexcore_best.pt")
    obs, _ = env.reset()

    print(f"{'Step':>5} | {'Batt':>7} {'EV':>7} {'Bid':>7} {'Ask':>7} "
          f"{'Grid':>7} | {'Reward':>8} | Model")
    print("-" * 70)
    total = 0.0
    for i in range(n_steps):
        act = runtime.step(obs)
        next_obs, rew, term, trunc, _ = env.step(act)
        total += rew
        model_tag = "[F]" if runtime.is_using_fallback else "[P]"
        print(f"{i:>5} | {act[0]:+7.3f} {act[1]:+7.3f} {act[2]:+7.3f} "
              f"{act[3]:+7.3f} {act[4]:+7.3f} | {rew:+8.2f} | {model_tag}")
        obs = next_obs
        if term or trunc:
            break
    print("-" * 70)
    print(f"Total reward: {total:.2f}  |  Fallback active: {runtime.is_using_fallback}")


def run_pecan(city: str, n_steps: int = 10) -> None:
    print("=" * 56)
    print(f"  Demo: Pecan Street ({city}, 10 homes)")
    print("=" * 56)
    env = _make_env("scenarios/phase1_debug.json", use_pecan=True, city=city)
    runtime = _load_model("outputs/checkpoints/cortexcore_best.pt")
    obs, _ = env.reset()

    print(f"{'Step':>5} | {'Batt':>7} {'EV':>7} {'Bid':>7} {'Ask':>7} "
          f"{'Grid':>7} | {'Reward':>8} | {'Load':>7} {'PV':>7} | Model")
    print("-" * 85)
    total = 0.0
    for i in range(n_steps):
        act = runtime.step(obs)
        next_obs, rew, term, trunc, _ = env.step(act)
        total += rew
        model_tag = "[F]" if runtime.is_using_fallback else "[P]"
        load_kw = obs[2]
        pv_kw = obs[1]
        print(f"{i:>5} | {act[0]:+7.3f} {act[1]:+7.3f} {act[2]:+7.3f} "
              f"{act[3]:+7.3f} {act[4]:+7.3f} | {rew:+8.2f} | "
              f"{load_kw:>7.2f} {pv_kw:>7.2f} | {model_tag}")
        obs = next_obs
        if term or trunc:
            break
    print("-" * 85)
    print(f"Total reward: {total:.2f}  |  Fallback active: {runtime.is_using_fallback}")


def run_fault(n_steps: int = 10) -> None:
    print("=" * 56)
    print("  Demo: Fault scenario")
    print("=" * 56)
    env = _make_env("scenarios/phase1_demo.json")
    runtime = _load_model("outputs/checkpoints/cortexcore_best.pt")
    obs, _ = env.reset()

    print(f"{'Step':>5} | {'Batt':>7} {'EV':>7} {'Bid':>7} {'Ask':>7} "
          f"{'Grid':>7} | {'Reward':>8} | Model")
    print("-" * 70)
    total = 0.0
    for i in range(n_steps):
        act = runtime.step(obs)
        next_obs, rew, term, trunc, _ = env.step(act)
        total += rew
        model_tag = "[F]" if runtime.is_using_fallback else "[P]"
        print(f"{i:>5} | {act[0]:+7.3f} {act[1]:+7.3f} {act[2]:+7.3f} "
              f"{act[3]:+7.3f} {act[4]:+7.3f} | {rew:+8.2f} | {model_tag}")
        obs = next_obs
        if term or trunc:
            break
    print("-" * 70)
    print(f"Total reward: {total:.2f}  |  Fallback active: {runtime.is_using_fallback}")


def run_compare() -> None:
    print("=" * 56)
    print("  Demo: PPO vs B1 (full day comparison)")
    print("=" * 56)
    env = _make_env("scenarios/phase1_debug.json")
    runtime = _load_model("outputs/checkpoints/cortexcore_best.pt")

    # Run PPO
    obs, _ = env.reset()
    ppo_rewards: list[float] = []
    term = trunc = False
    steps = 0
    while not (term or trunc) and steps < 1440:
        act = runtime.step(obs)
        obs, rew, term, trunc, _ = env.step(act)
        ppo_rewards.append(rew)
        steps += 1
    ppo_total = sum(ppo_rewards)
    print(f"  PPO ({'fallback' if runtime.is_using_fallback else 'model'}): "
          f"{steps} steps, total={ppo_total:.2f}")

    # Run B1 fallback explicitly
    env2 = _make_env("scenarios/phase1_debug.json")
    obs, _ = env2.reset()
    b1_rewards: list[float] = []
    term = trunc = False
    steps = 0
    while not (term or trunc) and steps < 1440:
        soc_norm = obs[0]
        t_sin, t_cos = obs[6], obs[7]
        hour = (np.degrees(np.arctan2(t_sin, t_cos)) % 360) / 15.0
        batt = 0.0
        if soc_norm > 0.2:
            if 17 <= hour < 21:
                batt = -0.6
            elif hour >= 22 or hour < 6:
                batt = 0.4
        act = np.array([batt, 0.5, 0.5, 0.5, 0.0], dtype=np.float32)
        obs, rew, term, trunc, _ = env2.step(act)
        b1_rewards.append(rew)
        steps += 1
    b1_total = sum(b1_rewards)
    print(f"  B1 (rule-based):           {steps} steps, total={b1_total:.2f}")
    delta = ppo_total - b1_total
    better = "PPO" if delta > 0 else "B1" if delta < 0 else "tie"
    print(f"  Difference: {delta:+.2f}  ({better} wins)")


def run_full() -> None:
    print("=" * 56)
    print("  Demo: Full day simulation with metrics")
    print("=" * 56)
    env = _make_env("scenarios/phase1_debug.json")
    runtime = _load_model("outputs/checkpoints/cortexcore_best.pt")
    obs, _ = env.reset()

    total_reward = 0.0
    step_count = 0
    t0 = time.perf_counter()
    term = trunc = False
    while not (term or trunc) and step_count < 1440:
        act = runtime.step(obs)
        obs, rew, term, trunc, _ = env.step(act)
        total_reward += rew
        step_count += 1
    elapsed = time.perf_counter() - t0

    print(f"  Steps simulated: {step_count}")
    print(f"  Wall time:       {elapsed:.1f}s")
    print(f"  Total reward:    {total_reward:.2f}")
    print(f"  Avg reward/step: {total_reward / max(step_count, 1):.4f}")
    print(f"  Model:           {'fallback' if runtime.is_using_fallback else 'trained PPO'}")
    print(f"  Steps/s:         {step_count / max(elapsed, 1e-6):.0f}")


# ── CLI ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VayuGrid Demo")
    parser.add_argument(
        "mode",
        nargs="?",
        default="basic",
        choices=["basic", "pecan", "fault", "compare", "full", "all"],
        help="Demo scenario to run",
    )
    parser.add_argument("--city", default="bangalore",
                        choices=["bangalore", "chennai", "delhi", "hyderabad", "kochi"])
    parser.add_argument("--steps", type=int, default=10,
                        help="Steps per scenario (basic/pecan/fault only)")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    modes = ["basic", "pecan", "fault", "compare", "full"] if args.mode == "all" else [args.mode]

    for mode in modes:
        t0 = time.perf_counter()
        if mode == "basic":
            run_basic(args.steps)
        elif mode == "pecan":
            run_pecan(args.city, args.steps)
        elif mode == "fault":
            run_fault(args.steps)
        elif mode == "compare":
            run_compare()
        elif mode == "full":
            run_full()
        elapsed = time.perf_counter() - t0
        print(f"  [{mode}] completed in {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
