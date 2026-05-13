#!/usr/bin/env python3
"""
PPO Training Script for CortexCore Agent.

Trains a CortexCore PPO agent using the VayuGridEnv Gymnasium environment.
Saves actor + critic checkpoints and exports INT8 ONNX for Pi deployment.

Usage:
    python scripts/train_cortexcore.py \
        --scenario scenarios/phase1_default.json \
        --total-timesteps 1_000_000 \
        --checkpoint-dir outputs/checkpoints
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from ai.core.cortexcore import (
    CortexCore,
    CortexCoreExporter,
    PPOConfig,
    RewardConfig,
)
from ai.core.normalizer import ObservationNormalizer
from ai.demo import make_demo_env_config, make_demo_ppo_config, make_demo_reward_config
from ai.env.gym_env import EnvConfig, VayuGridEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CortexCore PPO agent")
    parser.add_argument("--scenario", default="scenarios/phase1_default.json")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    parser.add_argument("--log-dir", default="outputs/logs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--demo", action="store_true",
                        help="Demo mode: tiny scenario, small model, fast iteration")
    parser.add_argument("--use-pecan", action="store_true",
                        help="Use Pecan Street real load profiles (overrides scenario config)")
    parser.add_argument("--city", default="bangalore",
                        choices=["bangalore", "chennai", "delhi", "hyderabad", "kochi"],
                        help="City for Pecan/NSRDB data")
    return parser.parse_args()


def _make_env(args: argparse.Namespace) -> VayuGridEnv:
    if args.demo:
        return VayuGridEnv(make_demo_env_config())
    return VayuGridEnv(EnvConfig(
        scenario_path=args.scenario,
        seed=args.seed,
        use_pecan=args.use_pecan,
        city=args.city,
    ))


def train(args: argparse.Namespace) -> CortexCore:
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    print(f"[train_cortexcore] Device: {device}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.use_pecan and not args.demo:
        args.scenario = "scenarios/phase1_debug.json"
        if args.total_timesteps >= 500_000:
            args.total_timesteps = 10_000
        print(f"[train_cortexcore] Real-data mode: scenario={args.scenario},"
              f" timesteps={args.total_timesteps}")

    normalizer = ObservationNormalizer((12,))
    if args.demo:
        ppo_cfg = make_demo_ppo_config()
    elif args.use_pecan:
        ppo_cfg = PPOConfig(reward_scale=0.01, lr_actor=3e-5, lr_critic=3e-4)
    else:
        ppo_cfg = PPOConfig()
    rew_cfg = make_demo_reward_config() if args.demo else RewardConfig()
    agent = CortexCore(
        cfg=ppo_cfg,
        reward=rew_cfg,
        normalizer=normalizer,
        device=device,
    )

    env = _make_env(args)
    obs, _ = env.reset()

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    ep_rew = 0.0
    ep_len = 0
    ep_count = 0
    best_mean_reward = -float("inf")
    timesteps_at_last_checkpoint = 0

    t0 = time.perf_counter()
    for step in range(1, args.total_timesteps + 1):
        # Select action (returns env_action + raw_action for correct PPO ratio)
        env_action_np, raw_action_np, log_prob, value = agent.select_action(obs)

        # Step environment with clipped action
        next_obs, env_reward, terminated, truncated, info = env.step(env_action_np)
        done = terminated or truncated

        # Use CortexCore's reward computer for proper reward shaping
        soc_norm = obs[0]
        rew, _ = agent.reward_computer.compute(
            p2p_revenue_dollar=info.get("p2p_revenue", 0.0),
            grid_import_cost_dollar=info.get("grid_cost", 0.0),
            soc_norm=soc_norm,
            ev_deadline_missed=False,
            gnn_signal_followed=False,
        )

        # Store transition with raw action (pre-clipping) for correct ratio
        agent.push_transition(obs, raw_action_np, rew, value, log_prob, done)

        ep_rew += rew
        ep_len += 1
        obs = next_obs

        # Update running normalizer
        normalizer.update(obs)

        # Episode tracking
        if done:
            episode_rewards.append(ep_rew)
            episode_lengths.append(ep_len)
            ep_rew = 0.0
            ep_len = 0
            ep_count += 1
            obs, _ = env.reset()

            window = episode_rewards[-20:]
            mean_rew = float(np.mean(window)) if window else 0.0
            if ep_count % 10 == 0:
                elapsed = time.perf_counter() - t0
                sps = step / elapsed
                print(
                    f"  Ep {ep_count:>4d} | step {step:>7d} | "
                    f"mean_reward={mean_rew:+7.2f} | {sps:5.0f} steps/s"
                )

        # PPO update when buffer is full
        if agent.buffer.full:
            metrics = agent.update(obs, done)
            elapsed = time.perf_counter() - t0
            sps = step / elapsed
            mean_rew = float(np.mean(episode_rewards[-20:])) if episode_rewards else 0.0
            es = " [ES]" if metrics.get("early_stopped") else ""
            print(
                f"  Update at step {step:>7d} | "
                f"policy_loss={metrics['policy_loss']:.4f} "
                f"value_loss={metrics['value_loss']:.4f} "
                f"entropy={metrics['entropy']:.4f} "
                f"approx_kl={metrics['approx_kl']:.4f} "
                f"mean_reward={mean_rew:+7.2f} | {sps:5.0f} steps/s{es}"
            )

            # Checkpoint if best so far
            if mean_rew > best_mean_reward:
                best_mean_reward = mean_rew
                ckpt_path = str(checkpoint_dir / "cortexcore_best.pt")
                agent.save(ckpt_path)
                timesteps_at_last_checkpoint = step

        if step % 5_000 == 0:
            ckpt_path = str(checkpoint_dir / f"cortexcore_step_{step}.pt")
            agent.save(ckpt_path)

    # Final save
    final_path = str(checkpoint_dir / "cortexcore_final.pt")
    agent.save(final_path)

    print(f"\nTraining complete: {args.total_timesteps} steps, {ep_count} episodes")
    print(f"Best mean reward: {best_mean_reward:.2f} (at step {timesteps_at_last_checkpoint})")

    # Export to ONNX
    print("\nExporting actor to INT8 ONNX...")
    exporter = CortexCoreExporter(agent.actor)
    onnx_path = exporter.full_pipeline(
        fp32_path=str(checkpoint_dir / "cortexcore_actor_fp32.onnx"),
        int8_path=str(checkpoint_dir / "cortexcore_actor_int8.onnx"),
    )
    print(f"Exported: {onnx_path}")

    return agent


if __name__ == "__main__":
    train(parse_args())
