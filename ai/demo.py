from __future__ import annotations

from dataclasses import dataclass

from ai.core.cortexcore import PPOConfig, RewardConfig
from ai.env.gym_env import EnvConfig
from ai.gnn.vayu_gnn import GNNTrainConfig
from ai.training.config import TrainingConfig


@dataclass
class DemoConfig:
    """Demo-mode overrides for all VayuGrid components.

    Reduces data intake, model size, and training duration
    so the system can be demonstrated in minutes instead of hours.
    """

    scenario_path: str = "scenarios/phase1_demo.json"
    num_homes: int = 10
    duration_minutes: int = 120

    gnn_snapshots: int = 3
    gnn_overload_steps: int = 5
    gnn_duck_steps: int = 10
    gnn_hidden_dim: int = 32
    gnn_gru_hidden: int = 64
    gnn_layers: int = 1
    gnn_epochs: int = 2
    gnn_num_episodes: int = 1

    ppo_rollout_steps: int = 128
    ppo_ppo_epochs: int = 3
    ppo_minibatch_size: int = 16
    ppo_total_timesteps: int = 5_000
    ppo_n_envs: int = 1
    ppo_episode_minutes: int = 60


def make_demo_env_config(demo: DemoConfig | None = None) -> EnvConfig:
    """Create an EnvConfig reduced for demo."""
    d = demo or DemoConfig()
    return EnvConfig(
        scenario_path=d.scenario_path,
        episode_minutes=d.ppo_episode_minutes,
        seed=42,
        use_pecan=False,
        city="bangalore",
    )


def make_demo_ppo_config(demo: DemoConfig | None = None) -> PPOConfig:
    """Create a PPOConfig reduced for demo."""
    d = demo or DemoConfig()
    return PPOConfig(
        rollout_steps=d.ppo_rollout_steps,
        ppo_epochs=d.ppo_ppo_epochs,
        minibatch_size=d.ppo_minibatch_size,
    )


def make_demo_training_config(demo: DemoConfig | None = None) -> TrainingConfig:
    """Create a TrainingConfig reduced for demo."""
    d = demo or DemoConfig()
    return TrainingConfig(
        total_timesteps=d.ppo_total_timesteps,
        steps_per_epoch=d.ppo_rollout_steps,
        n_envs=d.ppo_n_envs,
        scenario_path=d.scenario_path,
        curriculum_stages=[
            {"name": "demo_stage", "num_homes": 3,
             "p2p_enabled": False, "faults": False, "epochs": 1},
        ],
    )


def make_demo_gnn_config(demo: DemoConfig | None = None) -> GNNTrainConfig:
    """Create a GNNTrainConfig reduced for demo."""
    d = demo or DemoConfig()
    return GNNTrainConfig(
        lr=1e-3,
        epochs=d.gnn_epochs,
        patience=2,
        checkpoint_path="outputs/checkpoints/vayu_gnn_demo.pt",
    )


def make_demo_reward_config() -> RewardConfig:
    """Standard RewardConfig (unchanged for demo)."""
    return RewardConfig()
