from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    import torch.optim as optim

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]

from ai.core.models import Actor, CortexCorePolicy, Critic
from ai.core.normalizer import ObservationNormalizer
from ai.env.gym_env import VayuGridEnv

logger = logging.getLogger(__name__)


@dataclass
class PPOTrainingConfig:
    total_timesteps: int = 200_000
    steps_per_epoch: int = 2048
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    learning_rate: float = 3e-4
    target_kl: float = 0.02
    max_grad_norm: float = 0.5
    n_epochs: int = 10
    device: str = "cpu"
    log_interval: int = 10
    save_interval: int = 50


@dataclass
class RolloutBuffer:
    obs: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)

    def clear(self) -> None:
        self.obs.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()


class PPOTrainer:
    """PPO training loop for CortexCore agent."""

    def __init__(self, env: VayuGridEnv, config: PPOTrainingConfig | None = None) -> None:
        if torch is None:
            raise ImportError("torch is required for PPOTrainer")

        self.env = env
        self.cfg = config or PPOTrainingConfig()
        self.device = torch.device(self.cfg.device)

        assert env.observation_space is not None
        assert env.action_space is not None
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        self.actor = Actor(obs_dim, act_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)
        self.normalizer = ObservationNormalizer((obs_dim,))
        self.policy = CortexCorePolicy(self.actor, self.critic)
        self.policy.to(self.device)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.cfg.learning_rate)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.cfg.learning_rate)

        self.buffer = RolloutBuffer()
        self.step_count = 0
        self.epoch_count = 0

    def collect_rollout(self) -> RolloutBuffer:
        self.buffer.clear()
        obs, _ = self.env.reset()
        obs = self.normalizer.normalize(obs)

        for _ in range(self.cfg.steps_per_epoch):
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                action_t = self.actor.get_action(obs_t)
                value = self.critic(obs_t)
                log_prob = self.actor.log_prob(obs_t, action_t)

            action = action_t.cpu().numpy().squeeze(0)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)

            self.buffer.obs.append(obs.copy())
            self.buffer.actions.append(action.copy())
            self.buffer.rewards.append(reward)
            self.buffer.values.append(float(value.cpu()))
            self.buffer.log_probs.append(float(log_prob.cpu()))
            self.buffer.dones.append(terminated or truncated)

            obs = self.normalizer.normalize(next_obs)
            self.step_count += 1

            if terminated or truncated:
                obs, _ = self.env.reset()
                obs = self.normalizer.normalize(obs)

        return self.buffer

    def _compute_gae(
        self, rewards: list[float], values: list[float], dones: list[bool],
    ) -> tuple[np.ndarray, np.ndarray]:
        returns = np.zeros(len(rewards), dtype=np.float32)
        advantages = np.zeros(len(rewards), dtype=np.float32)
        gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0 if dones[t] else values[t]
            else:
                next_value = values[t + 1] if not dones[t] else 0.0

            delta = rewards[t] + self.cfg.gamma * next_value - values[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * (1 - float(dones[t])) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        return returns, advantages

    def train_epoch(self) -> dict[str, float]:
        self.actor.train()
        self.critic.train()

        buffer = self.collect_rollout()

        obs = np.array(buffer.obs, dtype=np.float32)
        actions = np.array(buffer.actions, dtype=np.float32)
        old_log_probs = np.array(buffer.log_probs, dtype=np.float32)
        returns, advantages = self._compute_gae(buffer.rewards, buffer.values, buffer.dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t = torch.from_numpy(obs).to(self.device)
        actions_t = torch.from_numpy(actions).to(self.device)
        returns_t = torch.from_numpy(returns).to(self.device)
        advantages_t = torch.from_numpy(advantages).to(self.device)
        old_log_probs_t = torch.from_numpy(old_log_probs).to(self.device)

        n = len(buffer.obs)
        idx = np.arange(n)
        policy_losses = []
        value_losses = []

        for _ in range(self.cfg.n_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.cfg.batch_size):
                end = start + self.cfg.batch_size
                batch_idx = idx[start:end]

                batch_obs = obs_t[batch_idx]
                batch_actions = actions_t[batch_idx]
                batch_returns = returns_t[batch_idx]
                batch_adv = advantages_t[batch_idx]
                batch_old_log = old_log_probs_t[batch_idx]

                log_probs = self.actor.log_prob(batch_obs, batch_actions)
                ratio = (log_probs - batch_old_log).exp()

                clipped = torch.clamp(
                    ratio, 1.0 - self.cfg.clip_epsilon, 1.0 + self.cfg.clip_epsilon,
                )
                clip_adv = clipped * batch_adv
                policy_loss = -torch.min(ratio * batch_adv, clip_adv).mean()

                values = self.critic(batch_obs)
                value_loss = nn.functional.mse_loss(values, batch_returns)

                self.actor_optim.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                self.critic_optim.step()

                policy_losses.append(float(policy_loss))
                value_losses.append(float(value_loss))

                with torch.no_grad():
                    kl = (old_log_probs_t[batch_idx] - log_probs).mean().item()
                    if kl > self.cfg.target_kl:
                        return {
                            "policy_loss": float(np.mean(policy_losses)),
                            "value_loss": float(np.mean(value_losses)),
                            "kl_divergence": kl,
                            "early_stopped": True,
                        }

        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "kl_divergence": 0.0,
            "early_stopped": False,
        }

    def train(self, callbacks: list[Any] | None = None) -> CortexCorePolicy:
        total_epochs = self.cfg.total_timesteps // self.cfg.steps_per_epoch

        for epoch in range(1, total_epochs + 1):
            self.epoch_count = epoch
            metrics = self.train_epoch()

            if epoch % self.cfg.log_interval == 0:
                logger.info(
                    "Epoch %3d/%d  policy_loss=%.4f  value_loss=%.4f  kl=%.4f  steps=%d",
                    epoch, total_epochs,
                    metrics["policy_loss"], metrics["value_loss"],
                    metrics["kl_divergence"], self.step_count,
                )

            if callbacks:
                for cb in callbacks:
                    cb(epoch, metrics, self)

        return self.policy
