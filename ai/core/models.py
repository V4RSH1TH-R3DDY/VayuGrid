from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

OBS_DIM = 16
ACT_DIM = 5


class Actor(nn.Module):
    """Small actor network for edge deployment.

    Two-layer MLP with tanh hidden, outputs mean + log_std for each
    continuous action dimension.
    """

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM, hidden: int = 64) -> None:
        super().__init__()
        if nn is None:
            raise ImportError("torch is required for Actor")

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(obs)
        mean = self.mean_head(h)
        return mean, self.log_std.expand_as(mean)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mean, log_std = self.forward(obs)
        if deterministic:
            return torch.tanh(mean)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        raw = dist.rsample()
        return torch.tanh(raw)

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        raw = torch.atanh(action.clamp(-0.999, 0.999))
        return dist.log_prob(raw).sum(dim=-1)


class Critic(nn.Module):
    """Larger critic network used only during training."""

    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 256) -> None:
        super().__init__()
        if nn is None:
            raise ImportError("torch is required for Critic")

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


@dataclass
class CortexCorePolicy:
    """Combined policy holding actor + critic + observation normalizer."""
    actor: Actor
    critic: Critic
    obs_mean: np.ndarray = field(default_factory=lambda: np.zeros(OBS_DIM, dtype=np.float32))
    obs_std: np.ndarray = field(default_factory=lambda: np.ones(OBS_DIM, dtype=np.float32))

    def to(self, device: torch.device) -> None:
        self.actor.to(device)
        self.critic.to(device)

    def eval(self) -> None:
        self.actor.eval()
        self.critic.eval()

    def train(self) -> None:
        self.actor.train()
        self.critic.train()

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "obs_mean": self.obs_mean,
            "obs_std": self.obs_std,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.obs_mean = state.get("obs_mean", self.obs_mean)
        self.obs_std = state.get("obs_std", self.obs_std)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM) -> CortexCorePolicy:
        inst = cls(Actor(obs_dim, act_dim), Critic(obs_dim))
        inst.load_state_dict(torch.load(path, map_location="cpu"))
        return inst
