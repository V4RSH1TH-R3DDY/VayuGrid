from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


@dataclass
class VayuGNNForecast:
    overload_probability: np.ndarray
    voltage_forecast: np.ndarray
    neighborhood_risk: float
    duck_curve_forecast: np.ndarray


class VayuGNN(nn.Module):
    """VayuGNN: Heterogeneous Graph Transformer + Temporal Self-Attention.

    Takes 12 snapshots of node features and predicts:
      - Per-minute overload probability (next 30 min)
      - Voltage forecast
      - Neighborhood risk score
      - 24-hour duck curve load forecast

    This is a scaffold — the full HGT + temporal attention layers
    will be implemented once Phase 3 training begins.
    """

    def __init__(
        self,
        node_feat_dim: int = 6,
        hidden_dim: int = 128,
        num_snapshots: int = 12,
        forecast_steps: int = 30,
    ) -> None:
        super().__init__()
        if nn is None:
            raise ImportError("torch is required for VayuGNN")

        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.temporal_encoder = nn.Sequential(
            nn.Linear(hidden_dim * num_snapshots, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.overload_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, forecast_steps),
            nn.Sigmoid(),
        )
        self.voltage_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, forecast_steps),
        )
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.duck_curve_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 96),
        )

    def forward(
        self, node_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_snapshots, num_nodes, feat_dim = node_features.shape
        x = node_features.view(batch_size * num_snapshots, num_nodes, feat_dim)
        h = self.node_encoder(x)
        h = h.view(batch_size, num_snapshots * h.shape[-1])
        h = self.temporal_encoder(h)

        overload = self.overload_head(h)
        voltage = self.voltage_head(h)
        risk = self.risk_head(h).squeeze(-1)
        duck = self.duck_curve_head(h)

        return overload, voltage, risk, duck

    @torch.no_grad()
    def predict(self, node_features: np.ndarray) -> VayuGNNForecast:
        self.eval()
        device = next(self.parameters()).device
        x = torch.from_numpy(node_features).float().unsqueeze(0).to(device)

        overload, voltage, risk, duck = self.forward(x)

        return VayuGNNForecast(
            overload_probability=overload.cpu().numpy().squeeze(0),
            voltage_forecast=voltage.cpu().numpy().squeeze(0),
            neighborhood_risk=float(risk.cpu().numpy().squeeze()),
            duck_curve_forecast=duck.cpu().numpy().squeeze(0),
        )
