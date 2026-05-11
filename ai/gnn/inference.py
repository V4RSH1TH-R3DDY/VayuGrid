"""
VayuGNN Inference Server.

Wraps VayuGNNPipeline for use in the API WebSocket stream.
Loads a trained model checkpoint on startup and provides
`predict()` that maps transformer readings to overload probabilities.

Falls back to a sigmoid approximation when no checkpoint is available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ai.gnn.vayu_gnn import (
    EDGE_FEAT_DIM,
    N_SNAPSHOTS,
    NODE_FEAT_DIM,
    GraphSnapshot,
    VayuGNN,
    VayuGNNPipeline,
)


class VayuGNNInferenceServer:
    """
    Lightweight GNN inference wrapper for the API.

    Usage:
        server = VayuGNNInferenceServer()
        server.load("outputs/checkpoints/vayu_gnn_best.pt")
        prob = server.predict_overload(transformer_loading_pu=0.95)
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cpu")
        self.model: Optional[VayuGNN] = None
        self.pipeline: Optional[VayuGNNPipeline] = None
        self._loaded = False

    def load(self, checkpoint_path: str) -> bool:
        path = Path(checkpoint_path)
        if not path.exists():
            print(
                f"[VayuGNNInferenceServer] Checkpoint not found:"
                f" {checkpoint_path} (using fallback)"
            )
            self._loaded = False
            return False

        self.model = VayuGNN().to(self.device)
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

        self.pipeline = VayuGNNPipeline(model=self.model, device=self.device)
        self._loaded = True
        print(f"[VayuGNNInferenceServer] Loaded checkpoint: {checkpoint_path}")
        return True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict_overload(
        self,
        transformer_loading_pu: float,
        max_branch_loading_pu: float = 0.0,
        temperature_c: float = 25.0,
    ) -> float:
        """
        Returns P(overload) in [0, 1] for a given transformer loading.

        Uses the GNN when a model is loaded; otherwise falls back to
        a sigmoid placeholder.
        """
        if not self._loaded:
            return self._sigmoid_fallback(transformer_loading_pu)

        assert self.model is not None, "Model not loaded"
        dummy_snapshots = self._build_dummy_snapshots(
            loading_pu=transformer_loading_pu,
            temperature_c=temperature_c,
        )
        with torch.no_grad():
            pred = self.model(dummy_snapshots)
        return float(pred.overload_prob[0, :5].mean().item())

    @staticmethod
    def _sigmoid_fallback(loading: float) -> float:
        return 1.0 / (1.0 + np.exp(-12.0 * (loading - 1.0)))

    def _build_dummy_snapshots(
        self,
        loading_pu: float = 0.0,
        temperature_c: float = 25.0,
    ) -> list[GraphSnapshot]:
        """Build N_SNAPSHOTS dummy snapshots seeded with the current reading."""
        device = self.device
        n_nodes = 20
        n_xfmr = 3
        n_edges = 38

        snaps = []
        for _ in range(N_SNAPSHOTS):
            node_feat = torch.randn(n_nodes, NODE_FEAT_DIM, device=device)
            xfmr_feat = torch.tensor(
                [[loading_pu, temperature_c, 1.0]] * n_xfmr,
                dtype=torch.float32, device=device,
            )
            edge_idx = torch.randint(0, n_nodes, (2, n_edges), device=device)
            edge_feat = torch.randn(n_edges, EDGE_FEAT_DIM, device=device)
            node_to_xfmr = torch.randint(0, n_xfmr, (n_nodes,), device=device)
            snaps.append(GraphSnapshot(
                node_features=node_feat,
                xfmr_features=xfmr_feat,
                edge_index=edge_idx,
                edge_features=edge_feat,
                node_to_xfmr=node_to_xfmr,
            ))
        return snaps
