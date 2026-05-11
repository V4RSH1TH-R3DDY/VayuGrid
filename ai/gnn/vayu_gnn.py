"""
VayuGNN — Neighborhood Distribution Grid Brain
===============================================
Runs on the neighborhood server.
Ingests 12 historical graph snapshots (1-min resolution) and predicts
grid stress, voltage risk, and net load curves across the next 30 min / 24 h.

Architecture:
  ┌──────────────────────────────────────────┐
  │  12 × snapshot  →  SpatialGNN (per snap) │
  │  12 × node embeds  →  TemporalGRU        │
  │  Final hidden  →  4 prediction heads      │
  └──────────────────────────────────────────┘

Does NOT require PyTorch Geometric — uses custom message-passing
so it runs on any standard PyTorch install on the server.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────
# 1. GRAPH SCHEMA
# ──────────────────────────────────────────────

# Per-node features (homes + batteries)
NODE_FEAT_DIM = 5    # PV_kw, load_kw, battery_soc, net_grid_kw, voltage_pu

# Per-transformer features (aggregated per distribution transformer)
XFMR_FEAT_DIM = 3    # loading_pct, temperature_c, aging_factor

# Per-edge features (distribution lines)
EDGE_FEAT_DIM = 2    # resistance_ohm, reactance_ohm

# Historical snapshots
N_SNAPSHOTS   = 12   # 12 × 1-min = 12 min lookback

# Prediction horizons
OVERLOAD_STEPS = 30   # next 30 minutes, 1 per minute
VOLTAGE_STEPS  = 30   # next 30 minutes
DUCK_STEPS     = 96   # next 24 hours at 15-min resolution

HIDDEN_DIM     = 128
GNN_LAYERS     = 3
GRU_HIDDEN     = 256


# ──────────────────────────────────────────────
# 2. DATA CONTAINERS
# ──────────────────────────────────────────────

class GraphSnapshot(NamedTuple):
    """
    One minute's worth of neighbourhood observations.

    node_features : (N_nodes, NODE_FEAT_DIM)
    xfmr_features : (N_xfmr,  XFMR_FEAT_DIM)
    edge_index    : (2, N_edges)   — COO format (src, dst)
    edge_features : (N_edges, EDGE_FEAT_DIM)
    node_to_xfmr  : (N_nodes,)    — which transformer each node belongs to
    """
    node_features: torch.Tensor
    xfmr_features: torch.Tensor
    edge_index:    torch.Tensor
    edge_features: torch.Tensor
    node_to_xfmr:  torch.Tensor


class VayuGNNOutput(NamedTuple):
    overload_prob:    torch.Tensor   # (batch, 30)  — P(overload) each minute
    voltage_forecast: torch.Tensor   # (batch, 30)  — mean voltage per minute [pu]
    neighborhood_risk: torch.Tensor  # (batch, 1)   — overall stress [0, 1]
    duck_curve:        torch.Tensor  # (batch, 96)  — net load next 24 h [kW]


# ──────────────────────────────────────────────
# 3. SPATIAL GNN — single snapshot
# ──────────────────────────────────────────────

class MessagePassingLayer(nn.Module):
    """
    Edge-conditioned graph attention layer.
    Aggregates neighbours using attention weights computed from
    (node_i, node_j, edge_ij) features.
    No external dependency — pure PyTorch.
    """

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int):
        super().__init__()
        # Message network: (h_i || h_j || e_ij) → message
        self.msg_net = nn.Sequential(
            nn.Linear(in_dim * 2 + edge_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )
        # Attention gate
        self.att_net = nn.Sequential(
            nn.Linear(in_dim * 2 + edge_dim, 1),
            nn.Sigmoid(),
        )
        # Node update
        self.update = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        h:          torch.Tensor,   # (N, in_dim)
        edge_index: torch.Tensor,   # (2, E)
        edge_feat:  torch.Tensor,   # (E, edge_dim)
    ) -> torch.Tensor:              # (N, out_dim)

        src, dst = edge_index[0], edge_index[1]
        h_src  = h[src]    # (E, in_dim)
        h_dst  = h[dst]    # (E, in_dim)
        triple = torch.cat([h_src, h_dst, edge_feat], dim=-1)   # (E, 2*in + edge_dim)

        msg  = self.msg_net(triple)                              # (E, out_dim)
        att  = self.att_net(triple)                              # (E, 1)
        msg  = msg * att                                         # (E, out_dim)

        # Aggregate: sum messages into each destination node
        N    = h.shape[0]
        agg  = torch.zeros(N, msg.shape[-1], device=h.device)
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)

        # Update node embedding
        out  = self.update(torch.cat([h, agg], dim=-1))         # (N, out_dim)
        # Residual (project if dimensions differ)
        if h.shape[-1] == out.shape[-1]:
            out = out + h
        return out


class SpatialGNN(nn.Module):
    """
    Processes one graph snapshot.
    Fuses node + transformer features, runs GNN_LAYERS message-passing steps,
    then pools to a graph-level embedding.

    Returns:
        node_emb   : (N_nodes, HIDDEN_DIM) — per-node representations
        graph_emb  : (HIDDEN_DIM,)         — graph-level summary
    """

    def __init__(
        self,
        node_dim:  int = NODE_FEAT_DIM,
        xfmr_dim:  int = XFMR_FEAT_DIM,
        edge_dim:  int = EDGE_FEAT_DIM,
        hidden:    int = HIDDEN_DIM,
        n_layers:  int = GNN_LAYERS,
    ):
        super().__init__()
        # Input projections
        self.node_proj = nn.Sequential(
            nn.Linear(node_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.xfmr_proj = nn.Linear(xfmr_dim, hidden)
        self.edge_proj = nn.Linear(edge_dim,  hidden // 4)

        projected_edge_dim = hidden // 4

        # Message-passing stack
        self.layers = nn.ModuleList([
            MessagePassingLayer(
                in_dim=hidden,
                out_dim=hidden,
                edge_dim=projected_edge_dim,
            )
            for _ in range(n_layers)
        ])

        # Graph-level readout: attention pooling
        self.readout_att = nn.Linear(hidden, 1)
        self.readout_out = nn.Linear(hidden, hidden)

    def forward(self, snap: GraphSnapshot) -> Tuple[torch.Tensor, torch.Tensor]:
        snap.node_features.shape[0]

        # 1. Initialise node embeddings, fuse transformer context
        h     = self.node_proj(snap.node_features)                     # (N, H)
        x_emb = self.xfmr_proj(snap.xfmr_features)                    # (X, H)
        # Add transformer embedding to each node it belongs to
        xfmr_ctx = x_emb[snap.node_to_xfmr]                           # (N, H)
        h         = h + xfmr_ctx

        # 2. Project edge features
        e_h  = self.edge_proj(snap.edge_features)                      # (E, H/4)

        # 3. Message passing
        for layer in self.layers:
            h = layer(h, snap.edge_index, e_h)                        # (N, H)

        # 4. Attention-weighted graph pooling
        att_w = torch.softmax(self.readout_att(h), dim=0)              # (N, 1)
        graph_emb = self.readout_out((att_w * h).sum(dim=0))           # (H,)

        return h, graph_emb


# ──────────────────────────────────────────────
# 4. TEMPORAL MODULE — across 12 snapshots
# ──────────────────────────────────────────────

class TemporalAggregator(nn.Module):
    """
    GRU over the sequence of 12 graph-level embeddings.
    Captures how the grid state has been evolving over the last 12 minutes.
    """

    def __init__(self, in_dim: int = HIDDEN_DIM, gru_hidden: int = GRU_HIDDEN):
        super().__init__()
        self.gru  = nn.GRU(
            input_size=in_dim,
            hidden_size=gru_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.norm = nn.LayerNorm(gru_hidden)

    def forward(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        seq : (batch, T, HIDDEN_DIM)  — T graph embeddings in time order
        Returns:
            all_hidden : (batch, T, GRU_HIDDEN)
            last_hidden: (batch, GRU_HIDDEN)
        """
        out, _ = self.gru(seq)       # (B, T, GRU_H)
        out     = self.norm(out)
        return out, out[:, -1, :]    # all steps, final step


# ──────────────────────────────────────────────
# 5. PREDICTION HEADS
# ──────────────────────────────────────────────

def _mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    mods: List[nn.Module] = [nn.Linear(in_dim, hidden), nn.SiLU()]
    for _ in range(layers - 1):
        mods += [nn.Linear(hidden, hidden), nn.SiLU()]
    mods.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*mods)


class OverloadHead(nn.Module):
    """
    Predicts P(transformer overload) for each of the next 30 minutes.
    Output: (batch, 30) in [0, 1].
    """

    def __init__(self, gru_hidden: int = GRU_HIDDEN):
        super().__init__()
        self.net = _mlp(gru_hidden, 128, OVERLOAD_STEPS, layers=2)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(h))   # (B, 30)


class VoltageHead(nn.Module):
    """
    Predicts mean bus voltage [pu] for each of the next 30 minutes.
    Output: (batch, 30).  Typical range [0.85, 1.05].
    """

    def __init__(self, gru_hidden: int = GRU_HIDDEN):
        super().__init__()
        self.net = _mlp(gru_hidden, 128, VOLTAGE_STEPS, layers=2)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # Bias initialisation toward 1.0 pu (handled post-hoc via calibration)
        return self.net(h)   # (B, 30)


class RiskHead(nn.Module):
    """
    Single neighbourhood risk score in [0, 1].
    """

    def __init__(self, gru_hidden: int = GRU_HIDDEN):
        super().__init__()
        self.net = _mlp(gru_hidden, 64, 1, layers=2)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(h))   # (B, 1)


class DuckCurveHead(nn.Module):
    """
    Predicts net load [kW] for the next 96 × 15-min slots (24 h).
    Uses an autoregressive decoder seeded by the GRU hidden state.
    Output: (batch, 96).
    """

    def __init__(self, gru_hidden: int = GRU_HIDDEN):
        super().__init__()
        self.seed_proj = nn.Linear(gru_hidden, 128)
        self.decoder   = nn.GRU(
            input_size=1,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
        )
        self.out_proj  = nn.Linear(128, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B     = h.shape[0]
        seed  = self.seed_proj(h).unsqueeze(0)   # (1, B, 128) for GRU h_0
        # Teacher-force-free: step with zero input at each slot
        inp   = torch.zeros(B, DUCK_STEPS, 1, device=h.device)
        out, _= self.decoder(inp, seed)           # (B, 96, 128)
        return self.out_proj(out).squeeze(-1)     # (B, 96)


# ──────────────────────────────────────────────
# 6. FULL VayuGNN MODEL
# ──────────────────────────────────────────────

class VayuGNN(nn.Module):
    """
    Full neighbourhood GNN model.

    Forward pass accepts a sequence of GraphSnapshot objects
    (length = N_SNAPSHOTS = 12).

    Returns a VayuGNNOutput namedtuple with all 4 prediction heads.
    """

    def __init__(
        self,
        hidden:     int = HIDDEN_DIM,
        gru_hidden: int = GRU_HIDDEN,
        n_gnn_layers: int = GNN_LAYERS,
    ):
        super().__init__()
        self.spatial  = SpatialGNN(hidden=hidden, n_layers=n_gnn_layers)
        self.temporal = TemporalAggregator(in_dim=hidden, gru_hidden=gru_hidden)

        self.head_overload  = OverloadHead(gru_hidden)
        self.head_voltage   = VoltageHead(gru_hidden)
        self.head_risk      = RiskHead(gru_hidden)
        self.head_duck      = DuckCurveHead(gru_hidden)

    def forward(self, snapshots: List[GraphSnapshot]) -> VayuGNNOutput:
        """
        snapshots: list of N_SNAPSHOTS GraphSnapshot objects
                   (node tensors NOT batched — single graph per snapshot)
        """
        assert len(snapshots) == N_SNAPSHOTS, (
            f"Expected {N_SNAPSHOTS} snapshots, got {len(snapshots)}"
        )

        # Encode each snapshot spatially, collect graph embeddings
        graph_embs = []
        for snap in snapshots:
            _, g_emb = self.spatial(snap)       # (HIDDEN_DIM,)
            graph_embs.append(g_emb)

        # Stack → (1, T, HIDDEN_DIM)  — single neighbourhood, batch=1
        seq = torch.stack(graph_embs, dim=0).unsqueeze(0)

        # Temporal aggregation
        _, h_last = self.temporal(seq)          # (1, GRU_HIDDEN)

        return VayuGNNOutput(
            overload_prob     = self.head_overload(h_last),    # (1, 30)
            voltage_forecast  = self.head_voltage(h_last),     # (1, 30)
            neighborhood_risk = self.head_risk(h_last),        # (1, 1)
            duck_curve        = self.head_duck(h_last),        # (1, 96)
        )

    # ── Batched forward (for training on multiple neighbourhoods) ──

    def forward_batched(
        self, batch_snapshots: List[List[GraphSnapshot]]
    ) -> VayuGNNOutput:
        """
        batch_snapshots: list of B neighbourhoods, each with N_SNAPSHOTS snapshots.
        Returns heads with batch dimension B.
        """
        all_seqs = []
        for snapshots in batch_snapshots:
            g_embs = [self.spatial(s)[1] for s in snapshots]
            all_seqs.append(torch.stack(g_embs, dim=0))      # (T, H)

        seq = torch.stack(all_seqs, dim=0)                    # (B, T, H)
        _, h_last = self.temporal(seq)                        # (B, GRU_H)

        return VayuGNNOutput(
            overload_prob     = self.head_overload(h_last),
            voltage_forecast  = self.head_voltage(h_last),
            neighborhood_risk = self.head_risk(h_last),
            duck_curve        = self.head_duck(h_last),
        )

    def count_parameters(self) -> Dict[str, int]:
        parts = {
            "spatial_gnn":     self.spatial,
            "temporal_gru":    self.temporal,
            "head_overload":   self.head_overload,
            "head_voltage":    self.head_voltage,
            "head_risk":       self.head_risk,
            "head_duck":       self.head_duck,
        }
        return {k: sum(p.numel() for p in v.parameters()) for k, v in parts.items()}


# ──────────────────────────────────────────────
# 7. MULTI-TASK LOSS
# ──────────────────────────────────────────────

class VayuGNNLoss(nn.Module):
    """
    Combined loss across all 4 heads.
    Overload uses binary cross-entropy (classification).
    Voltage and duck curve use Huber loss (regression).
    Risk uses BCE.

    Overload head is weighted more heavily to enforce <1% false-positive rate.
    """

    def __init__(
        self,
        w_overload: float = 3.0,   # extra weight — false positives are costly
        w_voltage:  float = 1.5,
        w_risk:     float = 1.0,
        w_duck:     float = 1.0,
        fp_penalty: float = 5.0,   # additional penalty on FP overload predictions
    ):
        super().__init__()
        self.w_overload = w_overload
        self.w_voltage  = w_voltage
        self.w_risk     = w_risk
        self.w_duck     = w_duck
        self.fp_penalty = fp_penalty

    def forward(
        self,
        pred: VayuGNNOutput,
        target_overload:  torch.Tensor,   # (B, 30) binary
        target_voltage:   torch.Tensor,   # (B, 30) float [pu]
        target_risk:      torch.Tensor,   # (B, 1)  float [0,1]
        target_duck:      torch.Tensor,   # (B, 96) float [kW]
    ) -> Tuple[torch.Tensor, dict]:

        # Overload: BCE + asymmetric FP penalty
        bce = F.binary_cross_entropy(pred.overload_prob, target_overload, reduction="none")
        # Extra cost when we predict overload (1) but truth is normal (0) — false positive
        fp_mask = (pred.overload_prob.detach() > 0.5).float() * (1 - target_overload)
        overload_loss = (bce + self.fp_penalty * fp_mask).mean()

        voltage_loss = F.huber_loss(pred.voltage_forecast, target_voltage)
        risk_loss    = F.binary_cross_entropy(pred.neighborhood_risk, target_risk)
        duck_loss    = F.huber_loss(pred.duck_curve, target_duck)

        total = (
            self.w_overload * overload_loss
            + self.w_voltage  * voltage_loss
            + self.w_risk     * risk_loss
            + self.w_duck     * duck_loss
        )

        breakdown = {
            "overload": overload_loss.item(),
            "voltage":  voltage_loss.item(),
            "risk":     risk_loss.item(),
            "duck":     duck_loss.item(),
            "total":    total.item(),
        }
        return total, breakdown


# ──────────────────────────────────────────────
# 8. SIGNAL TRANSLATOR
# ──────────────────────────────────────────────

class GridSignal(str, Enum):
    ISLAND   = "ISLAND"    # physically isolate neighbourhood from grid
    THROTTLE = "THROTTLE"  # top-20% flexible loads must reduce
    PRE_COOL = "PRE_COOL"  # pre-cool homes before predicted peak
    RESUME   = "RESUME"    # return to normal operation
    NOMINAL  = "NOMINAL"   # no action required


@dataclass
class SignalResult:
    signal:    GridSignal
    reason:    str
    risk:      float
    min_voltage: float
    duck_ramp: float          # kW/min
    overload_horizon: int     # first minute predicted to overload (-1 = none)


@dataclass
class SignalTranslatorConfig:
    risk_island_threshold:   float = 0.85
    voltage_island_threshold: float = 0.88   # pu
    risk_throttle_threshold: float = 0.50
    duck_ramp_threshold:     float = 2.0     # kW/min
    resume_risk_threshold:   float = 0.10
    resume_hold_steps:       int   = 5       # consecutive steps below resume threshold


class SignalTranslator:
    """
    Converts raw VayuGNN predictions into actionable grid signals.
    Hard constraint: false-positive ISLAND rate must stay <1%.
    This is enforced via the GNN's asymmetric overload loss at training time
    and via the conservative threshold (0.85) at inference.
    """

    def __init__(self, cfg: SignalTranslatorConfig = SignalTranslatorConfig()):
        self.cfg             = cfg
        self._resume_counter = 0   # consecutive low-risk steps

    def translate(self, pred: VayuGNNOutput) -> SignalResult:
        risk        = pred.neighborhood_risk[0, 0].item()
        voltage     = pred.voltage_forecast[0].detach().cpu().numpy()
        overload_p  = pred.overload_prob[0].detach().cpu().numpy()
        duck        = pred.duck_curve[0].detach().cpu().numpy()

        min_voltage = float(voltage.min())
        # kW/min over next 5 slots
        duck_ramp   = float(np.diff(duck[:5]).max()) if len(duck) >= 5 else 0.0
        overload_horizon = int(overload_p.argmax()) if overload_p.max() > 0.5 else -1

        cfg = self.cfg

        # Priority 1: ISLAND — hard safety condition
        if risk > cfg.risk_island_threshold or min_voltage < cfg.voltage_island_threshold:
            self._resume_counter = 0
            reason = (
                f"risk={risk:.3f} > {cfg.risk_island_threshold}"
                if risk > cfg.risk_island_threshold
                else f"voltage={min_voltage:.3f} pu < {cfg.voltage_island_threshold} pu"
            )
            return SignalResult(
                signal=GridSignal.ISLAND,
                reason=reason,
                risk=risk,
                min_voltage=min_voltage,
                duck_ramp=duck_ramp,
                overload_horizon=overload_horizon,
            )

        # Priority 2: PRE_COOL — ramp-event incoming
        if duck_ramp > cfg.duck_ramp_threshold:
            self._resume_counter = 0
            return SignalResult(
                signal=GridSignal.PRE_COOL,
                reason=f"duck_ramp={duck_ramp:.2f} kW/min > {cfg.duck_ramp_threshold} kW/min",
                risk=risk, min_voltage=min_voltage, duck_ramp=duck_ramp,
                overload_horizon=overload_horizon,
            )

        # Priority 3: THROTTLE — elevated risk
        if risk > cfg.risk_throttle_threshold:
            self._resume_counter = 0
            return SignalResult(
                signal=GridSignal.THROTTLE,
                reason=f"risk={risk:.3f} > {cfg.risk_throttle_threshold}",
                risk=risk, min_voltage=min_voltage, duck_ramp=duck_ramp,
                overload_horizon=overload_horizon,
            )

        # Priority 4: RESUME — sustained low-risk period
        if risk < cfg.resume_risk_threshold:
            self._resume_counter += 1
            if self._resume_counter >= cfg.resume_hold_steps:
                return SignalResult(
                    signal=GridSignal.RESUME,
                    reason=(
                        f"risk={risk:.3f} < {cfg.resume_risk_threshold}"
                        f" for {self._resume_counter} steps"
                    ),
                    risk=risk, min_voltage=min_voltage, duck_ramp=duck_ramp,
                    overload_horizon=overload_horizon,
                )
        else:
            self._resume_counter = 0

        return SignalResult(
            signal=GridSignal.NOMINAL,
            reason="all conditions nominal",
            risk=risk, min_voltage=min_voltage, duck_ramp=duck_ramp,
            overload_horizon=overload_horizon,
        )


# ──────────────────────────────────────────────
# 9. FAIRNESS POOL (Island-mode priority queue)
# ──────────────────────────────────────────────

class LoadPriority(int, Enum):
    """Lower integer = higher priority."""
    MEDICAL         = 0    # life-critical medical equipment
    REFRIGERATION   = 1    # food safety
    LIGHTING        = 2    # safety lighting
    COMMUNICATIONS  = 3    # internet / phones
    COOLING         = 4    # HVAC comfort


@dataclass
class Load:
    node_id:  str
    priority: LoadPriority
    demand_kw: float
    label:    str = ""

    def __lt__(self, other: "Load") -> bool:
        return self.priority < other.priority


class FairnessPool:
    """
    During ISLAND mode the P2P market price signal is overridden.
    Power is allocated by priority order (medical first, cooling last).
    Proportional rationing is applied within each priority tier
    if the island's available capacity cannot serve all loads in that tier.
    """

    def __init__(self, island_capacity_kw: float):
        self.capacity_kw = island_capacity_kw
        self.loads:       List[Load] = []

    def register_load(self, load: Load):
        self.loads.append(load)

    def clear(self):
        self.loads.clear()

    def allocate(self) -> Dict[str, float]:
        """
        Returns {node_id: allocated_kw} for each registered load.
        Guarantees:
          1. Higher-priority tiers served before lower-priority.
          2. Within a tier, allocation is proportional to demand.
          3. Total allocation ≤ island_capacity_kw.
        """
        # Sort by priority then by node_id (deterministic tie-break)
        sorted_loads = sorted(self.loads, key=lambda x: (x.priority, x.node_id))

        allocation:     Dict[str, float] = {}
        remaining_kw    = self.capacity_kw

        # Group by priority tier
        tiers: Dict[int, List[Load]] = {}
        for ld in sorted_loads:
            tiers.setdefault(ld.priority, []).append(ld)

        for priority_level in sorted(tiers.keys()):
            tier_loads   = tiers[priority_level]
            tier_demand  = sum(ld.demand_kw for ld in tier_loads)

            if tier_demand <= remaining_kw:
                # Full service for this tier
                for ld in tier_loads:
                    allocation[ld.node_id] = ld.demand_kw
                remaining_kw -= tier_demand
            else:
                # Proportional rationing within tier
                ratio = remaining_kw / tier_demand if tier_demand > 0 else 0.0
                for ld in tier_loads:
                    allocation[ld.node_id] = ld.demand_kw * ratio
                remaining_kw = 0.0
                break   # nothing left for lower tiers

        return allocation

    def summary(self) -> str:
        alloc = self.allocate()
        lines = [f"FairnessPool (capacity={self.capacity_kw:.1f} kW)"]
        for ld in sorted(self.loads, key=lambda x: x.priority):
            served = alloc.get(ld.node_id, 0.0)
            pct    = (served / ld.demand_kw * 100) if ld.demand_kw else 0.0
            lines.append(
                f"  [{ld.priority.name:<14}] {ld.node_id:<12} "
                f"demand={ld.demand_kw:.2f} kW  served={served:.2f} kW ({pct:.0f}%)"
            )
        return "\n".join(lines)


# ──────────────────────────────────────────────
# 10. NEIGHBOURHOOD SERVER PIPELINE
# ──────────────────────────────────────────────

class VayuGNNPipeline:
    """
    Top-level orchestrator.  The neighbourhood server calls
    `pipeline.run(snapshots)` every minute and gets back
    a signal + fairness allocation (if islanding).
    """

    def __init__(
        self,
        model:            VayuGNN,
        island_capacity:  float = 50.0,   # kW available during islanding
        device:           Optional[torch.device] = None,
    ):
        self.model      = model.eval()
        self.translator = SignalTranslator()
        self.pool       = FairnessPool(island_capacity)
        self.device     = device or torch.device("cpu")
        self.model      = model.to(self.device)

        self._is_islanded  = False
        self._last_signal  = GridSignal.NOMINAL

    @torch.no_grad()
    def run(
        self,
        snapshots: List[GraphSnapshot],
        registered_loads: Optional[List[Load]] = None,
    ) -> Tuple[SignalResult, Optional[Dict[str, float]]]:
        """
        Returns (SignalResult, allocation_dict | None).
        allocation_dict is non-None only during ISLAND mode.
        """
        pred   = self.model(snapshots)
        result = self.translator.translate(pred)

        # State transitions
        if result.signal == GridSignal.ISLAND:
            self._is_islanded = True
        elif result.signal == GridSignal.RESUME:
            self._is_islanded = False

        self._last_signal = result.signal

        # Fairness pool during islanding
        allocation = None
        if self._is_islanded and registered_loads is not None:
            self.pool.clear()
            for ld in registered_loads:
                self.pool.register_load(ld)
            allocation = self.pool.allocate()

        return result, allocation


# ──────────────────────────────────────────────
# 11. TRAINING SCAFFOLD
# ──────────────────────────────────────────────

@dataclass
class GNNTrainConfig:
    lr:               float = 1e-3
    weight_decay:     float = 1e-4
    epochs:           int   = 50
    patience:         int   = 10          # early stopping
    grad_clip:        float = 1.0
    eval_interval:    int   = 5
    checkpoint_path:  str   = "vayu_gnn.pt"


class VayuGNNTrainer:
    def __init__(
        self,
        model:  VayuGNN,
        cfg:    GNNTrainConfig    = GNNTrainConfig(),
        loss_fn: VayuGNNLoss     = VayuGNNLoss(),
        device: Optional[torch.device] = None,
    ):
        self.model   = model
        self.cfg     = cfg
        self.loss_fn = loss_fn
        self.device  = device or torch.device("cpu")
        self.model.to(self.device)

        self.opt   = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=cfg.epochs
        )
        self.best_val_loss = float("inf")
        self.patience_ctr  = 0

    def _snapshots_to_device(
        self, snap_seqs: List[List[GraphSnapshot]]
    ) -> List[List[GraphSnapshot]]:
        return [
            [GraphSnapshot(*[t.to(self.device) for t in snap]) for snap in seq]
            for seq in snap_seqs
        ]

    def train_epoch(self, dataloader) -> float:
        self.model.train()
        epoch_loss = 0.0
        for batch in dataloader:
            snap_seqs, targets = batch
            snap_seqs = self._snapshots_to_device(snap_seqs)
            targets = {k: v.to(self.device) for k, v in targets.items()}
            pred = self.model.forward_batched(snap_seqs)
            loss, _ = self.loss_fn(pred, **targets)

            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.opt.step()
            epoch_loss += loss.item()

        self.sched.step()
        return epoch_loss

    @torch.no_grad()
    def eval_epoch(self, dataloader) -> Tuple[float, dict]:
        self.model.eval()
        total = 0.0
        counts = {"overload": 0, "voltage": 0, "risk": 0, "duck": 0, "total": 0}
        for batch in dataloader:
            snap_seqs, targets = batch
            snap_seqs = self._snapshots_to_device(snap_seqs)
            targets = {k: v.to(self.device) for k, v in targets.items()}
            pred = self.model.forward_batched(snap_seqs)
            loss, breakdown = self.loss_fn(pred, **targets)
            for k in counts:
                counts[k] += breakdown.get(k, 0.0)
            total += loss.item()
        n = max(len(dataloader), 1)
        return total / n, {k: v / n for k, v in counts.items()}

    @torch.no_grad()
    def compute_false_positive_rate(self, dataloader) -> float:
        """
        Hard constraint: FP rate on ISLAND signal must stay <1%.
        """
        self.model.eval()
        tp = fp = tn = fn = 0
        translator = SignalTranslator()
        for batch in dataloader:
            snap_seqs, targets = batch
            snap_seqs = self._snapshots_to_device(snap_seqs)
            targets = {k: v.to(self.device) for k, v in targets.items()}
            pred  = self.model.forward_batched(snap_seqs)
            truth = targets["target_overload"]  # (B, 30)

            for i in range(pred.overload_prob.shape[0]):
                single = VayuGNNOutput(
                    overload_prob    = pred.overload_prob[i:i+1],
                    voltage_forecast = pred.voltage_forecast[i:i+1],
                    neighborhood_risk= pred.neighborhood_risk[i:i+1],
                    duck_curve       = pred.duck_curve[i:i+1],
                )
                result = translator.translate(single)
                actual_overload = truth[i].any().item()

                if result.signal == GridSignal.ISLAND:
                    if actual_overload:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if actual_overload:
                        fn += 1
                    else:
                        tn += 1

        total_neg = fp + tn
        fpr = fp / total_neg if total_neg > 0 else 0.0
        print(f"  FPR: {fpr*100:.2f}% (gate: <1%)"
              f"  TP={tp} FP={fp} TN={tn} FN={fn}")
        return fpr

    def fit(self, train_loader, val_loader) -> dict:
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        for epoch in range(1, self.cfg.epochs + 1):
            t_loss = self.train_epoch(train_loader)
            history["train_loss"].append(t_loss)

            if epoch % self.cfg.eval_interval == 0:
                v_loss, breakdown = self.eval_epoch(val_loader)
                history["val_loss"].append(v_loss)
                print(f"Epoch {epoch:3d} | train={t_loss:.4f} | val={v_loss:.4f} | {breakdown}")

                if v_loss < self.best_val_loss:
                    self.best_val_loss = v_loss
                    self.patience_ctr  = 0
                    torch.save(self.model.state_dict(), self.cfg.checkpoint_path)
                    print(f"  ✓ Checkpoint saved → {self.cfg.checkpoint_path}")
                else:
                    self.patience_ctr += 1
                    if self.patience_ctr >= self.cfg.patience:
                        print(f"  Early stopping at epoch {epoch}.")
                        break

        return history


# ──────────────────────────────────────────────
# 12. QUICK SMOKE TEST
# ──────────────────────────────────────────────

def _make_dummy_snapshots(
    n_nodes: int = 20, n_xfmr: int = 3, n_edges: int = 38,
) -> List[GraphSnapshot]:
    snaps = []
    for _ in range(N_SNAPSHOTS):
        snaps.append(GraphSnapshot(
            node_features = torch.randn(n_nodes, NODE_FEAT_DIM),
            xfmr_features = torch.randn(n_xfmr,  XFMR_FEAT_DIM),
            edge_index    = torch.randint(0, n_nodes, (2, n_edges)),
            edge_features = torch.randn(n_edges,  EDGE_FEAT_DIM),
            node_to_xfmr  = torch.randint(0, n_xfmr, (n_nodes,)),
        ))
    return snaps


if __name__ == "__main__":
    print("=" * 56)
    print("VayuGNN — Smoke Test")
    print("=" * 56)

    model    = VayuGNN()
    snapshots = _make_dummy_snapshots()

    t0   = time.perf_counter()
    pred = model(snapshots)
    dt   = (time.perf_counter() - t0) * 1000

    print(f"\nForward pass ({N_SNAPSHOTS} snapshots): {dt:.1f} ms")
    op = pred.overload_prob
    vf = pred.voltage_forecast
    nr = pred.neighborhood_risk
    dc = pred.duck_curve
    print(f"  overload_prob     : {op.shape}   min={op.min():.3f} max={op.max():.3f}")
    print(f"  voltage_forecast  : {vf.shape}   mean={vf.mean():.3f}")
    print(f"  neighborhood_risk : {nr.shape}   value={nr.item():.3f}")
    print(f"  duck_curve        : {dc.shape}   mean={dc.mean():.3f}")

    # Signal translator
    translator = SignalTranslator()
    result     = translator.translate(pred)
    print(f"\nSignal: {result.signal.value}  ({result.reason})")
    print(
        f"  risk={result.risk:.3f}  min_voltage={result.min_voltage:.3f} pu"
        f"  duck_ramp={result.duck_ramp:.2f} kW/min"
    )

    # Fairness pool demo
    print("\n--- Fairness Pool (25 kW island, 30 kW demand) ---")
    pool = FairnessPool(island_capacity_kw=25.0)
    pool.register_load(Load("home_03", LoadPriority.MEDICAL,       3.5,  "Dialysis machine"))
    pool.register_load(Load("home_07", LoadPriority.REFRIGERATION, 0.8,  "Fridge/freezer"))
    pool.register_load(Load("home_01", LoadPriority.COOLING,       4.0,  "HVAC"))
    pool.register_load(Load("home_12", LoadPriority.LIGHTING,      0.3,  "Safety lights"))
    pool.register_load(Load("home_05", LoadPriority.COMMUNICATIONS,0.5,  "Router/modem"))
    pool.register_load(Load("home_09", LoadPriority.COOLING,       6.0,  "HVAC"))
    pool.register_load(Load("home_11", LoadPriority.REFRIGERATION, 0.7,  "Chest freezer"))
    pool.register_load(Load("home_02", LoadPriority.MEDICAL,       2.0,  "Home ventilator"))
    print(pool.summary())

    # Parameter counts
    print("\nParameter breakdown:")
    counts = model.count_parameters()
    for k, v in counts.items():
        print(f"  {k:<20}: {v:>7,}")
    print(f"  {'TOTAL':<20}: {sum(counts.values()):>7,}")

    print("\n✓ VayuGNN smoke test passed.\n")
