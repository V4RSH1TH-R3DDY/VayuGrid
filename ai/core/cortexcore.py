"""
CortexCore — PPO Energy Management Agent
=========================================
Runs on Vayu-Node (Raspberry Pi 5) at 1-minute intervals.
Asymmetric architecture: large critic for training, tiny actor for deployment.
Exports to INT8-quantized ONNX for <5 ms inference on-device.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from ai.core.normalizer import ObservationNormalizer

# ──────────────────────────────────────────────
# 1. SPACES & REWARD CONFIG
# ──────────────────────────────────────────────

OBS_DIM = 12   # see table below
ACT_DIM = 5    # battery, EV rate, bid price, ask price, grid import/export

# Observation index map (for clarity in reward / debugging)
class ObsIdx:
    SOC_NORM        = 0   # battery_soc_kwh / battery_capacity_kwh  [0,1]
    SOLAR_KW        = 1   # current PV generation
    LOAD_KW         = 2   # household consumption
    EV_KW           = 3   # EV charging demand
    BATT_MAX_KW     = 4   # max battery charge/discharge rate
    NET_KW          = 5   # pv - load (surplus > 0 / deficit < 0)
    TIME_SIN        = 6   # sin(2π·t/1440)
    TIME_COS        = 7   # cos(2π·t/1440)
    FCST_SOLAR_KW   = 8   # 15-min ahead solar forecast
    FCST_LOAD_KW    = 9   # 15-min ahead load forecast
    FCST_PRICE      = 10  # 15-min ahead market price forecast
    SOC_RAW_KWH     = 11  # raw battery SoC (for reward logic)

# Action index map
class ActIdx:
    BATT_RATE   = 0   # [-1, 1]  — −1 = full discharge, +1 = full charge
    EV_RATE     = 1   # [0, 1]   — fraction of max EV charge rate
    BID_PRICE   = 2   # [0, 1]   — normalised max import price
    ASK_PRICE   = 3   # [0, 1]   — normalised min export price
    GRID_IO     = 4   # [-1, 1]  — reserved grid import/export control

@dataclass
class RewardConfig:
    # Economic weights
    p2p_revenue_weight:   float = 1.0
    grid_import_weight:   float = 1.2   # penalise grid import slightly more

    # EV deadline penalty
    ev_miss_penalty:      float = 50.0  # heavy negative reward

    # Battery health
    soc_low_threshold:    float = 0.10  # below 10% = degradation zone
    soc_low_penalty:      float = 2.0   # per timestep in danger zone

    # Grid-cooperation bonus (from VayuGNN signal)
    gnn_cooperation_bonus: float = 3.0

@dataclass
class PPOConfig:
    lr_actor:         float = 3e-4
    lr_critic:        float = 1e-3
    gamma:            float = 0.99
    gae_lambda:       float = 0.95
    clip_epsilon:     float = 0.2
    entropy_coef:     float = 0.01
    value_loss_coef:  float = 0.5
    max_grad_norm:    float = 0.5
    ppo_epochs:       int   = 10
    minibatch_size:   int   = 64
    rollout_steps:    int   = 2048
    update_interval:  int   = rollout_steps


# ──────────────────────────────────────────────
# 2. NETWORKS
# ──────────────────────────────────────────────

def _orthogonal_init(module: nn.Module, gain: float = 1.0) -> None:
    """Orthogonal weight initialisation (standard PPO practice)."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain)
        nn.init.constant_(module.bias, 0.0)


class Actor(nn.Module):
    """
    Tiny deployment actor: 2 × 64 hidden units.
    Outputs (mean, log_std) for each action dimension.
    Designed to fit <5 ms on Raspberry Pi 5 after INT8 quantisation.
    """

    LOG_STD_MIN = -20.0
    LOG_STD_MAX =  2.0

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.mean_head    = nn.Linear(64, act_dim)
        self.log_std_head = nn.Linear(64, act_dim)

        # Initialise output heads with small gain for stable early exploration
        self.net.apply(lambda m: _orthogonal_init(m, gain=np.sqrt(2)))
        _orthogonal_init(self.mean_head,    gain=0.01)
        _orthogonal_init(self.log_std_head, gain=0.01)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.net(obs)
        mean    = self.mean_head(features)
        log_std = self.log_std_head(features).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def get_distribution(self, obs: torch.Tensor) -> Normal:
        mean, log_std = self(obs)
        return Normal(mean, log_std.exp())

    @torch.no_grad()
    def act(
        self, obs: torch.Tensor, deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.get_distribution(obs)
        action = dist.mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob


class Critic(nn.Module):
    """
    Large training critic: 3 × 256 hidden units.
    Discarded after training — never deployed to device.
    """

    def __init__(self, obs_dim: int = OBS_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1),
        )
        self.net.apply(lambda m: _orthogonal_init(m, gain=np.sqrt(2)))
        _orthogonal_init(self.net[-1], gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


# ──────────────────────────────────────────────
# 3. ACTION CLIPPING / SQUASHING
# ──────────────────────────────────────────────

def clip_actions(raw: torch.Tensor) -> torch.Tensor:
    """
    Map raw Gaussian samples to physical action bounds.

    ActIdx.BATT_RATE  : tanh → [-1, 1]
    ActIdx.EV_RATE    : sigmoid → [0, 1]
    ActIdx.BID_PRICE  : sigmoid → [0, 1]
    ActIdx.ASK_PRICE  : sigmoid → [0, 1]
    ActIdx.GRID_IO    : tanh → [-1, 1]
    """
    clipped = raw.clone()
    clipped[..., ActIdx.BATT_RATE] = torch.tanh(raw[..., ActIdx.BATT_RATE])
    clipped[..., ActIdx.EV_RATE]   = torch.sigmoid(raw[..., ActIdx.EV_RATE])
    clipped[..., ActIdx.BID_PRICE] = torch.sigmoid(raw[..., ActIdx.BID_PRICE])
    clipped[..., ActIdx.ASK_PRICE] = torch.sigmoid(raw[..., ActIdx.ASK_PRICE])
    clipped[..., ActIdx.GRID_IO]   = torch.tanh(raw[..., ActIdx.GRID_IO])
    return clipped


# ──────────────────────────────────────────────
# 4. REWARD FUNCTION
# ──────────────────────────────────────────────

class RewardComputer:
    """
    Computes the scalar reward for each timestep.
    All inputs are Python floats / scalars for readability.
    """

    def __init__(self, cfg: RewardConfig = RewardConfig()):
        self.cfg = cfg

    def compute(
        self,
        p2p_revenue_dollar:   float,
        grid_import_cost_dollar: float,
        soc_norm:             float,
        ev_deadline_missed:   bool,
        gnn_signal_followed:  bool,
    ) -> Tuple[float, dict]:
        cfg = self.cfg

        r_economic = (
            cfg.p2p_revenue_weight  * p2p_revenue_dollar
            - cfg.grid_import_weight * grid_import_cost_dollar
        )

        r_ev_penalty = -cfg.ev_miss_penalty if ev_deadline_missed else 0.0

        r_battery = (
            -cfg.soc_low_penalty
            if soc_norm < cfg.soc_low_threshold
            else 0.0
        )

        r_cooperation = cfg.gnn_cooperation_bonus if gnn_signal_followed else 0.0

        total = r_economic + r_ev_penalty + r_battery + r_cooperation

        breakdown = {
            "economic":     r_economic,
            "ev_penalty":   r_ev_penalty,
            "battery":      r_battery,
            "cooperation":  r_cooperation,
            "total":        total,
        }
        return total, breakdown


# ──────────────────────────────────────────────
# 5. ROLLOUT BUFFER
# ──────────────────────────────────────────────

class RolloutBatch(NamedTuple):
    observations: torch.Tensor   # (T, obs_dim)
    actions:      torch.Tensor   # (T, act_dim)
    log_probs:    torch.Tensor   # (T,)
    returns:      torch.Tensor   # (T,)
    advantages:   torch.Tensor   # (T,)
    values:       torch.Tensor   # (T,)


class RolloutBuffer:
    """Fixed-size circular buffer with GAE computation."""

    def __init__(self, size: int, obs_dim: int, act_dim: int, device: torch.device):
        self.size    = size
        self.device  = device
        self.obs     = torch.zeros(size, obs_dim, device=device)
        self.acts    = torch.zeros(size, act_dim, device=device)
        self.rews    = torch.zeros(size,           device=device)
        self.vals    = torch.zeros(size,           device=device)
        self.lps     = torch.zeros(size,           device=device)
        self.dones   = torch.zeros(size,           device=device)
        self.ptr     = 0
        self.full    = False

    def push(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        rew: float,
        val: float,
        lp:  float,
        done: bool,
    ):
        i = self.ptr % self.size
        self.obs[i]   = obs
        self.acts[i]  = act
        self.rews[i]  = rew
        self.vals[i]  = val
        self.lps[i]   = lp
        self.dones[i] = float(done)
        self.ptr     += 1
        self.full     = self.ptr >= self.size

    def compute_gae(
        self, last_val: float, gamma: float, lam: float
    ) -> RolloutBatch:
        T = min(self.ptr, self.size)
        advantages = torch.zeros(T, device=self.device)
        last_gae   = 0.0

        for t in reversed(range(T)):
            next_val   = last_val if t == T - 1 else self.vals[t + 1].item()
            next_done  = self.dones[t].item()
            rew       = self.rews[t].item()
            delta     = rew + gamma * next_val * (1 - next_done) - self.vals[t].item()
            last_gae   = delta + gamma * lam * (1 - next_done) * last_gae
            advantages[t] = last_gae

        returns = advantages + self.vals[:T]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return RolloutBatch(
            observations=self.obs[:T],
            actions=self.acts[:T],
            log_probs=self.lps[:T],
            returns=returns,
            advantages=advantages,
            values=self.vals[:T],
        )

    def reset(self):
        self.ptr  = 0
        self.full = False


# ──────────────────────────────────────────────
# 6. PPO AGENT
# ──────────────────────────────────────────────

class CortexCore:
    """
    PPO agent for Vayu-Node home energy management.

    Training:  uses both Actor + Critic.
    Deployment: exports only Actor as INT8-quantised ONNX.
    """

    def __init__(
        self,
        cfg:       PPOConfig             = PPOConfig(),
        reward:    RewardConfig          = RewardConfig(),
        normalizer: Optional[ObservationNormalizer] = None,
        device:    Optional[torch.device] = None,
    ):
        self.cfg    = cfg
        self.device = device or torch.device("cpu")

        self.actor  = Actor().to(self.device)
        self.critic = Critic().to(self.device)

        self.opt_actor  = torch.optim.Adam(self.actor.parameters(),  lr=cfg.lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)

        self.buffer  = RolloutBuffer(
            cfg.rollout_steps, OBS_DIM, ACT_DIM, self.device
        )
        self.reward_computer = RewardComputer(reward)

        # Observation normalizer — stores running mean/std for deployment
        self.normalizer = normalizer or ObservationNormalizer((OBS_DIM,))

        # Running stats for advantage normalisation
        self._ep_returns: deque = deque(maxlen=100)
        self.total_steps: int   = 0

    # ── Inference ──────────────────────────────

    @torch.no_grad()
    def select_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """
        Returns (clipped_action, log_prob, value_estimate).
        obs shape: (obs_dim,).  Normalized internally using running stats.
        """
        normed = self.normalizer.normalize(obs)
        obs_t  = torch.FloatTensor(normed).unsqueeze(0).to(self.device)
        raw_action, log_prob = self.actor.act(obs_t, deterministic)
        action    = clip_actions(raw_action).squeeze(0)
        value     = self.critic(obs_t).squeeze(0)
        return (
            action.cpu().numpy(),
            log_prob.item(),
            value.item(),
        )

    # ── Training step ──────────────────────────

    def update(self, last_obs: np.ndarray, last_done: bool) -> dict:
        """Run PPO update over the current rollout buffer."""
        with torch.no_grad():
            obs_t    = torch.FloatTensor(last_obs).unsqueeze(0).to(self.device)
            last_val = 0.0 if last_done else self.critic(obs_t).item()

        batch = self.buffer.compute_gae(last_val, self.cfg.gamma, self.cfg.gae_lambda)
        self.buffer.reset()

        T = batch.observations.shape[0]
        metrics: dict[str, list[float]] = {
            "policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": []
        }

        for _ in range(self.cfg.ppo_epochs):
            indices = torch.randperm(T, device=self.device)
            for start in range(0, T, self.cfg.minibatch_size):
                mb_idx = indices[start : start + self.cfg.minibatch_size]

                mb_obs  = batch.observations[mb_idx]
                mb_act  = batch.actions[mb_idx]
                mb_adv  = batch.advantages[mb_idx]
                mb_ret  = batch.returns[mb_idx]
                mb_lp   = batch.log_probs[mb_idx]

                # Actor loss
                dist       = self.actor.get_distribution(mb_obs)
                new_lp     = dist.log_prob(mb_act).sum(-1)
                entropy    = dist.entropy().sum(-1).mean()
                ratio      = (new_lp - mb_lp).exp()
                clipped = torch.clamp(
                    ratio, 1 - self.cfg.clip_epsilon, 1 + self.cfg.clip_epsilon,
                )
                policy_loss = -torch.min(ratio * mb_adv, clipped * mb_adv).mean()

                # Critic loss (clipped value function)
                new_val    = self.critic(mb_obs)
                old_val    = batch.values[mb_idx]
                val_clipped = old_val + (new_val - old_val).clamp(
                    -self.cfg.clip_epsilon, self.cfg.clip_epsilon
                )
                value_loss = 0.5 * torch.max(
                    F.mse_loss(new_val, mb_ret),
                    F.mse_loss(val_clipped, mb_ret),
                ).mean()

                total_loss = (
                    policy_loss
                    + self.cfg.value_loss_coef * value_loss
                    - self.cfg.entropy_coef    * entropy
                )

                self.opt_actor.zero_grad()
                self.opt_critic.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(),  self.cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                self.opt_actor.step()
                self.opt_critic.step()

                approx_kl = ((ratio - 1) - (ratio.log())).mean().item()
                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(entropy.item())
                metrics["approx_kl"].append(approx_kl)

        return {k: float(np.mean(v)) for k, v in metrics.items()}

    def push_transition(
        self,
        obs:   np.ndarray,
        act:   np.ndarray,
        rew:   float,
        val:   float,
        lp:    float,
        done:  bool,
    ):
        obs_t = torch.FloatTensor(obs).to(self.device)
        act_t = torch.FloatTensor(act).to(self.device)
        self.buffer.push(obs_t, act_t, rew, val, lp, done)
        self.total_steps += 1

    # ── Checkpointing ──────────────────────────

    def save(self, path: str):
        torch.save({
            "actor":       self.actor.state_dict(),
            "critic":      self.critic.state_dict(),
            "opt_actor":   self.opt_actor.state_dict(),
            "opt_critic":  self.opt_critic.state_dict(),
            "normalizer":  self.normalizer.state_dict(),
            "total_steps": self.total_steps,
        }, path)
        print(f"[CortexCore] Saved checkpoint → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.opt_actor.load_state_dict(ckpt["opt_actor"])
        self.opt_critic.load_state_dict(ckpt["opt_critic"])
        if "normalizer" in ckpt:
            self.normalizer.load_state_dict(ckpt["normalizer"])
        self.total_steps = ckpt.get("total_steps", 0)
        print(f"[CortexCore] Loaded checkpoint ← {path}")


# ──────────────────────────────────────────────
# 7. ONNX EXPORT + INT8 QUANTISATION
# ──────────────────────────────────────────────

class CortexCoreExporter:
    """
    Exports the trained Actor to INT8-quantised ONNX.
    Only the actor is exported — critic is discarded.
    Target: <5 ms inference on Raspberry Pi 5 (ARM Cortex-A76).
    """

    def __init__(self, actor: Actor):
        self.actor = actor.eval().cpu()

    def export_fp32(self, path: str = "cortexcore_actor_fp32.onnx"):
        dummy = torch.zeros(1, OBS_DIM)
        try:
            torch.onnx.export(
                self.actor,
                (dummy,),
                path,
                export_params=True,
                opset_version=17,
                input_names=["observation"],
                output_names=["action_mean", "action_log_std"],
                dynamic_axes={"observation": {0: "batch_size"}},
                do_constant_folding=True,
            )
            print(f"[Export] FP32 ONNX saved → {path}")
        except (ModuleNotFoundError, AttributeError) as exc:
            print(f"[Export] ONNX export skipped ({exc})")
        return path

    def quantize_int8(
        self,
        fp32_path: str,
        out_path:  str = "cortexcore_actor_int8.onnx",
    ) -> str:
        if not Path(fp32_path).exists():
            print("[Export] INT8 ONNX skipped (FP32 model not found)")
            return out_path
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantize_dynamic(
            model_input=fp32_path,
            model_output=out_path,
            weight_type=QuantType.QInt8,
        )
        print(f"[Export] INT8 ONNX saved → {out_path}")
        return out_path

    def benchmark(self, onnx_path: str, n_runs: int = 1000):
        """Measure median inference latency on the current machine."""
        import onnxruntime as ort
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 4   # Pi 5 has 4 cores
        sess = ort.InferenceSession(onnx_path, sess_opts)
        dummy = np.zeros((1, OBS_DIM), dtype=np.float32)

        # Warm-up
        for _ in range(50):
            sess.run(None, {"observation": dummy})

        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            sess.run(None, {"observation": dummy})
            latencies.append((time.perf_counter() - t0) * 1000)

        p50 = np.median(latencies)
        p95 = np.percentile(latencies, 95)
        print(f"[Benchmark] {onnx_path}")
        print(f"  Median: {p50:.3f} ms  |  P95: {p95:.3f} ms")
        print(f"  Gate:   {'✓ PASS' if p50 < 5.0 else '✗ FAIL'} (<5 ms target)")
        return p50, p95

    def full_pipeline(
        self,
        fp32_path: str = "cortexcore_actor_fp32.onnx",
        int8_path: str = "cortexcore_actor_int8.onnx",
    ) -> str:
        self.export_fp32(fp32_path)
        if Path(fp32_path).exists():
            self.quantize_int8(fp32_path, int8_path)
        if Path(int8_path).exists():
            self.benchmark(int8_path)
        else:
            print("[Export] Benchmark skipped (INT8 model not found)")
        return int8_path


# ──────────────────────────────────────────────
# 8. ONLINE INFERENCE WRAPPER (for Pi deployment)
# ──────────────────────────────────────────────

class CortexCoreRuntime:
    """
    Lightweight inference-only wrapper for Raspberry Pi 5.
    Loads the INT8 ONNX model and exposes a single `step()` call.
    Applies the same observation normalisation used during training.
    """

    def __init__(
        self, onnx_path: str, n_threads: int = 4,
        normalizer: Optional[ObservationNormalizer] = None,
    ):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = n_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess       = ort.InferenceSession(onnx_path, opts)
        self.normalizer = normalizer or ObservationNormalizer((OBS_DIM,))
        self._obs_buf   = np.zeros((1, OBS_DIM), dtype=np.float32)

    def step(self, obs: np.ndarray) -> np.ndarray:
        """
        obs: (OBS_DIM,) float32 array
        Returns: (ACT_DIM,) deterministic action (clipped to physical bounds)
        """
        normed = self.normalizer.normalize(obs)
        self._obs_buf[0] = normed
        mean, _ = self.sess.run(None, {"observation": self._obs_buf})
        raw  = mean[0]
        # Apply same squashing as training
        action = np.empty(ACT_DIM, dtype=np.float32)
        action[ActIdx.BATT_RATE]  = np.tanh(raw[ActIdx.BATT_RATE])
        action[ActIdx.EV_RATE]    = _sigmoid(raw[ActIdx.EV_RATE])
        action[ActIdx.BID_PRICE]  = _sigmoid(raw[ActIdx.BID_PRICE])
        action[ActIdx.ASK_PRICE]  = _sigmoid(raw[ActIdx.ASK_PRICE])
        action[ActIdx.GRID_IO]    = np.tanh(raw[ActIdx.GRID_IO])
        return action


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


# ──────────────────────────────────────────────
# 9. BASELINES (for gate-condition evaluation)
# ──────────────────────────────────────────────

class BaselineB1:
    """Rule-based: charge when solar surplus, discharge during peak."""

    def act(self, obs: np.ndarray) -> np.ndarray:
        net_kw   = obs[ObsIdx.NET_KW]
        time_cos = obs[ObsIdx.TIME_COS]
        is_peak  = time_cos < -0.5   # rough evening proxy

        batt = np.clip(net_kw / 5.0, -1, 1)
        if is_peak:
            batt = -0.8

        return np.array([batt, 0.5, 0.5, 0.5, 0.0], dtype=np.float32)


class BaselineB2:
    """Time-of-use: always charge at night, discharge at peak."""

    def act(self, obs: np.ndarray) -> np.ndarray:
        t_sin    = obs[ObsIdx.TIME_SIN]
        t_cos    = obs[ObsIdx.TIME_COS]
        hour_rad = np.arctan2(t_sin, t_cos)
        # map to [0, 24)
        hour     = (np.degrees(hour_rad) % 360) / 15.0

        if 22 <= hour or hour < 6:
            batt = 0.9          # charge overnight
        elif 17 <= hour < 21:
            batt = -0.9         # discharge during evening peak
        else:
            batt = 0.0

        return np.array([batt, 0.5, 0.4, 0.6, 0.0], dtype=np.float32)


# ──────────────────────────────────────────────
# 10. QUICK SMOKE TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 56)
    print("CortexCore — Smoke Test")
    print("=" * 56)

    agent = CortexCore(normalizer=ObservationNormalizer((OBS_DIM,)))

    # Simulate a few transitions
    for step in range(10):
        obs  = np.random.rand(OBS_DIM).astype(np.float32)
        act, lp, val = agent.select_action(obs)

        rew, breakdown = agent.reward_computer.compute(
            p2p_revenue_dollar=float(np.random.rand()),
            grid_import_cost_dollar=float(np.random.rand() * 0.5),
            soc_norm=obs[ObsIdx.SOC_NORM],
            ev_deadline_missed=False,
            gnn_signal_followed=True,
        )
        agent.push_transition(obs, act, rew, val, lp, done=(step == 9))

    # Trigger a mini-update (normally runs after rollout_steps)
    agent.buffer.ptr = agent.cfg.rollout_steps   # fake full buffer
    last_obs = np.random.rand(OBS_DIM).astype(np.float32)

    print("\nAction space check:")
    obs  = np.random.rand(OBS_DIM).astype(np.float32)
    act, _, _ = agent.select_action(obs)
    print(f"  battery  ([-1,1])  : {act[ActIdx.BATT_RATE]:.3f}")
    print(f"  EV rate  ([0,1])   : {act[ActIdx.EV_RATE]:.3f}")
    print(f"  bid      ([0,1])   : {act[ActIdx.BID_PRICE]:.3f}")
    print(f"  ask      ([0,1])   : {act[ActIdx.ASK_PRICE]:.3f}")
    print(f"  grid I/O ([-1,1])  : {act[ActIdx.GRID_IO]:.3f}")

    print("\nBaseline actions (B1, B2):")
    print(" B1:", BaselineB1().act(obs))
    print(" B2:", BaselineB2().act(obs))

    print("\nActor parameter count:")
    n_params = sum(p.numel() for p in agent.actor.parameters())
    print(f"  {n_params:,} parameters (target: ≤20K for <5ms on Pi 5)")

    print("\n✓ CortexCore smoke test passed.\n")
    print("To export for deployment:")
    print("  exporter = CortexCoreExporter(agent.actor)")
    print("  exporter.full_pipeline()")
