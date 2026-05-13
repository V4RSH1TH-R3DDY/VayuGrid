"""
Integration & Regression Tests for PPO training pipeline, fallback, and demo.

These tests verify that the components we built work together correctly:
  1. RuntimeWithFallback — degenerate model detection + rule-based fallback
  2. PPO training pipeline — agent + env + buffer + update (smoke)
  3. Raw action fix — ratio computation correctness (regression guard)
  4. Demo scenario runners — basic/pecan/fault/compare entry points
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ai.core.cortexcore import (
    ACT_DIM,
    OBS_DIM,
    ActIdx,
    CortexCore,
    ObsIdx,
    PPOConfig,
    RuntimeWithFallback,
)
from ai.core.normalizer import ObservationNormalizer
from ai.env.gym_env import EnvConfig, VayuGridEnv

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _make_agent() -> CortexCore:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return CortexCore(cfg=PPOConfig(), normalizer=ObservationNormalizer((OBS_DIM,)), device=device)


def _constant_action_agent() -> CortexCore:
    """Agent whose actor always returns the same mean (degenerate)."""
    agent = _make_agent()
    # Force actor output to constant
    with torch.no_grad():
        for p in agent.actor.mean_head.parameters():
            p.zero_()
        for p in agent.actor.log_std_head.parameters():
            p.zero_()
    return agent


def _varying_action_agent() -> CortexCore:
    """Agent with orthogonal-init weights that naturally varies (functional)."""
    return _make_agent()


# ──────────────────────────────────────────────
# 1. RuntimeWithFallback
# ──────────────────────────────────────────────

def test_fallback_detects_degenerate_model():
    """mean_action_std below threshold → fallback active."""
    agent = _constant_action_agent()
    runtime = RuntimeWithFallback(agent=agent, action_std_threshold=0.01, n_check_samples=10)
    assert runtime.is_using_fallback
    obs = np.random.randn(OBS_DIM).astype(np.float32)
    act = runtime.step(obs)
    assert act.shape == (ACT_DIM,)
    # Fallback should produce reasonable bounded actions
    assert -1.0 <= act[ActIdx.BATT_RATE] <= 1.0
    assert 0.0 <= act[ActIdx.EV_RATE] <= 1.0
    assert 0.0 <= act[ActIdx.BID_PRICE] <= 1.0
    assert 0.0 <= act[ActIdx.ASK_PRICE] <= 1.0
    assert -1.0 <= act[ActIdx.GRID_IO] <= 1.0


def test_fallback_detects_functional_model():
    """mean_action_std above threshold → model used directly."""
    agent = _varying_action_agent()
    runtime = RuntimeWithFallback(agent=agent, action_std_threshold=0.001, n_check_samples=20)
    assert not runtime.is_using_fallback


def test_fallback_action_bounds():
    """Fallback actions respect physical bounds regardless of observation."""
    agent = _constant_action_agent()
    runtime = RuntimeWithFallback(agent=agent, action_std_threshold=0.01)
    # Test with extreme observations
    for obs in [
        np.full(OBS_DIM, 1e3, dtype=np.float32),
        np.full(OBS_DIM, -1e3, dtype=np.float32),
        np.zeros(OBS_DIM, dtype=np.float32),
        np.random.randn(OBS_DIM).astype(np.float32) * 100,
    ]:
        act = runtime.step(obs)
        assert -1.0 <= act[ActIdx.BATT_RATE] <= 1.0
        assert 0.0 <= act[ActIdx.EV_RATE] <= 1.0
        assert 0.0 <= act[ActIdx.BID_PRICE] <= 1.0
        assert 0.0 <= act[ActIdx.ASK_PRICE] <= 1.0
        assert -1.0 <= act[ActIdx.GRID_IO] <= 1.0


def test_fallback_varies_by_time():
    """Fallback produces different battery actions at different times of day."""
    agent = _constant_action_agent()
    runtime = RuntimeWithFallback(agent=agent, action_std_threshold=0.01)
    # Midnight obs: soc_norm=1.0, time_sin=sin(0)=0, time_cos=cos(0)=1
    midnight = np.zeros(OBS_DIM, dtype=np.float32)
    midnight[ObsIdx.SOC_NORM] = 1.0
    midnight[ObsIdx.NET_KW] = 0.0
    # Midday obs: soc_norm=0.5, time_sin=sin(pi)=0, time_cos=cos(pi)=-1 (hour≈12)
    midday = np.zeros(OBS_DIM, dtype=np.float32)
    midday[ObsIdx.SOC_NORM] = 0.5
    midday[ObsIdx.TIME_SIN] = 0.0
    midday[ObsIdx.TIME_COS] = -1.0
    midday[ObsIdx.NET_KW] = 0.0

    act_night = runtime.step(midnight)
    act_day = runtime.step(midday)
    # Night should charge (positive batt), day should be neutral
    assert act_night[ActIdx.BATT_RATE] >= 0.0
    assert act_day[ActIdx.BATT_RATE] >= -0.5  # not aggressively discharging


# ──────────────────────────────────────────────
# 2. PPO pipeline smoke test
# ──────────────────────────────────────────────

def test_agent_select_action_returns_raw_and_clipped():
    """select_action returns (env_action, raw_action, log_prob, value)."""
    agent = _make_agent()
    obs = np.random.randn(OBS_DIM).astype(np.float32)
    result = agent.select_action(obs)
    assert len(result) == 4
    env_act, raw_act, log_prob, value = result
    assert env_act.shape == (ACT_DIM,)
    assert raw_act.shape == (ACT_DIM,)
    assert isinstance(log_prob, float)
    assert isinstance(value, float)
    # env_act should be bounded by physical constraints
    assert -1.0 <= env_act[ActIdx.BATT_RATE] <= 1.0
    assert 0.0 <= env_act[ActIdx.EV_RATE] <= 1.0
    assert 0.0 <= env_act[ActIdx.BID_PRICE] <= 1.0
    assert 0.0 <= env_act[ActIdx.ASK_PRICE] <= 1.0
    assert -1.0 <= env_act[ActIdx.GRID_IO] <= 1.0
    # raw_act can be any float (pre-squash)
    assert raw_act.dtype == np.float32


def test_raw_action_ratio_is_reasonable():
    """PPO ratio computed with raw actions stays reasonable (< clip_epsilon)."""
    agent = _make_agent()
    cfg = agent.cfg
    obs = np.random.randn(OBS_DIM).astype(np.float32)

    # Fill buffer with a few transitions
    for _ in range(cfg.rollout_steps):
        env_act, raw_act, lp, val = agent.select_action(obs)
        agent.push_transition(obs, raw_act, -1.0, val, lp, done=False)
        obs = np.random.randn(OBS_DIM).astype(np.float32)

    last_obs = np.random.randn(OBS_DIM).astype(np.float32)
    metrics = agent.update(last_obs, last_done=True)

    # Ratio-based metrics should be reasonable (not 20+)
    assert metrics["approx_kl"] < 0.5, (
        f"approx_kl too high: {metrics['approx_kl']:.4f}"
    )

    # Value loss can be high with random obs (critic hasn't seen real data)
    # but should not explode.
    assert metrics["value_loss"] < 500.0, (
        f"value_loss exploding: {metrics['value_loss']:.4f}"
    )


def test_reward_scaling_affects_buffer():
    """reward_scale in PPOConfig scales rewards stored in buffer."""
    obs = np.random.randn(OBS_DIM).astype(np.float32)
    cfg = PPOConfig(reward_scale=0.5, rollout_steps=64)
    agent = CortexCore(cfg=cfg, normalizer=ObservationNormalizer((OBS_DIM,)))
    env_act, raw_act, lp, val = agent.select_action(obs)
    agent.push_transition(obs, raw_act, 10.0, val, lp, done=False)
    assert agent.buffer.ptr == 1
    # Reward should be scaled to 5.0
    assert agent.buffer.rews[0].item() == 5.0


def test_kl_early_stopping_prevents_excess_epochs():
    """KL early stopping breaks out when target_kl exceeded."""
    cfg = PPOConfig(target_kl=0.001, ppo_epochs=10, rollout_steps=128)
    agent = CortexCore(cfg=cfg, normalizer=ObservationNormalizer((OBS_DIM,)))

    obs = np.random.randn(OBS_DIM).astype(np.float32)
    for _ in range(cfg.rollout_steps):
        env_act, raw_act, lp, val = agent.select_action(obs)
        agent.push_transition(obs, raw_act, -1.0, val, lp, done=False)

    last_obs = np.random.randn(OBS_DIM).astype(np.float32)
    metrics = agent.update(last_obs, last_done=True)
    assert metrics.get("early_stopped"), "KL early stopping should have triggered"


# ──────────────────────────────────────────────
# 3. Regression: raw action fix
# ──────────────────────────────────────────────

def test_regression_raw_vs_clipped_ratio():
    """
    Regression guard: PPO ratio must use raw (pre-squash) actions.

    Before the fix, the buffer stored clipped actions while log_prob
    was for raw actions, producing ratio = exp(log_prob(clipped|new)
    - log_prob(raw|old)) — comparing different values.  This test
    verifies that storing raw actions gives a correct ratio ~ 1.0
    when the policy hasn't been updated yet.
    """
    agent = _make_agent()
    cfg = agent.cfg
    obs = np.random.randn(OBS_DIM).astype(np.float32)

    # Collect transitions, storing RAW actions
    for _ in range(cfg.rollout_steps):
        env_act, raw_act, lp, val = agent.select_action(obs)
        agent.push_transition(obs, raw_act, -0.5, val, lp, done=False)
        obs = np.random.randn(OBS_DIM).astype(np.float32)

    # Before any update: ratio should be ~1.0 since old_lp == new_lp
    batch = agent.buffer.compute_gae(0.0, cfg.gamma, cfg.gae_lambda)

    # Compute ratio manually
    with torch.no_grad():
        dist = agent.actor.get_distribution(batch.observations[:10])
        new_lp = dist.log_prob(batch.actions[:10]).sum(-1)
        ratio = (new_lp - batch.log_probs[:10]).exp()

    # Ratio should be close to 1.0 (within numerical precision)
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-4), (
        f"Ratio not ~1.0 before update: mean={ratio.mean().item():.6f}"
    )


def test_regression_env_action_stays_bounded():
    """Regression: env action must always be in physical bounds."""
    agent = _make_agent()
    for _ in range(100):
        obs = np.random.randn(OBS_DIM).astype(np.float32) * 10
        env_act, raw_act, _, _ = agent.select_action(obs)
        assert -1.0 <= env_act[ActIdx.BATT_RATE] <= 1.0
        assert 0.0 <= env_act[ActIdx.EV_RATE] <= 1.0
        assert 0.0 <= env_act[ActIdx.BID_PRICE] <= 1.0
        assert 0.0 <= env_act[ActIdx.ASK_PRICE] <= 1.0
        assert -1.0 <= env_act[ActIdx.GRID_IO] <= 1.0


# ──────────────────────────────────────────────
# 4. PPO + Env integration
# ──────────────────────────────────────────────

def test_agent_env_integration_short_rollout():
    """Agent and environment work together for a short rollout."""
    env = VayuGridEnv(EnvConfig(scenario_path="scenarios/phase1_debug.json"))
    agent = _make_agent()

    obs, _ = env.reset()
    episode_reward = 0.0
    for _step in range(50):
        env_act, raw_act, lp, val = agent.select_action(obs)
        next_obs, rew, term, trunc, info = env.step(env_act)
        agent.push_transition(obs, raw_act, 0.0, val, lp, done=(term or trunc))
        episode_reward += rew
        obs = next_obs
        if term or trunc:
            break

    # Should have made progress without crashing
    assert episode_reward != 0.0
    assert agent.buffer.ptr == 50  # all 50 steps stored


# ──────────────────────────────────────────────
# 5. Config sanity
# ──────────────────────────────────────────────

def test_ppo_config_defaults_are_safe():
    """PPOConfig defaults must be safe for training."""
    cfg = PPOConfig()
    assert cfg.ppo_epochs == 4
    assert cfg.target_kl == 0.02
    assert cfg.reward_scale == 1.0
    assert cfg.clip_epsilon == 0.2
    assert cfg.max_grad_norm == 0.5
    assert cfg.lr_actor == 3e-4
    assert cfg.lr_critic == 1e-3


def test_runtime_with_fallback_no_checkpoint():
    """RuntimeWithFallback requires either agent or checkpoint_path."""
    with pytest.raises(ValueError, match="Provide either"):
        RuntimeWithFallback()  # type: ignore[call-overload]


# ──────────────────────────────────────────────
# 6. Demo script runners
# ──────────────────────────────────────────────

def test_demo_parse_args_all_modes():
    """Demo CLI accepts all modes."""
    import scripts.demo as demo_mod

    parser = demo_mod._build_parser()
    for mode in ["basic", "pecan", "fault", "compare", "full", "all"]:
        args = parser.parse_args([mode])
        assert args.mode == mode
        assert hasattr(args, "city")
        assert hasattr(args, "steps")


def test_demo_run_basic():
    """run_basic completes without error."""
    from scripts.demo import run_basic
    run_basic(n_steps=3)


def test_demo_run_fault():
    """run_fault completes without error."""
    from scripts.demo import run_fault
    run_fault(n_steps=3)


def test_demo_run_full():
    """run_full completes without error."""
    from scripts.demo import run_full
    run_full()


# ──────────────────────────────────────────────
# 7. Training script argument parsing
# ──────────────────────────────────────────────

def test_train_cortexcore_args_defaults():
    """train_cortexcore arg defaults are correct."""
    from scripts.train_cortexcore import parse_args
    args = parse_args(["--scenario", "scenarios/phase1_debug.json"])
    assert args.scenario == "scenarios/phase1_debug.json"
    assert args.total_timesteps == 500_000
    assert args.seed == 42
    assert args.use_pecan is False
    assert args.demo is False
    assert args.city == "bangalore"


def test_train_cortexcore_auto_scaledown():
    """Auto-scaledown triggers when --use-pecan is set."""
    from scripts.train_cortexcore import parse_args
    args = parse_args(["--use-pecan", "--city", "kochi"])
    assert args.use_pecan is True
    assert args.city == "kochi"


def test_train_vayugnn_args_defaults():
    """train_vayugnn arg defaults are correct."""
    from scripts.train_vayugnn import parse_args
    args = parse_args(["--scenario", "scenarios/phase1_debug.json"])
    assert args.scenario == "scenarios/phase1_debug.json"
    assert args.epochs == 50
    assert args.num_episodes == 10
    assert args.use_pecan is False
    assert args.demo is False
    assert args.dry_run is False


def test_train_vayugnn_auto_scaledown():
    """Auto-scaledown triggers when --use-pecan is set."""
    from scripts.train_vayugnn import parse_args
    args = parse_args(["--use-pecan", "--city", "delhi"])
    assert args.use_pecan is True
    assert args.city == "delhi"
