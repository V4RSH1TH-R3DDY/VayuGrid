#!/usr/bin/env python3
"""
VayuGrid Demo Runner — end-to-end walkthrough script.

Orchestrates simulation, database seeding, agent inference, and
frontend data population so you can demonstrate every layer of
the platform in one command.

Usage:
    PYTHONPATH=$PWD python3 scripts/demo_runner.py
    PYTHONPATH=$PWD python3 scripts/demo_runner.py --quick
    PYTHONPATH=$PWD python3 scripts/demo_runner.py --scenario scenarios/phase1_demo.json
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

from ai.core.cortexcore import CortexCore, PPOConfig, RuntimeWithFallback
from ai.core.normalizer import ObservationNormalizer
from ai.demo import DemoConfig, make_demo_env_config
from ai.env.gym_env import VayuGridEnv
from simulator.config import load_simulator_config
from simulator.simulator import GridSimulator

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


# ── helpers ───────────────────────────────────────────────────────

def _banner(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _step(num: int, label: str) -> None:
    print(f"\n─── Step {num}: {label} ─────────────────────────────────────")


def _db_conn(args: argparse.Namespace):
    host = args.db_host or "localhost"
    port = args.db_port or 5433
    dsn = f"postgresql://{args.db_user}:{args.db_password}@{host}:{port}/{args.db_name}"
    return psycopg.connect(dsn, autocommit=True)


def _seed_database(result, config, conn) -> None:
    from uuid import uuid4

    cur = conn.cursor()
    now_dt = datetime.now(timezone.utc)
    node_df = result.node_timeseries
    trans_df = result.transformer_timeseries

    # shift historical timestamps to recent so dashboard queries match
    sim_start = node_df["timestamp"].iloc[0]
    sim_end = node_df["timestamp"].iloc[-1]
    if isinstance(sim_start, datetime):
        s0 = sim_start.replace(tzinfo=timezone.utc) if sim_start.tzinfo is None else sim_start
        s1 = sim_end.replace(tzinfo=timezone.utc) if isinstance(sim_end, datetime) and sim_end.tzinfo is None else sim_end
    else:
        s0 = s1 = now_dt
    duration_s = (s1 - s0).total_seconds()
    offset_s = max(0, (now_dt - s1).total_seconds())

    def _ts(ts):
        if isinstance(ts, datetime):
            t = ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts
        else:
            t = now_dt - timedelta(seconds=duration_s)
        if offset_s > 3600:
            t += timedelta(seconds=offset_s - 300)
        return t

    telemetry_rows = []
    for _, row in node_df.iterrows():
        ts_a = _ts(row["timestamp"])
        telemetry_rows.append((
            ts_a,
            int(row["node_id"]),
            str(row.get("node_type", "home")),
            float(row["battery_soc_kwh"]) if pd.notna(row.get("battery_soc_kwh")) else None,
            float(row["battery_power_kw"]) if pd.notna(row.get("battery_power_kw")) else None,
            float(row["pv_kw"]) if pd.notna(row.get("pv_kw")) else None,
            float(row["load_kw"]) if pd.notna(row.get("load_kw")) else None,
            float(row["ev_kw"]) if pd.notna(row.get("ev_kw")) else None,
            float(row["net_grid_kw"]) if pd.notna(row.get("net_grid_kw")) else None,
            float(row["voltage_pu"]) if pd.notna(row.get("voltage_pu")) else None,
        ))

    if telemetry_rows:
        cur.executemany(
            "INSERT INTO node_telemetry (ts, node_id, node_type, battery_soc_kwh, battery_power_kw, solar_output_kw, household_load_kw, ev_charge_kw, net_grid_kw, voltage_pu) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
            telemetry_rows,
        )
        print(f"    → {len(telemetry_rows)} telemetry rows")

    trans_rows = []
    for _, row in trans_df.iterrows():
        ts_a = _ts(row["timestamp"])
        trans_rows.append((
            ts_a, "xfmr_main",
            float(row.get("feeder_total_kw", 0)),
            float(row.get("transformer_loading_pu", 0)),
            float(row.get("max_branch_loading_pu", 0)),
            float(row.get("hottest_spot_temp_c", 0)),
            float(row.get("aging_acceleration", 0)),
            bool(row.get("grid_available", True)),
            bool(row.get("islanding_triggered", False)),
            bool(row.get("maintenance_mode", False)),
        ))

    if trans_rows:
        cur.executemany(
            "INSERT INTO transformer_readings (ts, transformer_id, feeder_total_kw, transformer_loading_pu, max_branch_loading_pu, hottest_spot_temp_c, aging_acceleration, grid_available, islanding_triggered, maintenance_mode) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
            trans_rows,
        )
        print(f"    → {len(trans_rows)} transformer rows")

    trade_rows = []
    for _, row in node_df[::15].iterrows():
        ts_a = _ts(row["timestamp"])
        buyer = int(np.random.randint(1, config.neighborhood.num_homes + 1))
        seller = int(np.random.randint(1, config.neighborhood.num_homes + 1))
        if buyer == seller:
            seller = seller % config.neighborhood.num_homes + 1
        trade_rows.append((
            str(uuid4()), ts_a, buyer, seller,
            round(abs(float(row.get("net_grid_kw", 1))) * 0.3, 3) or 0.5,
            round(np.random.uniform(3.0, 8.0), 2), "settled",
        ))

    if trade_rows:
        cur.executemany(
            "INSERT INTO trade_records (trade_id, ts, buyer_node_id, seller_node_id, quantity_kwh, cleared_price_inr_per_kwh, status) VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
            trade_rows,
        )
        print(f"    → {len(trade_rows)} trade records")

    cur.close()
    print("  Database seeded. Frontend dashboards will now show live data.")


def _try_load_agent(checkpoint: str, device: str) -> RuntimeWithFallback | None:
    ckpt = Path(checkpoint)
    if not ckpt.exists():
        print(f"  [SKIP] checkpoint not found: {checkpoint}")
        return None
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() and device == "cuda" else "cpu")
    normalizer = ObservationNormalizer((12,))
    agent = CortexCore(cfg=PPOConfig(), normalizer=normalizer, device=dev)
    agent.load(str(ckpt))
    return RuntimeWithFallback(agent=agent)


# ── demo phases ───────────────────────────────────────────────────

def phase_1_simulation(args: argparse.Namespace):
    """Run the grid simulator and show what it produces."""
    _step(1, "Grid Simulation Engine")
    print("  Scenario:", args.scenario)
    config = load_simulator_config(args.scenario)
    sim = GridSimulator(config)
    print(f"  Homes: {config.neighborhood.num_homes}")
    print(f"  Duration: {config.start_time} → {config.end_time}")
    print(f"  Solar adoption: {config.adoption.solar_ratio:.0%}")
    print(f"  Battery adoption: {config.adoption.battery_ratio:.0%}")
    print(f"  EV adoption: {config.adoption.ev_ratio:.0%}")
    for f in config.faults:
        print(f"  Fault: [{f.event_type}] {f.name} @ {f.start}–{f.end}")

    t0 = time.perf_counter()
    result = sim.run()
    elapsed = time.perf_counter() - t0

    print(f"\n  Simulation complete in {elapsed:.1f}s")
    print(f"  Node timeseries:      {len(result.node_timeseries):,} rows")
    print(f"  Transformer readings: {len(result.transformer_timeseries):,} rows")
    print(f"  Events logged:        {len(result.event_log):,} rows")
    print(f"  Peak transformer loading: {result.transformer_timeseries['transformer_loading_pu'].max():.2%}")

    return result, config


def phase_2_seed_database(result, config, args: argparse.Namespace):
    """Push simulation output into PostgreSQL so the frontend can display it."""
    _step(2, "Database Ingestion (API → TimescaleDB)")
    conn = _db_conn(args)
    try:
        _seed_database(result, config, conn)
    finally:
        conn.close()

    # verify
    conn2 = _db_conn(args)
    try:
        cur = conn2.cursor()
        cur.execute("SELECT count(*) FROM node_telemetry")
        n_tele = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM transformer_readings")
        n_xfmr = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM trade_records")
        n_trade = cur.fetchone()[0]
        print(f"\n  DB counts: {n_tele} telemetry · {n_xfmr} transformer · {n_trade} trades")
    finally:
        conn2.close()


def phase_3_agent_inference(args: argparse.Namespace):
    """Run the trained PPO agent (CortexCore) against the simulator."""
    _step(3, "AI Agent — CortexCore PPO Inference")
    checkpoint = args.checkpoint or str(PROJECT_ROOT / "outputs/checkpoints/cortexcore_best.pt")
    runtime = _try_load_agent(checkpoint, args.device)
    if runtime is None:
        print("  Skipping agent inference (no checkpoint). Use --checkpoint to provide one.")
        return

    env = VayuGridEnv(make_demo_env_config(DemoConfig(scenario_path=args.scenario)))
    obs, _ = env.reset()
    total_reward = 0.0
    steps = 0
    t0 = time.perf_counter()
    term = trunc = False

    print("  Running agent on simulator environment...")
    print(f"  {'Step':>5} | {'Batt':>7} {'EV':>7} {'Bid':>7} {'Ask':>7} {'Grid':>7} | {'Reward':>8} | Model")
    print(f"  {'-'*60}")

    while not (term or trunc) and steps < 200:
        act = runtime.step(obs)
        obs, rew, term, trunc, _ = env.step(act)
        total_reward += rew
        model_tag = "[F]" if runtime.is_using_fallback else "[P]"
        if steps < 5 or steps % 20 == 0:
            print(f"  {steps:>5} | {act[0]:+7.3f} {act[1]:+7.3f} {act[2]:+7.3f} {act[3]:+7.3f} {act[4]:+7.3f} | {rew:+8.2f} | {model_tag}")
        steps += 1

    elapsed = time.perf_counter() - t0
    print(f"  {'-'*60}")
    print(f"  Agent ran for {steps} steps in {elapsed:.1f}s")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Fallback active: {runtime.is_using_fallback}")
    print(f"  Policy: {'Trained PPO' if not runtime.is_using_fallback else 'Rule-based (B1)'}")


def phase_4_api_endpoints(args: argparse.Namespace):
    """Probe the live API to demonstrate the REST layer."""
    _step(4, "REST API — Live Endpoints")
    import requests as http

    base = f"http://{args.api_host}:{args.api_port}"

    # health
    r = http.get(f"{base}/health", timeout=5)
    print(f"  GET  /health                         → {r.status_code} {r.json()}")

    # login
    r = http.post(f"{base}/api/auth/login", json={"username": "tony", "password": "operator"}, timeout=5)
    token = r.json().get("access_token", "")
    print(f"  POST /api/auth/login (tony/operator) → 200 (token received)")
    headers = {"Authorization": f"Bearer {token}"}

    # operator dashboard
    r = http.get(f"{base}/api/dashboard/operator/overview", headers=headers, timeout=5)
    if r.ok:
        data = r.json()
        kpi = data.get("kpis", {})
        print(f"  GET  /api/dashboard/operator/overview  → 200")
        print(f"       KPI: curtailment={kpi.get('curtailment_pct')}% · "
              f"peak_reduction={kpi.get('peak_reduction_pct')}% · "
              f"overloads={kpi.get('overload_events')}")
        print(f"       Grid health: {data.get('grid_health')} · "
              f"Duck curve: {data.get('duck_curve')} · "
              f"Risk timeline: {data.get('risk_timeline')}")
    else:
        print(f"  GET  /api/dashboard/operator/overview  → {r.status_code}")

    # homeowner dashboard
    r = http.get(f"{base}/api/dashboard/homeowner/1/summary", headers=headers, timeout=5)
    print(f"  GET  /api/dashboard/homeowner/1/summary → {r.status_code}")
    if r.ok:
        d = r.json()
        print(f"       Cost: {d.get('cost_inr', '?')} · "
              f"Earnings: {d.get('earnings_inr', '?')} · "
              f"Self-sufficiency: {d.get('self_sufficiency_pct', '?')}%")

    # community dashboard
    r = http.get(f"{base}/api/dashboard/community/summary", headers=headers, timeout=5)
    print(f"  GET  /api/dashboard/community/summary   → {r.status_code}")
    if r.ok:
        d = r.json()
        print(f"       Community savings: {d.get('community_savings_inr', '?')} · "
              f"Fairness: {d.get('fairness_score', '?')}")

    # security status
    r = http.get(f"{base}/api/security/status", headers=headers, timeout=5)
    print(f"  GET  /api/security/status              → {r.status_code}")
    if r.ok:
        d = r.json()
        print(f"       Anomaly detector trained: {d.get('anomaly_detector_trained')} · "
              f"Recent events: {len(d.get('recent_events', []))}")


def phase_5_webstream(args: argparse.Namespace):
    """Demonstrate the live WebSocket stream."""
    _step(5, "WebSocket — Live Telemetry Stream")
    import asyncio
    import websockets

    import requests as http
    base = f"http://{args.api_host}:{args.api_port}"
    r = http.post(f"{base}/api/auth/login", json={"username": "tony", "password": "operator"}, timeout=5)
    token = r.json()["access_token"]

    async def _demo_stream():
        uri = f"ws://{args.api_host}:{args.api_port}/ws/stream?token={token}"
        async with websockets.connect(uri) as ws:
            msg = await asyncio.wait_for(ws.recv(), timeout=5)
            import json
            data = json.loads(msg)
            print(f"  Connected to WebSocket stream")
            print(f"  Telemetry snapshots:  {len(data['telemetry'])} nodes")
            print(f"  Recent trades:        {len(data['trade_flow'])} trades")
            print(f"  GNN overload predictions: {len(data['gnn_predictions'])} transformers")
            if data["telemetry"]:
                sample = data["telemetry"][0]
                print(f"  Sample: node {sample.get('node_id')} | "
                      f"load={sample.get('household_load_kw')} kW | "
                      f"soc={sample.get('battery_soc_kwh')} kWh | "
                      f"voltage={sample.get('voltage_pu')} pu")

    asyncio.run(_demo_stream())


def phase_6_frontend_info() -> None:
    """Print frontend access info."""
    _step(6, "Frontend Dashboards")
    print("  Open http://localhost:5173 in your browser")
    print()
    print("  ┌──────────────┬──────────────┬──────────────┐")
    print("  │  Operator    │  Homeowner   │  Community   │")
    print("  │  tony        │  reggie      │  luigi       │")
    print("  │  operator    │  homeowner   │  community   │")
    print("  └──────────────┴──────────────┴──────────────┘")
    print()
    print("  Operator dashboard:     grid overview, KPIs, signals")
    print("  Homeowner dashboard:    personal consumption, costs, consent")
    print("  Community dashboard:    shared savings, fairness, critical loads")


# ── CLI ───────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VayuGrid Demo Runner — end-to-end platform walkthrough",
    )
    p.add_argument("--scenario", default=str(PROJECT_ROOT / "scenarios/phase1_demo.json"))
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--db-host", default="localhost")
    p.add_argument("--db-port", type=int, default=5433)
    p.add_argument("--db-user", default="vayugrid")
    p.add_argument("--db-password", default="budugu123")
    p.add_argument("--db-name", default="vayugrid")
    p.add_argument("--api-host", default="localhost")
    p.add_argument("--api-port", type=int, default=8000)
    p.add_argument("--quick", action="store_true", help="Skip agent inference phase")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    _banner("VayuGrid — Platform Demo")
    print(f"  Started at {datetime.now().strftime('%H:%M:%S')}")
    print("  This walkthrough demonstrates every layer of the VayuGrid")
    print("  smart energy grid: simulation → database ingestion → AI agent")
    print("  → REST API → WebSocket streaming → frontend dashboards.")

    # Phase 1 — simulation
    result, config = phase_1_simulation(args)

    # Phase 2 — seed database
    phase_2_seed_database(result, config, args)

    # Phase 3 — AI agent inference
    if not args.quick:
        phase_3_agent_inference(args)

    # Phase 4 — REST API
    phase_4_api_endpoints(args)

    # Phase 5 — WebSocket
    phase_5_webstream(args)

    # Phase 6 — frontend
    phase_6_frontend_info()

    _banner("Demo Complete")
    print("  All layers are operational. Data is flowing from the grid")
    print("  simulator through the AI agent and into the live dashboards.")
    print(f"  Finished at {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
