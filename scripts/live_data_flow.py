#!/usr/bin/env python3
"""
VayuGrid Live Data Flow — inserts simulation data incrementally so you
can watch the dashboard update in real time.

Opens a browser tab to http://localhost:5173, then feeds data into
PostgreSQL one time-chunk at a time with configurable delays.
After each batch it prints the current dashboard KPIs so you can
see the numbers change live.

Usage:
    PYTHONPATH=$PWD /home/varshith/VayuGrid/.venv/bin/python3 scripts/live_data_flow.py
    PYTHONPATH=$PWD /home/varshith/VayuGrid/.venv/bin/python3 scripts/live_data_flow.py --delay 4 --batch-size 5
    PYTHONPATH=$PWD /home/varshith/VayuGrid/.venv/bin/python3 scripts/live_data_flow.py --scenario scenarios/phase1_default.json
"""

from __future__ import annotations

import argparse
import time
import webbrowser
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
import requests as http

from simulator.config import load_simulator_config
from simulator.simulator import GridSimulator

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# ── SQL ─────────────────────────────────────────────────────────────

INSERT_TELEMETRY = """
    INSERT INTO node_telemetry (
        ts, node_id, node_type, battery_soc_kwh, battery_power_kw,
        solar_output_kw, household_load_kw, ev_charge_kw, net_grid_kw, voltage_pu
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT DO NOTHING
"""

INSERT_TRANSFORMER = """
    INSERT INTO transformer_readings (
        ts, transformer_id, feeder_total_kw, transformer_loading_pu,
        max_branch_loading_pu, hottest_spot_temp_c, aging_acceleration,
        grid_available, islanding_triggered, maintenance_mode
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT DO NOTHING
"""

INSERT_TRADE = """
    INSERT INTO trade_records (
        trade_id, ts, buyer_node_id, seller_node_id,
        quantity_kwh, cleared_price_inr_per_kwh, status
    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT DO NOTHING
"""


# ── helpers ─────────────────────────────────────────────────────────

def _shift_ts(ts_val: pd.Timestamp | datetime, sim_start: datetime, sim_end: datetime,
              now_dt: datetime) -> datetime:
    if isinstance(ts_val, pd.Timestamp):
        t = ts_val.to_pydatetime()
    else:
        t = ts_val
    t = t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t
    duration_s = (sim_end - sim_start).total_seconds()
    offset_s = max(0, (now_dt - sim_end).total_seconds())
    if offset_s > 3600:
        t += timedelta(seconds=offset_s - 300)
    return t


def _load_api(base: str, path: str, token: str | None = None) -> dict | None:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        r = http.get(f"{base}{path}", headers=headers, timeout=5)
        return r.json() if r.ok else None
    except Exception:
        return None


# ── main ────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="VayuGrid Live Data Flow")
    p.add_argument("--scenario", default=str(PROJECT_ROOT / "scenarios/phase1_demo.json"))
    p.add_argument("--delay", type=float, default=3.0, help="Seconds between batches (default 3)")
    p.add_argument("--batch-size", type=int, default=3, help="Simulated timesteps per batch (default 3)")
    p.add_argument("--api-host", default="localhost")
    p.add_argument("--api-port", type=int, default=8000)
    p.add_argument("--db-host", default="localhost")
    p.add_argument("--db-port", type=int, default=5433)
    p.add_argument("--db-user", default="vayugrid")
    p.add_argument("--db-password", default="budugu123")
    p.add_argument("--db-name", default="vayugrid")
    p.add_argument("--no-clear", action="store_true", help="Don't truncate existing data before starting")
    p.add_argument("--no-browser", action="store_true", help="Don't auto-open the browser")
    args = p.parse_args()

    base = f"http://{args.api_host}:{args.api_port}"

    print("=" * 72)
    print("  VayuGrid — Live Data Flow")
    print("=" * 72)
    print(f"  Each batch inserts {args.batch_size} simulated minute(s),")
    print(f"  then pauses {args.delay}s so the frontend dashboard")
    print("  can re-render with fresh data.")
    print()

    # ── 1. Run simulation ──────────────────────────────────────────
    print("  [1/5] Running simulation …", end=" ", flush=True)
    config = load_simulator_config(args.scenario)
    sim = GridSimulator(config)
    result = sim.run()
    node_df = result.node_timeseries.copy()
    trans_df = result.transformer_timeseries.copy()
    print(f"done — {len(node_df)} telemetry rows, {len(trans_df)} transformer rows")

    # ── 2. Shift timestamps to recent ──────────────────────────────
    print("  [2/5] Shifting timestamps to recent …", end=" ", flush=True)
    now_dt = datetime.now(timezone.utc)
    sim_start = node_df["timestamp"].iloc[0]
    sim_end = node_df["timestamp"].iloc[-1]
    if isinstance(sim_start, pd.Timestamp):
        s0 = sim_start.to_pydatetime().replace(tzinfo=timezone.utc)
        s1 = sim_end.to_pydatetime().replace(tzinfo=timezone.utc)
    else:
        s0 = s1 = now_dt
    if s1.tzinfo is None:
        s1 = s1.replace(tzinfo=timezone.utc)
    if s0.tzinfo is None:
        s0 = s0.replace(tzinfo=timezone.utc)

    node_df["shifted_ts"] = node_df["timestamp"].apply(
        lambda v: _shift_ts(v, s0, s1, now_dt)
    )
    trans_df["shifted_ts"] = trans_df["timestamp"].apply(
        lambda v: _shift_ts(v, s0, s1, now_dt)
    )

    # sort by shifted timestamp
    node_df = node_df.sort_values("shifted_ts").reset_index(drop=True)
    trans_df = trans_df.sort_values("shifted_ts").reset_index(drop=True)
    print("done")

    # ── 3. Pre-build trade records ─────────────────────────────────
    print("  [3/5] Building trade records …", end=" ", flush=True)
    trade_rows_raw = []
    num_homes = config.neighborhood.num_homes
    for _, row in node_df[::15].iterrows():
        buyer = int(np.random.randint(1, num_homes + 1))
        seller = int(np.random.randint(1, num_homes + 1))
        if buyer == seller:
            seller = seller % num_homes + 1
        ts_val = row["shifted_ts"]
        net = row.get("net_grid_kw")
        qty = round(abs(float(net)) * 0.3, 3) if pd.notna(net) and net is not None else 0.5
        if qty < 0.01:
            qty = 0.5
        trade_rows_raw.append((
            ts_val, buyer, seller, qty,
            round(np.random.uniform(3.0, 8.0), 2),
        ))
    trade_df = pd.DataFrame(trade_rows_raw, columns=["shifted_ts", "buyer", "seller", "qty", "price"])
    print(f"done — {len(trade_df)} trades")

    # ── 4. Connect to DB & clear ───────────────────────────────────
    print("  [4/5] Connecting to database …", end=" ", flush=True)
    dsn = f"postgresql://{args.db_user}:{args.db_password}@{args.db_host}:{args.db_port}/{args.db_name}"
    conn = psycopg.connect(dsn, autocommit=True)
    cur = conn.cursor()
    if not args.no_clear:
        for table in ("node_telemetry", "transformer_readings", "trade_records", "signal_history"):
            cur.execute(f"TRUNCATE {table} RESTART IDENTITY CASCADE")
        print("cleared + ", end="")
    print("connected")

    # ── 5. Login to get token ──────────────────────────────────────
    token = None
    try:
        r = http.post(f"{base}/api/auth/login", json={"username": "tony", "password": "operator"}, timeout=5)
        if r.ok:
            token = r.json()["access_token"]
    except Exception:
        pass

    # ── 6. Open browser ────────────────────────────────────────────
    if not args.no_browser:
        webbrowser.open("http://localhost:5173")
        print("  [5/5] Browser opened to http://localhost:5173")
        time.sleep(1)

    # ── 7. Insert in batches ───────────────────────────────────────
    print()
    print(f"  Starting live data flow — {args.batch_size} min(s) per batch, {args.delay}s gap")
    print(f"  {'Batch':>6} {'Tele':>5} {'Xfmr':>5} {'Trades':>5} | "
          f"{'Curtail':>9} {'Overloads':>9} {'RiskPts':>7} {'P2PVol':>7} | "
          f"{'Time range'}")
    print(f"  {'-'*65}")

    unique_ts = sorted(node_df["shifted_ts"].unique())
    total_batches = (len(unique_ts) + args.batch_size - 1) // args.batch_size

    for bi in range(0, len(unique_ts), args.batch_size):
        batch_ts = set(unique_ts[bi: bi + args.batch_size])

        # filter rows for this batch
        n_batch = node_df[node_df["shifted_ts"].isin(batch_ts)]
        t_batch = trans_df[trans_df["shifted_ts"].isin(batch_ts)]
        tr_batch = trade_df[trade_df["shifted_ts"].isin(batch_ts)]

        tele_rows = []
        for _, row in n_batch.iterrows():
            tele_rows.append((
                row["shifted_ts"],
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

        xfmr_rows = []
        for _, row in t_batch.iterrows():
            xfmr_rows.append((
                row["shifted_ts"],
                "xfmr_main",
                float(row.get("feeder_total_kw", 0)),
                float(row.get("transformer_loading_pu", 0)),
                float(row.get("max_branch_loading_pu", 0)),
                float(row.get("hottest_spot_temp_c", 0)),
                float(row.get("aging_acceleration", 0)),
                bool(row.get("grid_available", True)),
                bool(row.get("islanding_triggered", False)),
                bool(row.get("maintenance_mode", False)),
            ))

        trade_rows_sql = []
        for _, row in tr_batch.iterrows():
            from uuid import uuid4
            trade_rows_sql.append((
                str(uuid4()),
                row["shifted_ts"],
                int(row["buyer"]),
                int(row["seller"]),
                float(row["qty"]),
                float(row["price"]),
                "settled",
            ))

        # execute inserts
        if tele_rows:
            cur.executemany(INSERT_TELEMETRY, tele_rows)
        if xfmr_rows:
            cur.executemany(INSERT_TRANSFORMER, xfmr_rows)
        if trade_rows_sql:
            cur.executemany(INSERT_TRADE, trade_rows_sql)

        # fetch current KPI from API
        kpi_strs = ["—", "—", "—", "—"]
        if token:
            data = _load_api(base, "/api/dashboard/operator/overview", token)
            if data:
                k = data.get("kpis", {})
                curtail = k.get("curtailment_pct")
                overloads = k.get("overload_events")
                risk_pts = len(data.get("risk_timeline", []))
                p2p = k.get("p2p_volume_kwh")
                kpi_strs = [
                    f"{curtail:.2f}%" if curtail is not None else "—",
                    str(overloads) if overloads is not None else "—",
                    str(risk_pts) if risk_pts else "0",
                    f"{p2p:.2f}" if p2p is not None else "—",
                ]

        batch_num = bi // args.batch_size + 1
        t_min = n_batch["shifted_ts"].min().strftime("%H:%M:%S") if len(n_batch) else "—"
        t_max = n_batch["shifted_ts"].max().strftime("%H:%M:%S") if len(n_batch) else "—"
        time_str = f"{t_min} – {t_max}" if t_min != t_max else t_min
        print(f"  {batch_num:>4}/{total_batches} {len(tele_rows):>5} {len(xfmr_rows):>5} "
              f"{len(trade_rows_sql):>5} | {kpi_strs[0]:>9} {kpi_strs[1]:>9} "
              f"{kpi_strs[2]:>7} {kpi_strs[3]:>7} | {time_str}")

        if bi + args.batch_size < len(unique_ts):
            time.sleep(args.delay)

    # ── 8. Final summary ────────────────────────────────────────────
    cur.execute("SELECT count(*) FROM node_telemetry")
    final_tele = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM transformer_readings")
    final_xfmr = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM trade_records")
    final_trades = cur.fetchone()[0]

    print(f"  {'-'*65}")
    print()
    print("  All data inserted. Final dashboard KPIs:")
    if token:
        data = _load_api(base, "/api/dashboard/operator/overview", token)
        if data:
            k = data.get("kpis", {})
            print(f"    Curtailment:    {k.get('curtailment_pct')}%")
            print(f"    Overload events: {k.get('overload_events')}")
            print(f"    Risk points:    {len(data.get('risk_timeline', []))}")
            print(f"    P2P volume:     {data.get('p2p_volume_kwh')} kWh")

    print()
    print(f"  DB: {final_tele} telemetry · {final_xfmr} transformer · {final_trades} trades")

    # homeowner & community dashboards (login with correct roles)
    try:
        rt = http.post(f"{base}/api/auth/login", json={"username": "reggie", "password": "homeowner"}, timeout=5)
        h_token = rt.json().get("access_token", "") if rt.ok else ""
        if h_token:
            h = _load_api(base, "/api/dashboard/homeowner/1/summary", h_token)
            if h:
                print(f"  Homeowner node 1 — cost: ₹{h.get('cost_inr','?')} · "
                      f"earnings: ₹{h.get('earnings_inr','?')} · "
                      f"self-sufficiency: {h.get('self_sufficiency_pct','?')}%")
    except Exception:
        pass

    try:
        rc = http.post(f"{base}/api/auth/login", json={"username": "luigi", "password": "community"}, timeout=5)
        c_token = rc.json().get("access_token", "") if rc.ok else ""
        if c_token:
            c = _load_api(base, "/api/dashboard/community/summary", c_token)
            if c:
                print(f"  Community — savings: ₹{c.get('community_savings_inr','?')} · "
                      f"fairness: {c.get('fairness_score','?')}")
    except Exception:
        pass

    print()
    print("  Dashboard: http://localhost:5173")
    print("=" * 72)

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
