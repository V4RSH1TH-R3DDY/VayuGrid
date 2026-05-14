#!/usr/bin/env python3
"""
Run a grid simulation and push results to the database so the frontend
displays live telemetry, transformer readings, and market data.

Usage:
    python scripts/seed_frontend.py
    python scripts/seed_frontend.py --scenario scenarios/phase1_demo.json
    python scripts/seed_frontend.py --scenario scenarios/phase1_debug.json --db-port 5433
"""

from __future__ import annotations

import argparse
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from psycopg.rows import dict_row

from simulator.config import load_simulator_config
from simulator.simulator import GridSimulator


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seed frontend with simulation data")
    p.add_argument("--scenario", default="scenarios/phase1_demo.json")
    p.add_argument("--db-host", default="localhost")
    p.add_argument("--db-port", type=int, default=5433)
    p.add_argument("--db-user", default="vayugrid")
    p.add_argument("--db-password", default="budugu123")
    p.add_argument("--db-name", default="vayugrid")
    p.add_argument("--max-rows", type=int, default=5000,
                    help="Max telemetry rows to insert (avoid overload)")
    return p.parse_args()


def _conn_str(args: argparse.Namespace) -> str:
    return (
        f"postgresql://{args.db_user}:{args.db_password}"
        f"@{args.db_host}:{args.db_port}/{args.db_name}"
    )


def main() -> None:
    args = _parse_args()
    scenario_path = Path(args.scenario)
    if not scenario_path.exists():
        print(f"Scenario file not found: {scenario_path}")
        return

    print(f"Loading scenario: {scenario_path}")
    config = load_simulator_config(str(scenario_path))
    sim = GridSimulator(config)
    print(f"Running simulation ({config.neighborhood.num_homes} homes)...")
    result = sim.run()

    node_df = result.node_timeseries
    trans_df = result.transformer_timeseries

    print(f"  Node timeseries:      {len(node_df)} rows")
    print(f"  Transformer readings: {len(trans_df)} rows")

    conn_str = _conn_str(args)
    conn = psycopg.connect(conn_str, autocommit=True)
    cur = conn.cursor()

    now = datetime.now(timezone.utc)

    # shift historical timestamps to recent so dashboard queries match
    sim_start = node_df["timestamp"].iloc[0]
    sim_end = node_df["timestamp"].iloc[-1]
    if isinstance(sim_start, datetime):
        sim_start_utc = sim_start.replace(tzinfo=timezone.utc) if sim_start.tzinfo is None else sim_start
        sim_end_utc = sim_end.replace(tzinfo=timezone.utc) if isinstance(sim_end, datetime) and sim_end.tzinfo is None else sim_end
    else:
        sim_start_utc = sim_end_utc = now
    duration = (sim_end_utc - sim_start_utc).total_seconds()
    offset_sec = max(0, (now - sim_end_utc).total_seconds())

    def _shift_ts(ts: datetime) -> datetime:
        if isinstance(ts, datetime):
            ts_u = ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts
        else:
            ts_u = now - timedelta(seconds=duration)
        if offset_sec > 3600:
            ts_u += timedelta(seconds=offset_sec - 300)
        return ts_u

    # ---------- node telemetry ----------
    telemetry_rows: list[tuple] = []
    for _, row in node_df.iterrows():
        ts_aware = _shift_ts(row["timestamp"])

        telemetry_rows.append((
            ts_aware,
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
        batch = telemetry_rows[:args.max_rows]
        cur.executemany(
            """
            INSERT INTO node_telemetry (
                ts, node_id, node_type, battery_soc_kwh, battery_power_kw,
                solar_output_kw, household_load_kw, ev_charge_kw, net_grid_kw, voltage_pu
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            batch,
        )
        print(f"  Inserted {len(batch)} telemetry rows")

    # ---------- transformer readings ----------
    trans_rows: list[tuple] = []
    for _, row in trans_df.iterrows():
        ts_aware = _shift_ts(row["timestamp"])
        trans_rows.append((
            ts_aware,
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

    if trans_rows:
        cur.executemany(
            """
            INSERT INTO transformer_readings (
                ts, transformer_id, feeder_total_kw, transformer_loading_pu,
                max_branch_loading_pu, hottest_spot_temp_c, aging_acceleration,
                grid_available, islanding_triggered, maintenance_mode
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            trans_rows,
        )
        print(f"  Inserted {len(trans_rows)} transformer rows")

    # ---------- seed a couple trade records ----------
    trade_rows: list[tuple] = []
    for _, row in node_df[::10].iterrows():
        ts_aware = _shift_ts(row["timestamp"])
        buyer = int(np.random.randint(1, config.neighborhood.num_homes + 1))
        seller = int(np.random.randint(1, config.neighborhood.num_homes + 1))
        if buyer == seller:
            seller = seller % config.neighborhood.num_homes + 1
        trade_rows.append((
            str(uuid.uuid4()),
            ts_aware,
            buyer,
            seller,
            round(abs(float(row.get("net_grid_kw", 1))) * 0.3, 3) or 0.5,
            round(np.random.uniform(3.0, 8.0), 2),
            "settled",
        ))

    if trade_rows:
        cur.executemany(
            """
            INSERT INTO trade_records (
                trade_id, ts, buyer_node_id, seller_node_id,
                quantity_kwh, cleared_price_inr_per_kwh, status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            trade_rows,
        )
        print(f"  Inserted {len(trade_rows)} trade records")

    cur.close()
    conn.close()
    print("Done. Refresh the frontend dashboard to see the data.")


if __name__ == "__main__":
    main()
