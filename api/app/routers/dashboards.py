from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status

from ..config import settings
from ..db import fetch_all, fetch_one
from ..rate_limit import limiter
from ..security import UserClaims, require_roles

router = APIRouter(tags=["dashboards"])


def _stress_level(voltage_pu: float | None, load_kw: float | None) -> tuple[str, float]:
    voltage_pu = voltage_pu or 1.0
    load_kw = load_kw or 0.0
    voltage_penalty = abs(1 - voltage_pu) * 100
    load_penalty = max(0.0, load_kw - 3.0) * 2
    score = voltage_penalty + load_penalty
    if score >= 20:
        level = "critical"
    elif score >= 12:
        level = "high"
    elif score >= 6:
        level = "medium"
    else:
        level = "low"
    return level, score


def _overload_probability(loading_pu: float | None) -> float:
    if loading_pu is None:
        return 0.0
    return 1 / (1 + math.exp(-12 * (loading_pu - 1.0)))


@router.get("/dashboard/operator/overview")
@limiter.limit("100/minute")
def operator_overview(
    request: Request, _: UserClaims = Depends(require_roles(["operator"]))
) -> dict:
    latest_nodes = fetch_all(
        """
        SELECT DISTINCT ON (node_id)
            node_id, node_type, battery_soc_kwh, battery_power_kw, solar_output_kw,
            household_load_kw, ev_charge_kw, net_grid_kw, voltage_pu, ts
        FROM node_telemetry
        ORDER BY node_id, ts DESC
        """
    )
    nodes = []
    for row in latest_nodes:
        level, score = _stress_level(row.get("voltage_pu"), row.get("household_load_kw"))
        nodes.append(
            {
                "node_id": row.get("node_id"),
                "node_type": row.get("node_type"),
                "battery_soc_kwh": row.get("battery_soc_kwh"),
                "solar_output_kw": row.get("solar_output_kw"),
                "household_load_kw": row.get("household_load_kw"),
                "net_grid_kw": row.get("net_grid_kw"),
                "voltage_pu": row.get("voltage_pu"),
                "stress_level": level,
                "stress_score": round(score, 2),
                "last_seen": row["ts"].isoformat() if row.get("ts") is not None else None,
            }
        )

    duck_rows = fetch_all(
        """
        SELECT date_trunc('hour', ts) AS bucket, SUM(net_grid_kw) AS net_grid_kw
        FROM node_telemetry
        WHERE ts >= now() - interval '48 hours'
        GROUP BY bucket
        ORDER BY bucket
        """
    )
    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(hours=24)
    prev_cutoff = now_utc - timedelta(hours=48)
    previous_map = {}
    for row in duck_rows:
        bucket = row["bucket"]
        if prev_cutoff <= bucket < cutoff:
            previous_map[bucket + timedelta(hours=24)] = row["net_grid_kw"]
    duck_curve = []
    for row in duck_rows:
        bucket = row["bucket"]
        if bucket < cutoff:
            continue
        forecast = previous_map.get(bucket, row["net_grid_kw"])
        duck_curve.append(
            {
                "ts": bucket.isoformat(),
                "actual_kw": row["net_grid_kw"],
                "forecast_kw": forecast,
            }
        )

    risk_rows = fetch_all(
        """
        SELECT ts, transformer_id, transformer_loading_pu, max_branch_loading_pu
        FROM transformer_readings
        WHERE ts >= now() - interval '30 minutes'
        ORDER BY ts
        """
    )
    risk_timeline = [
        {
            "ts": row["ts"].isoformat(),
            "transformer_id": row["transformer_id"],
            "loading_pu": row["transformer_loading_pu"],
            "max_branch_loading_pu": row["max_branch_loading_pu"],
            "overload_probability": round(_overload_probability(row["transformer_loading_pu"]), 4),
        }
        for row in risk_rows
    ]

    signals = fetch_all(
        """
        SELECT signal_id, ts, signal_type, severity, target_node_ids, reason
        FROM signal_history
        ORDER BY ts DESC
        LIMIT 100
        """
    )
    signal_history = [
        {
            **row,
            "ts": row["ts"].isoformat() if row.get("ts") else None,
        }
        for row in signals
    ]

    telemetry_rows = fetch_all(
        """
        SELECT solar_output_kw, household_load_kw, ev_charge_kw, battery_power_kw, net_grid_kw
        FROM node_telemetry
        WHERE ts >= now() - interval '24 hours'
        """
    )
    curtailment_sum = 0.0
    solar_sum = 0.0
    for row in telemetry_rows:
        solar = row.get("solar_output_kw") or 0.0
        load = row.get("household_load_kw") or 0.0
        ev = row.get("ev_charge_kw") or 0.0
        battery_power = row.get("battery_power_kw") or 0.0
        net_grid = row.get("net_grid_kw") or 0.0
        export_kw = max(0.0, -net_grid)
        battery_charge_kw = max(0.0, battery_power)
        used_kw = load + ev + export_kw + battery_charge_kw
        curtailment_sum += max(0.0, solar - used_kw)
        solar_sum += solar
    curtailment_pct = round(curtailment_sum / solar_sum, 4) if solar_sum > 0 else None

    peak_rows = fetch_all(
        """
        SELECT date_trunc('hour', ts) AS bucket, SUM(net_grid_kw) AS net_grid_kw
        FROM node_telemetry
        WHERE ts >= now() - interval '48 hours'
        GROUP BY bucket
        ORDER BY bucket
        """
    )
    today_peak = None
    prev_peak = None
    for row in peak_rows:
        bucket = row["bucket"]
        if bucket >= cutoff:
            today_peak = max(today_peak or row["net_grid_kw"], row["net_grid_kw"])
        elif bucket >= prev_cutoff:
            prev_peak = max(prev_peak or row["net_grid_kw"], row["net_grid_kw"])
    peak_reduction_pct = None
    if prev_peak and prev_peak > 0 and today_peak is not None:
        peak_reduction_pct = round(1 - (today_peak / prev_peak), 4)

    transformer_metrics = fetch_one(
        """
        SELECT AVG(aging_acceleration) AS avg_aging
        FROM transformer_readings
        WHERE ts >= now() - interval '1 hour'
        """
    )
    trade_metrics = fetch_one(
        """
        SELECT SUM(quantity_kwh) AS total_kwh,
               SUM(quantity_kwh * cleared_price_inr_per_kwh) AS total_value
        FROM trade_records
        WHERE ts >= now() - interval '24 hours'
        """
    )
    overload_events = fetch_one(
        """
        SELECT COUNT(*) AS count
        FROM transformer_readings
        WHERE ts >= now() - interval '24 hours' AND transformer_loading_pu >= 1.2
        """
    )

    return {
        "kpis": {
            "curtailment_pct": curtailment_pct,
            "peak_reduction_pct": peak_reduction_pct,
            "transformer_aging_rate": (
                round(float(transformer_metrics["avg_aging"]), 4)
                if transformer_metrics and transformer_metrics.get("avg_aging") is not None
                else None
            ),
            "p2p_volume_kwh": trade_metrics.get("total_kwh") if trade_metrics else 0,
            "p2p_value_inr": trade_metrics.get("total_value") if trade_metrics else 0,
            "overload_events": overload_events.get("count") if overload_events else 0,
        },
        "grid_health": nodes,
        "duck_curve": duck_curve,
        "risk_timeline": risk_timeline,
        "signal_history": signal_history,
    }


@router.get("/dashboard/homeowner/{node_id}/summary")
@limiter.limit("100/minute")
def homeowner_summary(
    request: Request,
    node_id: int,
    user: UserClaims = Depends(require_roles(["homeowner", "operator"])),
) -> dict:
    if user.role == "homeowner" and user.node_id and user.node_id != node_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

    telemetry_totals = (
        fetch_one(
            """
        SELECT
            SUM(COALESCE(solar_output_kw, 0)) AS solar_kw_sum,
            SUM(COALESCE(household_load_kw, 0)) AS load_kw_sum,
            SUM(CASE WHEN net_grid_kw > 0 THEN net_grid_kw ELSE 0 END) AS grid_import_kw_sum,
            SUM(CASE WHEN net_grid_kw < 0 THEN -net_grid_kw ELSE 0 END) AS grid_export_kw_sum,
            SUM(COALESCE(ev_charge_kw, 0)) AS ev_kw_sum
        FROM node_telemetry
        WHERE node_id = %s AND ts >= date_trunc('day', now())
        """,
            (node_id,),
        )
        or {}
    )
    def to_kwh(value: float | None) -> float:
        return round((value or 0) / 60, 3)

    latest = fetch_one(
        "SELECT * FROM node_telemetry WHERE node_id = %s ORDER BY ts DESC LIMIT 1",
        (node_id,),
    )
    metadata = latest.get("metadata") if latest else {}
    battery_capacity = float(metadata.get("battery_capacity_kwh", 13.5)) if metadata else 13.5
    ev_target = metadata.get("ev_target_kwh") if metadata else None
    ev_deadline = metadata.get("ev_deadline") if metadata else None
    ev_soc = metadata.get("ev_soc_kwh") if metadata else None
    ev_progress = None
    if ev_target and ev_soc is not None:
        ev_progress = round(min(1.0, ev_soc / ev_target), 3)

    trades = (
        fetch_one(
            """
        SELECT
            COALESCE(
                SUM(CASE WHEN seller_node_id = %s THEN quantity_kwh ELSE 0 END), 0
            ) AS sold_kwh,
            COALESCE(
                SUM(CASE WHEN buyer_node_id = %s THEN quantity_kwh ELSE 0 END), 0
            ) AS bought_kwh,
            COALESCE(
                SUM(CASE
                    WHEN seller_node_id = %s
                    THEN quantity_kwh * cleared_price_inr_per_kwh
                    ELSE 0
                END), 0
            ) AS revenue_inr,
            COALESCE(
                SUM(CASE
                    WHEN buyer_node_id = %s
                    THEN quantity_kwh * cleared_price_inr_per_kwh
                    ELSE 0
                END), 0
            ) AS cost_inr
        FROM trade_records
        WHERE ts >= date_trunc('day', now()) AND (buyer_node_id = %s OR seller_node_id = %s)
        """,
            (node_id, node_id, node_id, node_id, node_id, node_id),
        )
        or {}
    )

    market_rows = fetch_all(
        """
        SELECT to_timestamp(floor(extract(epoch from ts) / 600) * 600) AT TIME ZONE 'UTC' AS bucket,
               AVG(cleared_price_inr_per_kwh) AS avg_price
        FROM trade_records
        WHERE ts >= now() - interval '1 hour'
        GROUP BY bucket
        ORDER BY bucket
        """
    )
    live_market = [
        {"ts": row["bucket"].isoformat(), "avg_price": row["avg_price"]} for row in market_rows
    ]

    sold_kwh = trades.get("sold_kwh", 0)
    bought_kwh = trades.get("bought_kwh", 0)
    revenue_inr = trades.get("revenue_inr", 0)
    cost_inr = trades.get("cost_inr", 0)
    net_metering_revenue = sold_kwh * settings.net_metering_rate_inr_per_kwh

    return {
        "node_id": node_id,
        "today_energy": {
            "solar_generated_kwh": to_kwh(telemetry_totals.get("solar_kw_sum")),
            "home_consumed_kwh": to_kwh(telemetry_totals.get("load_kw_sum")),
            "grid_imported_kwh": to_kwh(telemetry_totals.get("grid_import_kw_sum")),
            "grid_exported_kwh": to_kwh(telemetry_totals.get("grid_export_kw_sum")),
            "ev_charging_kwh": to_kwh(telemetry_totals.get("ev_kw_sum")),
            "p2p_sold_kwh": sold_kwh,
            "p2p_bought_kwh": bought_kwh,
            "net_bill_inr": round(
                to_kwh(telemetry_totals.get("grid_import_kw_sum"))
                * settings.utility_rate_inr_per_kwh
                - revenue_inr
                + cost_inr,
                2,
            ),
        },
        "ev_status": {
            "current_kwh": ev_soc,
            "target_kwh": ev_target,
            "deadline": ev_deadline,
            "progress_pct": ev_progress,
            "current_charge_kw": latest.get("ev_charge_kw") if latest else None,
        },
        "battery_health": {
            "soc_kwh": latest.get("battery_soc_kwh") if latest else None,
            "capacity_kwh": battery_capacity,
            "health_pct": metadata.get("battery_health_pct") if metadata else None,
        },
        "earnings": {
            "p2p_revenue_inr": revenue_inr,
            "p2p_cost_inr": cost_inr,
            "net_metering_revenue_inr": round(net_metering_revenue, 2),
            "delta_vs_net_metering_inr": round(revenue_inr - net_metering_revenue, 2),
        },
        "live_market": live_market,
    }


@router.get("/dashboard/community/summary")
@limiter.limit("100/minute")
def community_summary(
    request: Request, _: UserClaims = Depends(require_roles(["community", "operator"]))
) -> dict:
    battery_totals = fetch_one(
        """
        SELECT SUM(battery_soc_kwh) AS total_battery_soc_kwh
        FROM (
            SELECT DISTINCT ON (node_id) node_id, battery_soc_kwh
            FROM node_telemetry
            ORDER BY node_id, ts DESC
        ) latest
        """
    ) or {"total_battery_soc_kwh": 0}
    load_avg = fetch_one(
        """
        SELECT AVG(total_load_kw) AS avg_load_kw
        FROM (
            SELECT
                to_timestamp(floor(extract(epoch from ts) / 300) * 300)
                    AT TIME ZONE 'UTC' AS bucket,
                   SUM(household_load_kw) AS total_load_kw
            FROM node_telemetry
            WHERE ts >= now() - interval '1 hour'
            GROUP BY bucket
        ) buckets
        """
    ) or {"avg_load_kw": 0}

    total_battery = battery_totals.get("total_battery_soc_kwh") or 0
    avg_load_kw = load_avg.get("avg_load_kw") or 0
    backup_hours = round(total_battery / avg_load_kw, 2) if avg_load_kw > 0 else None

    savings = (
        fetch_one(
            """
        SELECT
            SUM(CASE WHEN ts >= date_trunc('day', now())
                THEN (%s - cleared_price_inr_per_kwh) * quantity_kwh
                ELSE 0 END) AS today_savings,
            SUM(CASE WHEN ts >= date_trunc('month', now())
                THEN (%s - cleared_price_inr_per_kwh) * quantity_kwh
                ELSE 0 END) AS month_savings,
            SUM((%s - cleared_price_inr_per_kwh) * quantity_kwh) AS total_savings
        FROM trade_records
        """,
            (
                settings.utility_rate_inr_per_kwh,
                settings.utility_rate_inr_per_kwh,
                settings.utility_rate_inr_per_kwh,
            ),
        )
        or {}
    )

    fairness = fetch_all(
        """
        SELECT priority_tier, COUNT(*) AS count
        FROM critical_load_flags
        WHERE active = TRUE
        GROUP BY priority_tier
        ORDER BY count DESC
        """
    )

    return {
        "backup_hours": backup_hours,
        "community_savings": {
            "today_inr": round(savings.get("today_savings") or 0, 2),
            "month_inr": round(savings.get("month_savings") or 0, 2),
            "total_inr": round(savings.get("total_savings") or 0, 2),
        },
        "fairness_allocation": fairness,
    }
