from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from psycopg.types.json import Json

from ..config import settings
from ..db import execute_many, fetch_one
from ..rate_limit import limiter
from ..schemas import TelemetryIn, TradeRecordIn, TransformerReadingIn
from ..security import hash_api_key

router = APIRouter(tags=["nodes"])


def get_node_id_from_api_key(
    x_api_key: str | None = Header(default=None, alias="X-Api-Key"),
) -> int:
    if not x_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing API key")
    key_hash = hash_api_key(x_api_key)
    record = fetch_one(
        "SELECT node_id FROM node_api_keys WHERE key_hash = %s AND active = TRUE",
        (key_hash,),
    )
    if not record:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return int(record["node_id"])


@router.post("/nodes/telemetry")
@limiter.limit("1000/minute")
def ingest_telemetry(
    request: Request,
    payload: List[TelemetryIn],
    node_id: int = Depends(get_node_id_from_api_key),
) -> dict:
    _now = datetime.now(timezone.utc)
    for item in payload:
        _ts = item.ts.replace(tzinfo=timezone.utc) if item.ts.tzinfo is None else item.ts
        _age = (_now - _ts).total_seconds()
        if _age > settings.telemetry_replay_window_seconds or _age < -60:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Telemetry timestamp out of acceptable window for node {item.node_id}",
            )

    rows = []
    for item in payload:
        if item.node_id != node_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Node ID mismatch for API key",
            )
        # --- data precision anonymization ---
        if item.battery_soc_kwh is not None:
            item.battery_soc_kwh = round(item.battery_soc_kwh, 2)
        if item.household_load_kw is not None:
            # bucket to nearest 0.5 kW
            item.household_load_kw = round(round(item.household_load_kw / 0.5) * 0.5, 2)
        if item.solar_output_kw is not None:
            item.solar_output_kw = round(item.solar_output_kw, 3)
        rows.append(
            (
                item.ts,
                item.node_id,
                item.node_type,
                item.battery_soc_kwh,
                item.battery_power_kw,
                item.solar_output_kw,
                item.household_load_kw,
                item.ev_charge_kw,
                item.net_grid_kw,
                item.voltage_pu,
                Json(item.metadata or {}),
            )
        )

    execute_many(
        """
        INSERT INTO node_telemetry (
            ts, node_id, node_type, battery_soc_kwh, battery_power_kw,
            solar_output_kw, household_load_kw, ev_charge_kw, net_grid_kw,
            voltage_pu, metadata
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        """,
        rows,
    )

    # --- anomaly detection ---
    try:
        from ..anomaly import detector
        from ..breach import notifier

        for item in payload:
            reading = {
                "battery_soc_kwh": item.battery_soc_kwh,
                "solar_output_kw": item.solar_output_kw,
                "household_load_kw": item.household_load_kw,
                "ev_charge_kw": item.ev_charge_kw,
                "net_grid_kw": item.net_grid_kw,
                "voltage_pu": item.voltage_pu,
            }
            is_bad, score = detector.is_anomalous(reading, node_id=item.node_id)
            if is_bad:
                notifier.record_event(
                    "anomaly",
                    min(abs(score) * 2, 1.0),
                    f"Anomalous telemetry from node {item.node_id} (score={score:.4f})",
                    node_id=item.node_id,
                    metadata={"score": score, "features": reading},
                )
    except Exception as _exc:
        import logging as _log

        _log.getLogger(__name__).warning("Anomaly detection error (non-fatal): %s", _exc)

    return {"inserted": len(rows)}


@router.post("/nodes/transformer-readings")
@limiter.limit("1000/minute")
def ingest_transformer_readings(
    request: Request,
    payload: List[TransformerReadingIn],
    _: int = Depends(get_node_id_from_api_key),
) -> dict:
    rows = [
        (
            item.ts,
            item.transformer_id,
            item.feeder_total_kw,
            item.transformer_loading_pu,
            item.max_branch_loading_pu,
            item.hottest_spot_temp_c,
            item.aging_acceleration,
            item.grid_available,
            item.islanding_triggered,
            item.maintenance_mode,
            Json(item.metadata or {}),
        )
        for item in payload
    ]
    execute_many(
        """
        INSERT INTO transformer_readings (
            ts, transformer_id, feeder_total_kw, transformer_loading_pu,
            max_branch_loading_pu, hottest_spot_temp_c, aging_acceleration,
            grid_available, islanding_triggered, maintenance_mode, metadata
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        """,
        rows,
    )
    return {"inserted": len(rows)}


@router.post("/nodes/trades")
@limiter.limit("1000/minute")
def ingest_trades(
    request: Request,
    payload: List[TradeRecordIn],
    _: int = Depends(get_node_id_from_api_key),
) -> dict:
    rows = [
        (
            item.trade_id,
            item.ts,
            item.buyer_node_id,
            item.seller_node_id,
            item.quantity_kwh,
            item.cleared_price_inr_per_kwh,
            item.status,
            Json(item.metadata or {}),
        )
        for item in payload
    ]
    execute_many(
        """
        INSERT INTO trade_records (
            trade_id, ts, buyer_node_id, seller_node_id,
            quantity_kwh, cleared_price_inr_per_kwh, status, metadata
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        """,
        rows,
    )
    return {"inserted": len(rows)}
