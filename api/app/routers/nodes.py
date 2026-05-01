from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, Header, HTTPException, status
from psycopg.types.json import Json

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
    payload: List[TelemetryIn],
    node_id: int = Depends(get_node_id_from_api_key),
) -> dict:
    rows = []
    for item in payload:
        if item.node_id != node_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Node ID mismatch for API key",
            )
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
    return {"inserted": len(rows)}


@router.post("/nodes/transformer-readings")
@limiter.limit("1000/minute")
def ingest_transformer_readings(
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
