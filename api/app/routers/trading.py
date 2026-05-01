from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from psycopg.types.json import Json

from trading.db_matching import settle_open_orders
from trading.ledger import GENESIS_HASH, AyaLedger
from trading.models import LedgerEntry, MarketMode

from ..db import execute, fetch_all, fetch_one
from ..rate_limit import limiter
from ..schemas import TradeOrderIn
from ..security import UserClaims, require_roles
from .nodes import get_node_id_from_api_key

router = APIRouter(tags=["trading"])

MAX_ORDER_KWH = 5.0
MAX_ORDERS_PER_MINUTE = 10
ORDER_TTL_SECONDS = 60


def _market_state() -> dict:
    row = fetch_one(
        """
        SELECT mode, stress_score, price_floor_inr_per_kwh, price_cap_inr_per_kwh,
               accepts_orders, reason, updated_at
        FROM market_state
        WHERE id = TRUE
        """
    )
    return row or {
        "mode": "normal",
        "stress_score": 0,
        "price_floor_inr_per_kwh": 3,
        "price_cap_inr_per_kwh": 12,
        "accepts_orders": True,
        "reason": "normal",
        "updated_at": datetime.now(timezone.utc),
    }


def _clamp_price(price: float, market: dict) -> float:
    floor = float(market["price_floor_inr_per_kwh"])
    cap = float(market["price_cap_inr_per_kwh"])
    return round(min(max(price, floor), cap), 4)


@router.get("/trading/market")
@limiter.limit("100/minute")
def get_market_state(
    request: Request,
    _: UserClaims = Depends(require_roles(["operator", "homeowner", "community"])),
) -> dict:
    market = _market_state()
    recent = fetch_one(
        """
        SELECT
            COALESCE(
                SUM(CASE WHEN side = 'buy' AND status = 'open' THEN remaining_kwh ELSE 0 END),
                0
            )
                AS open_buy_kwh,
            COALESCE(
                SUM(CASE WHEN side = 'sell' AND status = 'open' THEN remaining_kwh ELSE 0 END),
                0
            )
                AS open_sell_kwh,
            COUNT(*) FILTER (WHERE status = 'open') AS open_order_count
        FROM trade_orders
        """
    ) or {}
    return {"market": market, "liquidity": recent}


@router.post("/trading/orders")
@limiter.limit("1000/minute")
def submit_order(
    request: Request,
    payload: TradeOrderIn,
    node_id: int = Depends(get_node_id_from_api_key),
) -> dict:
    side = payload.side.lower()
    if side not in {"buy", "sell"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="side must be buy or sell",
        )

    market = _market_state()
    status_value = "open"
    metadata = payload.metadata or {}
    if not market["accepts_orders"] or market["mode"] == MarketMode.ISLANDED.value:
        status_value = "rejected"
        metadata["rejection_reason"] = f"market is {market['mode']}"

    recent_orders = fetch_one(
        """
        SELECT COUNT(*) AS count
        FROM trade_orders
        WHERE node_id = %s AND created_at >= now() - interval '1 minute'
        """,
        (node_id,),
    )
    if recent_orders and int(recent_orders["count"]) >= MAX_ORDERS_PER_MINUTE:
        status_value = "rejected"
        metadata["rejection_reason"] = "node exceeded 10 orders per minute"

    order_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc)
    expires_at = created_at + timedelta(seconds=ORDER_TTL_SECONDS)
    limit_price = _clamp_price(payload.limit_price_inr_per_kwh, market)
    execute(
        """
        INSERT INTO trade_orders (
            order_id, node_id, side, quantity_kwh, remaining_kwh,
            limit_price_inr_per_kwh, status, created_at, expires_at, metadata
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            order_id,
            node_id,
            side,
            payload.quantity_kwh,
            payload.quantity_kwh if status_value == "open" else 0,
            limit_price,
            status_value,
            created_at,
            expires_at,
            Json(metadata),
        ),
    )
    return {
        "order_id": order_id,
        "node_id": node_id,
        "side": side,
        "quantity_kwh": payload.quantity_kwh,
        "remaining_kwh": payload.quantity_kwh if status_value == "open" else 0,
        "limit_price_inr_per_kwh": limit_price,
        "status": status_value,
        "created_at": created_at,
        "expires_at": expires_at,
        "metadata": metadata,
    }


@router.get("/trading/orders")
@limiter.limit("100/minute")
def list_orders(
    request: Request,
    status_filter: str | None = Query(default=None, alias="status"),
    user: UserClaims = Depends(require_roles(["operator", "homeowner"])),
) -> dict:
    clauses = []
    params: list[object] = []
    if status_filter:
        clauses.append("status = %s")
        params.append(status_filter)
    if user.role == "homeowner":
        clauses.append("node_id = %s")
        params.append(user.node_id)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = fetch_all(
        f"""
        SELECT *
        FROM trade_orders
        {where}
        ORDER BY created_at DESC
        LIMIT 200
        """,
        params,
    )
    return {"orders": rows}


@router.post("/trading/orders/{order_id}/cancel")
@limiter.limit("100/minute")
def cancel_order(
    request: Request,
    order_id: str,
    node_id: int = Depends(get_node_id_from_api_key),
) -> dict:
    order = fetch_one("SELECT node_id, status FROM trade_orders WHERE order_id = %s", (order_id,))
    if not order:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Order not found")
    if int(order["node_id"]) != node_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Node ID mismatch")
    if order["status"] != "open":
        return {"order_id": order_id, "cancelled": False, "status": order["status"]}
    execute(
        "UPDATE trade_orders SET status = 'cancelled', remaining_kwh = 0 WHERE order_id = %s",
        (order_id,),
    )
    return {"order_id": order_id, "cancelled": True, "status": "cancelled"}


@router.post("/trading/settle")
@limiter.limit("100/minute")
def settle_orders(
    request: Request, _: UserClaims = Depends(require_roles(["operator"]))
) -> dict:
    settlements = settle_open_orders()
    return {"settled": len(settlements), "settlements": settlements}


@router.get("/trading/ledger/monthly-summary")
@limiter.limit("100/minute")
def monthly_ledger_summary(
    request: Request, _: UserClaims = Depends(require_roles(["operator", "community"]))
) -> dict:
    rows = fetch_all(
        """
        SELECT
            date_trunc('month', ts) AS month,
            buyer_node_id,
            seller_node_id,
            SUM(quantity_kwh) AS quantity_kwh,
            SUM(quantity_kwh * cleared_price_inr_per_kwh) AS value_inr
        FROM ledger_entries
        GROUP BY month, buyer_node_id, seller_node_id
        ORDER BY month DESC, buyer_node_id, seller_node_id
        LIMIT 500
        """
    )
    latest = fetch_one("SELECT entry_hash FROM ledger_entries ORDER BY sequence DESC LIMIT 1")
    return {
        "latest_hash": latest["entry_hash"] if latest else GENESIS_HASH,
        "summary": rows,
    }


@router.get("/trading/ledger/verify")
@limiter.limit("100/minute")
def verify_ledger(
    request: Request, _: UserClaims = Depends(require_roles(["operator"]))
) -> dict:
    rows = fetch_all("SELECT * FROM ledger_entries ORDER BY sequence")
    ledger = AyaLedger(
        [
            LedgerEntry(
                sequence=int(row["sequence"]),
                trade_id=row["trade_id"],
                ts=row["ts"],
                buyer_node_id=int(row["buyer_node_id"]),
                seller_node_id=int(row["seller_node_id"]),
                quantity_kwh=float(row["quantity_kwh"]),
                cleared_price_inr_per_kwh=float(row["cleared_price_inr_per_kwh"]),
                previous_hash=row["previous_hash"],
                entry_hash=row["entry_hash"],
                buyer_signature=row.get("buyer_signature"),
                seller_signature=row.get("seller_signature"),
                metadata=row.get("metadata") or {},
            )
            for row in rows
        ]
    )
    return {"valid": ledger.verify(), "entries": len(rows), "latest_hash": ledger.latest_hash}
