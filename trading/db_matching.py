from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Json

from api.app.db import pool

from .ledger import GENESIS_HASH, AyaLedger
from .models import LedgerEntry, OrderSide, TradeOrder
from .order_book import OrderBook


def _as_order(row: dict[str, Any]) -> TradeOrder:
    return TradeOrder(
        order_id=row["order_id"],
        node_id=int(row["node_id"]),
        side=OrderSide(row["side"]),
        quantity_kwh=float(row["quantity_kwh"]),
        remaining_kwh=float(row["remaining_kwh"]),
        limit_price_inr_per_kwh=float(row["limit_price_inr_per_kwh"]),
        created_at=row["created_at"],
        expires_at=row["expires_at"],
        metadata=row.get("metadata") or {},
    )


def _entry_from_row(row: dict[str, Any]) -> LedgerEntry:
    return LedgerEntry(
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


def settle_open_orders(now: datetime | None = None) -> list[dict[str, Any]]:
    now = now or datetime.now(timezone.utc)
    with pool.connection() as conn:
        with conn.transaction():
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM trade_orders
                    WHERE status = 'open'
                    ORDER BY created_at
                    FOR UPDATE SKIP LOCKED
                    """
                )
                rows = list(cur.fetchall())
                cur.execute(
                    """
                    UPDATE trade_orders
                    SET status = 'expired'
                    WHERE status = 'open' AND expires_at <= %s
                    RETURNING order_id
                    """,
                    (now,),
                )
                expired = [row["order_id"] for row in cur.fetchall()]

                book = OrderBook()
                for row in rows:
                    if row["order_id"] not in expired:
                        book.add(_as_order(row), now=now)
                matches = book.match(now=now)

                cur.execute("SELECT * FROM ledger_entries ORDER BY sequence")
                ledger = AyaLedger(_entry_from_row(row) for row in cur.fetchall())

                settlements = []
                for match in matches:
                    ledger_entry = ledger.append(match)
                    cur.execute(
                        """
                        INSERT INTO trade_records (
                            trade_id, ts, buyer_node_id, seller_node_id,
                            quantity_kwh, cleared_price_inr_per_kwh, status, metadata
                        ) VALUES (%s, %s, %s, %s, %s, %s, 'settled', %s)
                        ON CONFLICT (trade_id) DO NOTHING
                        """,
                        (
                            match.trade_id,
                            match.matched_at,
                            match.buyer_node_id,
                            match.seller_node_id,
                            match.quantity_kwh,
                            match.cleared_price_inr_per_kwh,
                            Json(
                                {
                                    "buyer_order_id": match.buyer_order_id,
                                    "seller_order_id": match.seller_order_id,
                                }
                            ),
                        ),
                    )
                    cur.execute(
                        """
                        INSERT INTO ledger_entries (
                            trade_id, ts, buyer_node_id, seller_node_id,
                            quantity_kwh, cleared_price_inr_per_kwh, previous_hash,
                            entry_hash, buyer_signature, seller_signature, metadata
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (trade_id) DO NOTHING
                        """,
                        (
                            ledger_entry.trade_id,
                            ledger_entry.ts,
                            ledger_entry.buyer_node_id,
                            ledger_entry.seller_node_id,
                            ledger_entry.quantity_kwh,
                            ledger_entry.cleared_price_inr_per_kwh,
                            ledger_entry.previous_hash,
                            ledger_entry.entry_hash,
                            ledger_entry.buyer_signature,
                            ledger_entry.seller_signature,
                            Json(ledger_entry.metadata),
                        ),
                    )
                    settlements.append(
                        {"match": match.to_dict(), "ledger_entry": ledger_entry.to_dict()}
                    )

                for order in book.orders:
                    cur.execute(
                        """
                        UPDATE trade_orders
                        SET remaining_kwh = %s,
                            status = %s,
                            matched_at = CASE WHEN %s = 'matched' THEN %s ELSE matched_at END,
                            metadata = %s
                        WHERE order_id = %s
                        """,
                        (
                            order.remaining_kwh,
                            order.status.value,
                            order.status.value,
                            now,
                            Json(order.metadata),
                            order.order_id,
                        ),
                    )

                if matches or expired:
                    cur.execute(
                        """
                        INSERT INTO market_events (event_id, ts, event_type, payload)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (
                            str(uuid.uuid4()),
                            now,
                            "settlement_cycle",
                            Json({"matches": len(matches), "expired_orders": expired}),
                        ),
                    )
                return settlements


def latest_ledger_hash() -> str:
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT entry_hash FROM ledger_entries ORDER BY sequence DESC LIMIT 1")
            row = cur.fetchone()
            return row["entry_hash"] if row else GENESIS_HASH
