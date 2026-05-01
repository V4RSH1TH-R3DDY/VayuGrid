from __future__ import annotations

from datetime import datetime, timedelta, timezone

from trading.engine import MatchingEngine
from trading.models import MarketMode, MarketState, OrderSide, OrderStatus, TradeOrder
from trading.order_book import OrderBook


def test_order_book_matches_best_bid_and_ask_at_midpoint() -> None:
    now = datetime(2026, 5, 1, 10, 0, tzinfo=timezone.utc)
    book = OrderBook()
    book.add(
        TradeOrder(
            order_id="sell-1",
            node_id=1,
            side=OrderSide.SELL,
            quantity_kwh=2.0,
            limit_price_inr_per_kwh=7.0,
            created_at=now,
            expires_at=now + timedelta(seconds=60),
        ),
        now=now,
    )
    book.add(
        TradeOrder(
            order_id="buy-1",
            node_id=2,
            side=OrderSide.BUY,
            quantity_kwh=2.0,
            limit_price_inr_per_kwh=9.0,
            created_at=now,
            expires_at=now + timedelta(seconds=60),
        ),
        now=now,
    )

    matches = book.match(now=now)

    assert len(matches) == 1
    assert matches[0].quantity_kwh == 2.0
    assert matches[0].cleared_price_inr_per_kwh == 8.0


def test_order_book_rejects_wash_trade() -> None:
    now = datetime(2026, 5, 1, 10, 0, tzinfo=timezone.utc)
    engine = MatchingEngine()
    engine.submit_order(1, OrderSide.SELL, 1.0, 6.0, now=now, order_id="sell")
    engine.submit_order(1, OrderSide.BUY, 1.0, 10.0, now=now, order_id="buy")

    settlements = engine.settle_once(now=now)

    assert settlements == []
    rejected = [order for order in engine.order_book.orders if order.status == OrderStatus.REJECTED]
    assert rejected
    assert rejected[0].metadata["rejection_reason"] == "wash trade blocked"


def test_engine_rejects_orders_above_phase4_quantity_limit() -> None:
    engine = MatchingEngine()
    order = engine.submit_order(1, OrderSide.SELL, 5.1, 8.0)

    assert order.status == OrderStatus.REJECTED
    assert "quantity exceeds" in order.metadata["rejection_reason"]


def test_engine_rate_limits_more_than_ten_orders_per_minute() -> None:
    engine = MatchingEngine()

    accepted = [
        engine.submit_order(1, OrderSide.SELL, 0.1, 8.0, order_id=f"order-{index}")
        for index in range(10)
    ]
    rejected = engine.submit_order(1, OrderSide.SELL, 0.1, 8.0, order_id="order-10")

    assert all(order.status == OrderStatus.OPEN for order in accepted)
    assert rejected.status == OrderStatus.REJECTED
    assert rejected.metadata["rejection_reason"] == "node exceeded 10 orders per minute"


def test_engine_suspends_orders_during_island_mode() -> None:
    engine = MatchingEngine(
        market_state=MarketState(
            mode=MarketMode.ISLANDED,
            accepts_orders=False,
            reason="island signal active",
        )
    )

    order = engine.submit_order(1, OrderSide.BUY, 1.0, 8.0)

    assert order.status == OrderStatus.REJECTED
    assert order.metadata["rejection_reason"] == "market is islanded"


def test_circuit_breaker_halts_orders_at_queue_depth_limit() -> None:
    now = datetime(2026, 5, 1, 10, 0, tzinfo=timezone.utc)
    engine = MatchingEngine(max_queue_depth=3)

    # Submit 3 BUY orders from different nodes — all should be accepted (open)
    for i in range(3):
        order = engine.submit_order(i + 10, OrderSide.BUY, 1.0, 8.0, now=now, order_id=f"buy-{i}")
        assert order.status == OrderStatus.OPEN, f"Expected OPEN, got {order.status}"

    # 4th order must be rejected by the circuit breaker
    rejected = engine.submit_order(99, OrderSide.BUY, 1.0, 8.0, now=now, order_id="overflow")
    assert rejected.status == OrderStatus.REJECTED
    assert "circuit breaker" in rejected.metadata["rejection_reason"]
