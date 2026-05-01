from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone

from .ledger import AyaLedger
from .models import MarketState, OrderSide, OrderStatus, TradeOrder
from .order_book import OrderBook


class MatchingEngine:
    def __init__(
        self,
        market_state: MarketState | None = None,
        order_book: OrderBook | None = None,
        ledger: AyaLedger | None = None,
        max_orders_per_minute: int = 10,
        default_order_ttl_seconds: int = 60,
        max_queue_depth: int = 500,
    ) -> None:
        self.market_state = market_state or MarketState(updated_at=datetime.now(timezone.utc))
        self.order_book = order_book or OrderBook()
        self.ledger = ledger or AyaLedger()
        self.max_orders_per_minute = max_orders_per_minute
        self.default_order_ttl_seconds = default_order_ttl_seconds
        self.max_queue_depth = max_queue_depth
        self._node_order_times: dict[int, deque[float]] = defaultdict(deque)

    def update_market_state(self, market_state: MarketState) -> None:
        self.market_state = market_state

    def submit_order(
        self,
        node_id: int,
        side: OrderSide,
        quantity_kwh: float,
        limit_price_inr_per_kwh: float,
        now: datetime | None = None,
        order_id: str | None = None,
        metadata: dict | None = None,
    ) -> TradeOrder:
        now = now or datetime.now(timezone.utc)
        order = TradeOrder(
            order_id=order_id or str(uuid.uuid4()),
            node_id=node_id,
            side=side,
            quantity_kwh=quantity_kwh,
            limit_price_inr_per_kwh=self.market_state.clamp_price(limit_price_inr_per_kwh),
            created_at=now,
            expires_at=now + timedelta(seconds=self.default_order_ttl_seconds),
            metadata=metadata or {},
        )

        depth = len(self.order_book.open_orders)
        if depth >= self.max_queue_depth:
            order.status = OrderStatus.REJECTED
            order.metadata["rejection_reason"] = (
                f"circuit breaker open: queue depth {depth} >= {self.max_queue_depth}"
            )
        elif not self.market_state.accepts_orders:
            order.status = OrderStatus.REJECTED
            order.metadata["rejection_reason"] = f"market is {self.market_state.mode.value}"
        elif self._is_rate_limited(node_id):
            order.status = OrderStatus.REJECTED
            order.metadata["rejection_reason"] = "node exceeded 10 orders per minute"

        if order.status == OrderStatus.OPEN:
            self._record_order_time(node_id)

        return self.order_book.add(order, now=now)

    def cancel_order(self, order_id: str) -> bool:
        return self.order_book.cancel(order_id)

    def settle_once(self, now: datetime | None = None) -> list[dict]:
        matches = self.order_book.match(now=now)
        settlements = []
        for match in matches:
            ledger_entry = self.ledger.append(match)
            settlements.append({"match": match, "ledger_entry": ledger_entry})
        return settlements

    def _is_rate_limited(self, node_id: int) -> bool:
        now = time.monotonic()
        recent = self._node_order_times[node_id]
        while recent and now - recent[0] > 60:
            recent.popleft()
        return len(recent) >= self.max_orders_per_minute

    def _record_order_time(self, node_id: int) -> None:
        self._node_order_times[node_id].append(time.monotonic())
