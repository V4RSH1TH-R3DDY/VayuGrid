from __future__ import annotations

import uuid
from datetime import datetime, timezone

from .models import MatchResult, OrderSide, OrderStatus, TradeOrder


class OrderBook:
    def __init__(self, max_order_kwh: float = 5.0) -> None:
        self.max_order_kwh = max_order_kwh
        self._orders: dict[str, TradeOrder] = {}

    @property
    def orders(self) -> list[TradeOrder]:
        return list(self._orders.values())

    @property
    def open_orders(self) -> list[TradeOrder]:
        return [order for order in self._orders.values() if order.status == OrderStatus.OPEN]

    def add(self, order: TradeOrder, now: datetime | None = None) -> TradeOrder:
        now = now or datetime.now(timezone.utc)
        if order.quantity_kwh <= 0:
            order.status = OrderStatus.REJECTED
            order.metadata["rejection_reason"] = "quantity must be positive"
        elif order.quantity_kwh > self.max_order_kwh:
            order.status = OrderStatus.REJECTED
            order.metadata["rejection_reason"] = f"quantity exceeds {self.max_order_kwh} kWh"
        elif order.expires_at <= now:
            order.status = OrderStatus.REJECTED
            order.metadata["rejection_reason"] = "order expires in the past"

        self._orders[order.order_id] = order
        return order

    def cancel(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if not order or order.status != OrderStatus.OPEN:
            return False
        order.status = OrderStatus.CANCELLED
        return True

    def expire(self, now: datetime | None = None) -> list[TradeOrder]:
        now = now or datetime.now(timezone.utc)
        expired = []
        for order in self.open_orders:
            if order.expires_at <= now:
                order.status = OrderStatus.EXPIRED
                expired.append(order)
        return expired

    def match(self, now: datetime | None = None) -> list[MatchResult]:
        now = now or datetime.now(timezone.utc)
        self.expire(now)
        matches: list[MatchResult] = []

        while True:
            buys = sorted(
                [o for o in self.open_orders if o.side == OrderSide.BUY],
                key=lambda o: (-o.limit_price_inr_per_kwh, o.created_at),
            )
            sells = sorted(
                [o for o in self.open_orders if o.side == OrderSide.SELL],
                key=lambda o: (o.limit_price_inr_per_kwh, o.created_at),
            )
            if not buys or not sells:
                break

            best_bid = buys[0]
            best_ask = sells[0]
            if best_bid.node_id == best_ask.node_id:
                newer = max(best_bid, best_ask, key=lambda o: o.created_at)
                newer.status = OrderStatus.REJECTED
                newer.metadata["rejection_reason"] = "wash trade blocked"
                continue
            if best_bid.limit_price_inr_per_kwh < best_ask.limit_price_inr_per_kwh:
                break

            quantity = min(best_bid.remaining_kwh or 0.0, best_ask.remaining_kwh or 0.0)
            if quantity <= 0:
                break
            price = round(
                (best_bid.limit_price_inr_per_kwh + best_ask.limit_price_inr_per_kwh) / 2,
                4,
            )
            matches.append(
                MatchResult(
                    trade_id=str(uuid.uuid4()),
                    buyer_order_id=best_bid.order_id,
                    seller_order_id=best_ask.order_id,
                    buyer_node_id=best_bid.node_id,
                    seller_node_id=best_ask.node_id,
                    quantity_kwh=round(quantity, 6),
                    cleared_price_inr_per_kwh=price,
                    matched_at=now,
                )
            )

            best_bid.remaining_kwh = round((best_bid.remaining_kwh or 0.0) - quantity, 6)
            best_ask.remaining_kwh = round((best_ask.remaining_kwh or 0.0) - quantity, 6)
            if best_bid.remaining_kwh <= 0:
                best_bid.status = OrderStatus.MATCHED
            if best_ask.remaining_kwh <= 0:
                best_ask.status = OrderStatus.MATCHED

        return matches
