from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    OPEN = "open"
    MATCHED = "matched"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    REJECTED = "rejected"
    SETTLED = "settled"


class MarketMode(str, Enum):
    NORMAL = "normal"
    STRESSED = "stressed"
    ISLANDED = "islanded"
    SUSPENDED = "suspended"


@dataclass(slots=True)
class MarketState:
    mode: MarketMode = MarketMode.NORMAL
    stress_score: float = 0.0
    price_floor_inr_per_kwh: float = 3.0
    price_cap_inr_per_kwh: float = 12.0
    accepts_orders: bool = True
    reason: str = "normal"
    updated_at: datetime | None = None

    def clamp_price(self, price: float) -> float:
        return min(max(price, self.price_floor_inr_per_kwh), self.price_cap_inr_per_kwh)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["mode"] = self.mode.value
        payload["updated_at"] = self.updated_at.isoformat() if self.updated_at else None
        return payload


@dataclass(slots=True)
class TradeOrder:
    order_id: str
    node_id: int
    side: OrderSide
    quantity_kwh: float
    limit_price_inr_per_kwh: float
    created_at: datetime
    expires_at: datetime
    status: OrderStatus = OrderStatus.OPEN
    remaining_kwh: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.remaining_kwh is None:
            self.remaining_kwh = self.quantity_kwh

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["side"] = self.side.value
        payload["status"] = self.status.value
        payload["created_at"] = self.created_at.isoformat()
        payload["expires_at"] = self.expires_at.isoformat()
        return payload


@dataclass(slots=True)
class MatchResult:
    trade_id: str
    buyer_order_id: str
    seller_order_id: str
    buyer_node_id: int
    seller_node_id: int
    quantity_kwh: float
    cleared_price_inr_per_kwh: float
    matched_at: datetime

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["matched_at"] = self.matched_at.isoformat()
        return payload


@dataclass(slots=True)
class LedgerEntry:
    sequence: int
    trade_id: str
    ts: datetime
    buyer_node_id: int
    seller_node_id: int
    quantity_kwh: float
    cleared_price_inr_per_kwh: float
    previous_hash: str
    entry_hash: str
    buyer_signature: str | None = None
    seller_signature: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ts"] = self.ts.isoformat()
        return payload
