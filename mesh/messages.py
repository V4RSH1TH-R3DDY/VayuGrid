from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class MessageKind(str, Enum):
    TRADE_ORDER = "trade_order"
    HEARTBEAT = "heartbeat"
    STATE_UPDATE = "state_update"
    SETTLEMENT = "settlement"
    SIGNAL = "signal"


@dataclass(slots=True)
class TradeOrderPayload:
    order_id: str
    node_id: int
    side: str
    quantity_kwh: float
    limit_price_inr_per_kwh: float
    expires_at: datetime

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["expires_at"] = self.expires_at.isoformat()
        return payload


@dataclass(slots=True)
class HeartbeatPayload:
    node_id: int
    battery_soc_kwh: float
    solar_output_kw: float
    household_load_kw: float
    voltage_pu: float
    buy_price_inr_per_kwh: float
    sell_price_inr_per_kwh: float


@dataclass(slots=True)
class GossipMessage:
    kind: MessageKind
    source_node_id: int
    payload: dict[str, Any]
    created_at: datetime
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ttl_hops: int = 3
    signature: str | None = None

    def should_forward(self) -> bool:
        return self.ttl_hops > 0

    def forwarded(self) -> "GossipMessage":
        return GossipMessage(
            kind=self.kind,
            source_node_id=self.source_node_id,
            payload=self.payload,
            created_at=self.created_at,
            message_id=self.message_id,
            ttl_hops=max(0, self.ttl_hops - 1),
            signature=self.signature,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        payload["created_at"] = self.created_at.isoformat()
        return payload
