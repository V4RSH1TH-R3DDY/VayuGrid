from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TradeOrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class TradeOrderStatus(str, Enum):
    OPEN = "open"
    MATCHED = "matched"
    CANCELLED = "cancelled"
    SETTLED = "settled"


class NeighborhoodSignalType(str, Enum):
    THROTTLE = "throttle"
    PRE_COOL = "pre_cool"
    ISLAND = "island"
    RESUME = "resume"


@dataclass(slots=True)
class NodeState:
    timestamp: datetime
    node_id: int
    battery_soc_kwh: float
    battery_power_kw: float
    solar_output_kw: float
    household_load_kw: float
    ev_charge_kw: float
    ev_target_soc: float
    ev_hours_to_deadline: float
    net_grid_kw: float
    voltage_pu: float
    market_buy_price_inr_per_kwh: float
    market_sell_price_inr_per_kwh: float
    active_signal: NeighborhoodSignalType | None = None
    forecast_solar_kw_15m: float = 0.0
    forecast_load_kw_15m: float = 0.0
    forecast_price_inr_per_kwh_15m: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        payload["active_signal"] = self.active_signal.value if self.active_signal else None
        return payload


@dataclass(slots=True)
class TradeOrder:
    order_id: str
    timestamp: datetime
    node_id: int
    side: TradeOrderSide
    quantity_kwh: float
    limit_price_inr_per_kwh: float
    expires_at: datetime
    status: TradeOrderStatus = TradeOrderStatus.OPEN
    counterparty_node_id: int | None = None
    cleared_price_inr_per_kwh: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        payload["expires_at"] = self.expires_at.isoformat()
        payload["side"] = self.side.value
        payload["status"] = self.status.value
        return payload


@dataclass(slots=True)
class GridTelemetry:
    timestamp: datetime
    transformer_id: str
    feeder_total_kw: float
    transformer_loading_pu: float
    max_branch_loading_pu: float
    hottest_spot_temp_c: float
    aging_acceleration: float
    grid_available: bool
    islanding_triggered: bool
    maintenance_mode: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload


@dataclass(slots=True)
class NeighborhoodSignal:
    signal_id: str
    timestamp: datetime
    signal_type: NeighborhoodSignalType
    severity: float
    target_node_ids: list[int]
    reason: str
    recommended_price_floor_inr_per_kwh: float | None = None
    recommended_price_cap_inr_per_kwh: float | None = None
    expires_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        payload["signal_type"] = self.signal_type.value
        payload["expires_at"] = self.expires_at.isoformat() if self.expires_at else None
        return payload
