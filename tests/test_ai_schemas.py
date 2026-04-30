from __future__ import annotations

from datetime import datetime, timedelta

from ai.schemas import (
    NeighborhoodSignal,
    NeighborhoodSignalType,
    NodeState,
    TradeOrder,
    TradeOrderSide,
)


def test_node_state_serializes_signal_and_timestamp() -> None:
    state = NodeState(
        timestamp=datetime(2026, 4, 30, 12, 0, 0),
        node_id=7,
        battery_soc_kwh=4.2,
        battery_power_kw=-1.5,
        solar_output_kw=2.8,
        household_load_kw=1.9,
        ev_charge_kw=0.0,
        ev_target_soc=0.8,
        ev_hours_to_deadline=6.0,
        net_grid_kw=-0.9,
        voltage_pu=0.97,
        market_buy_price_inr_per_kwh=8.2,
        market_sell_price_inr_per_kwh=7.6,
        active_signal=NeighborhoodSignalType.THROTTLE,
    )

    payload = state.to_dict()

    assert payload["timestamp"] == "2026-04-30T12:00:00"
    assert payload["active_signal"] == "throttle"


def test_trade_order_serializes_enum_fields() -> None:
    order = TradeOrder(
        order_id="ord-1",
        timestamp=datetime(2026, 4, 30, 12, 0, 0),
        node_id=11,
        side=TradeOrderSide.SELL,
        quantity_kwh=3.0,
        limit_price_inr_per_kwh=9.5,
        expires_at=datetime(2026, 4, 30, 12, 5, 0),
    )

    payload = order.to_dict()

    assert payload["side"] == "sell"
    assert payload["status"] == "open"
    assert payload["expires_at"] == "2026-04-30T12:05:00"


def test_neighborhood_signal_optional_expiry_serialization() -> None:
    signal = NeighborhoodSignal(
        signal_id="sig-1",
        timestamp=datetime(2026, 4, 30, 18, 0, 0),
        signal_type=NeighborhoodSignalType.ISLAND,
        severity=0.91,
        target_node_ids=[1, 2, 3],
        reason="Predicted voltage collapse",
        expires_at=datetime(2026, 4, 30, 18, 0, 0) + timedelta(minutes=30),
    )

    payload = signal.to_dict()

    assert payload["signal_type"] == "island"
    assert payload["expires_at"] == "2026-04-30T18:30:00"
