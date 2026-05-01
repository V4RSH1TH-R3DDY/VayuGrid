from __future__ import annotations

from datetime import datetime, timezone

from ai.schemas import NeighborhoodSignal, NeighborhoodSignalType
from trading.models import MarketMode
from trading.pricing import PricingPolicy


def test_pricing_policy_raises_cap_and_floor_under_stress() -> None:
    signal = NeighborhoodSignal(
        signal_id="sig-1",
        timestamp=datetime(2026, 5, 1, 10, 0, tzinfo=timezone.utc),
        signal_type=NeighborhoodSignalType.THROTTLE,
        severity=0.7,
        target_node_ids=[1, 2],
        reason="transformer loading rising",
    )

    state = PricingPolicy().from_signal(signal)

    assert state.mode == MarketMode.STRESSED
    assert state.accepts_orders
    assert state.price_floor_inr_per_kwh > 3.0
    assert state.price_cap_inr_per_kwh > 12.0


def test_pricing_policy_suspends_market_for_island_signal() -> None:
    signal = NeighborhoodSignal(
        signal_id="sig-1",
        timestamp=datetime(2026, 5, 1, 10, 0, tzinfo=timezone.utc),
        signal_type=NeighborhoodSignalType.ISLAND,
        severity=0.95,
        target_node_ids=[1, 2],
        reason="voltage collapse",
    )

    state = PricingPolicy().from_signal(signal)

    assert state.mode == MarketMode.ISLANDED
    assert not state.accepts_orders
