from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from ai.schemas import NeighborhoodSignal, NeighborhoodSignalType

from .models import MarketMode, MarketState


@dataclass(slots=True)
class PricingPolicy:
    normal_floor_inr_per_kwh: float = 3.0
    normal_cap_inr_per_kwh: float = 12.0
    moderate_stress_threshold: float = 0.55
    severe_stress_threshold: float = 0.85
    stressed_floor_lift_inr: float = 1.0
    stressed_cap_lift_inr: float = 6.0

    def from_signal(self, signal: NeighborhoodSignal | None) -> MarketState:
        now = datetime.now(timezone.utc)
        if signal is None:
            return MarketState(updated_at=now)

        if signal.signal_type == NeighborhoodSignalType.ISLAND:
            return MarketState(
                mode=MarketMode.ISLANDED,
                stress_score=signal.severity,
                price_floor_inr_per_kwh=self.normal_floor_inr_per_kwh,
                price_cap_inr_per_kwh=self.normal_cap_inr_per_kwh,
                accepts_orders=False,
                reason=signal.reason or "island signal active",
                updated_at=now,
            )

        if signal.signal_type == NeighborhoodSignalType.RESUME:
            return MarketState(reason=signal.reason or "resume signal", updated_at=now)

        floor = (
            signal.recommended_price_floor_inr_per_kwh
            if signal.recommended_price_floor_inr_per_kwh is not None
            else self.normal_floor_inr_per_kwh
        )
        cap = (
            signal.recommended_price_cap_inr_per_kwh
            if signal.recommended_price_cap_inr_per_kwh is not None
            else self.normal_cap_inr_per_kwh
        )

        if signal.severity >= self.moderate_stress_threshold:
            lift_ratio = min(1.0, signal.severity)
            floor += self.stressed_floor_lift_inr * lift_ratio
            cap += self.stressed_cap_lift_inr * lift_ratio
            mode = MarketMode.STRESSED
        else:
            mode = MarketMode.NORMAL

        if signal.severity >= self.severe_stress_threshold:
            cap += self.stressed_cap_lift_inr * 0.5

        return MarketState(
            mode=mode,
            stress_score=signal.severity,
            price_floor_inr_per_kwh=round(floor, 4),
            price_cap_inr_per_kwh=round(cap, 4),
            accepts_orders=True,
            reason=signal.reason,
            updated_at=now,
        )
