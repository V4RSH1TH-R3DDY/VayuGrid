from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from ai.schemas import NeighborhoodSignal, NeighborhoodSignalType


@dataclass
class SignalDecision:
    signal: NeighborhoodSignal | None = None
    reason: str = ""


class SignalTranslator:
    """Translates GNN predictions into actionable NeighborhoodSignals.

    Threshold logic:
      - Risk > 0.85 or voltage < 0.88 pu in 5 min → ISLAND
      - Risk > 0.5 → THROTTLE to top 20% flexible loads
      - Duck curve ramp > 2 kW/min predicted → PRE_COOL
      - Risk < 0.1 for 5 continuous checks → RESUME
    """

    RISK_ISLAND = 0.85
    VOLTAGE_ISLAND = 0.88
    RISK_THROTTLE = 0.5
    DUCK_RAMP_THRESHOLD = 2.0
    RISK_RESUME = 0.1
    STABLE_CHECKS = 5

    def __init__(self) -> None:
        self._stable_count = 0

    def decide(
        self,
        overload_prob: list[float],
        voltage_forecast: list[float],
        risk: float,
        duck_ramp_rate: float,
        target_node_ids: list[int],
    ) -> SignalDecision:
        min_voltage_5min = min(voltage_forecast[:5]) if len(voltage_forecast) >= 5 else 1.0
        max_overload_30min = max(overload_prob) if overload_prob else 0.0

        if risk > self.RISK_ISLAND or min_voltage_5min < self.VOLTAGE_ISLAND:
            self._stable_count = 0
            return SignalDecision(
                signal=NeighborhoodSignal(
                    signal_id="",
                    timestamp=datetime.now(timezone.utc),
                    signal_type=NeighborhoodSignalType.ISLAND,
                    severity=risk,
                    target_node_ids=target_node_ids,
                    reason=(
                        f"Risk {risk:.2f} > {self.RISK_ISLAND}"
                        f" or voltage {min_voltage_5min:.3f} < {self.VOLTAGE_ISLAND}"
                    ),
                ),
                reason=f"ISLAND: risk={risk:.2f}, min_voltage_5min={min_voltage_5min:.3f}",
            )

        if risk > self.RISK_THROTTLE:
            self._stable_count = 0
            return SignalDecision(
                signal=NeighborhoodSignal(
                    signal_id="",
                    timestamp=datetime.now(timezone.utc),
                    signal_type=NeighborhoodSignalType.THROTTLE,
                    severity=risk,
                    target_node_ids=target_node_ids,
                    reason=f"Risk {risk:.2f} > {self.RISK_THROTTLE}",
                ),
                reason=f"THROTTLE: risk={risk:.2f}",
            )

        if duck_ramp_rate > self.DUCK_RAMP_THRESHOLD:
            self._stable_count = 0
            return SignalDecision(
                signal=NeighborhoodSignal(
                    signal_id="",
                    timestamp=datetime.now(timezone.utc),
                    signal_type=NeighborhoodSignalType.PRE_COOL,
                    severity=duck_ramp_rate / 5.0,
                    target_node_ids=target_node_ids,
                    reason=(
                        f"Duck ramp {duck_ramp_rate:.1f} kW/min"
                        f" > {self.DUCK_RAMP_THRESHOLD}"
                    ),
                ),
                reason=f"PRE_COOL: duck_ramp={duck_ramp_rate:.1f} kW/min",
            )

        if max_overload_30min < self.RISK_RESUME:
            self._stable_count += 1
            if self._stable_count >= self.STABLE_CHECKS:
                self._stable_count = 0
                return SignalDecision(
                    signal=NeighborhoodSignal(
                        signal_id="",
                        timestamp=datetime.now(timezone.utc),
                        signal_type=NeighborhoodSignalType.RESUME,
                        severity=0.0,
                        target_node_ids=target_node_ids,
                        reason="Stable for 5 consecutive checks",
                    ),
                    reason="RESUME: stable",
                )
        else:
            self._stable_count = 0

        return SignalDecision(reason="no action")
