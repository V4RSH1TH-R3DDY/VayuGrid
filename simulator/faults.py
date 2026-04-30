from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

FAULT_OVERLOAD = "overload"
FAULT_SOLAR_DROPOUT = "solar_dropout"
FAULT_GRID_OUTAGE = "grid_outage"
FAULT_PLANNED_MAINTENANCE = "planned_maintenance"


@dataclass
class FaultEvent:
    name: str
    event_type: str
    start: datetime
    end: datetime
    target: str = "all"
    params: dict[str, Any] | None = None

    def is_active(self, timestamp: datetime) -> bool:
        return self.start <= timestamp < self.end


@dataclass
class FaultApplication:
    load_kw: np.ndarray
    pv_kw: np.ndarray
    grid_available: bool
    islanding_allowed: bool
    maintenance_mode: bool
    active_faults: list[str]


class FaultLibrary:
    @staticmethod
    def overload_event(
        start: datetime,
        end: datetime,
        load_multiplier: float = 1.35,
        target_ratio: float = 0.35,
        name: str = "overload_event",
    ) -> FaultEvent:
        return FaultEvent(
            name=name,
            event_type=FAULT_OVERLOAD,
            start=start,
            end=end,
            target="random_cluster",
            params={"load_multiplier": load_multiplier, "target_ratio": target_ratio},
        )

    @staticmethod
    def solar_dropout_event(
        start: datetime,
        end: datetime,
        drop_fraction: float = 0.8,
        name: str = "solar_dropout",
    ) -> FaultEvent:
        return FaultEvent(
            name=name,
            event_type=FAULT_SOLAR_DROPOUT,
            start=start,
            end=end,
            target="all",
            params={"drop_fraction": drop_fraction},
        )

    @staticmethod
    def grid_outage_event(start: datetime, end: datetime, name: str = "grid_outage") -> FaultEvent:
        return FaultEvent(
            name=name,
            event_type=FAULT_GRID_OUTAGE,
            start=start,
            end=end,
            target="all",
            params={},
        )

    @staticmethod
    def planned_maintenance_event(
        start: datetime,
        end: datetime,
        name: str = "planned_maintenance",
    ) -> FaultEvent:
        return FaultEvent(
            name=name,
            event_type=FAULT_PLANNED_MAINTENANCE,
            start=start,
            end=end,
            target="all",
            params={},
        )


class FaultEngine:
    def __init__(self, events: list[FaultEvent], num_homes: int, random_seed: int = 42) -> None:
        self.events = events
        self.num_homes = num_homes
        self.rng = np.random.default_rng(random_seed)
        self._target_cache: dict[str, np.ndarray] = {}

    def _event_targets(self, event: FaultEvent) -> np.ndarray:
        if event.target == "all":
            return np.arange(self.num_homes, dtype=np.int64)

        if event.name in self._target_cache:
            return self._target_cache[event.name]

        params = event.params or {}
        target_ratio = float(params.get("target_ratio", 0.35))
        count = max(1, int(round(target_ratio * self.num_homes)))
        selected = self.rng.choice(self.num_homes, size=count, replace=False)
        selected.sort()
        self._target_cache[event.name] = selected
        return selected

    def apply(
        self,
        timestamp: datetime,
        base_load_kw: np.ndarray,
        base_pv_kw: np.ndarray,
    ) -> FaultApplication:
        load_kw = base_load_kw.copy()
        pv_kw = base_pv_kw.copy()

        grid_available = True
        islanding_allowed = True
        maintenance_mode = False
        active_faults: list[str] = []

        for event in self.events:
            if not event.is_active(timestamp):
                continue

            active_faults.append(event.event_type)
            params = event.params or {}

            if event.event_type == FAULT_OVERLOAD:
                targets = self._event_targets(event)
                multiplier = float(params.get("load_multiplier", 1.35))
                load_kw[targets] *= multiplier
            elif event.event_type == FAULT_SOLAR_DROPOUT:
                targets = self._event_targets(event)
                drop_fraction = float(params.get("drop_fraction", 0.8))
                pv_kw[targets] *= max(0.0, 1.0 - drop_fraction)
            elif event.event_type == FAULT_GRID_OUTAGE:
                grid_available = False
                islanding_allowed = True
            elif event.event_type == FAULT_PLANNED_MAINTENANCE:
                grid_available = False
                islanding_allowed = False
                maintenance_mode = True

        return FaultApplication(
            load_kw=load_kw,
            pv_kw=pv_kw,
            grid_available=grid_available,
            islanding_allowed=islanding_allowed,
            maintenance_mode=maintenance_mode,
            active_faults=active_faults,
        )
