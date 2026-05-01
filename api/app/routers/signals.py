from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from psycopg.types.json import Json

from ai.schemas import NeighborhoodSignal, NeighborhoodSignalType
from trading.pricing import PricingPolicy

from ..db import execute
from ..rate_limit import limiter
from ..schemas import SignalIn
from ..security import UserClaims, require_roles

router = APIRouter(tags=["signals"])


@router.post("/signals")
@limiter.limit("100/minute")
def create_signal(
    request: Request, payload: SignalIn, _: UserClaims = Depends(require_roles(["operator"]))
) -> dict:
    signal_id = str(uuid.uuid4())
    try:
        signal_type = NeighborhoodSignalType(payload.signal_type.lower())
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Unsupported signal_type",
        ) from exc

    execute(
        """
        INSERT INTO signal_history (
            signal_id, ts, signal_type, severity, target_node_ids, reason,
            recommended_price_floor_inr_per_kwh, recommended_price_cap_inr_per_kwh,
            expires_at, metadata
        ) VALUES (%s, now(), %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            signal_id,
            payload.signal_type.lower(),
            payload.severity,
            Json(payload.target_node_ids),
            payload.reason,
            payload.recommended_price_floor_inr_per_kwh,
            payload.recommended_price_cap_inr_per_kwh,
            payload.expires_at,
            Json(payload.metadata or {}),
        ),
    )
    signal = NeighborhoodSignal(
        signal_id=signal_id,
        timestamp=datetime.now(timezone.utc),
        signal_type=signal_type,
        severity=payload.severity,
        target_node_ids=payload.target_node_ids,
        reason=payload.reason,
        recommended_price_floor_inr_per_kwh=payload.recommended_price_floor_inr_per_kwh,
        recommended_price_cap_inr_per_kwh=payload.recommended_price_cap_inr_per_kwh,
        expires_at=payload.expires_at,
        metadata=payload.metadata or {},
    )
    market_state = PricingPolicy().from_signal(signal)
    execute(
        """
        UPDATE market_state
        SET mode = %s, stress_score = %s, price_floor_inr_per_kwh = %s,
            price_cap_inr_per_kwh = %s, accepts_orders = %s, reason = %s, updated_at = now()
        WHERE id = TRUE
        """,
        (
            market_state.mode.value,
            market_state.stress_score,
            market_state.price_floor_inr_per_kwh,
            market_state.price_cap_inr_per_kwh,
            market_state.accepts_orders,
            market_state.reason,
        ),
    )
    return {"signal_id": signal_id, "market_state": market_state.to_dict()}
