from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from psycopg.types.json import Json

from ..db import execute
from ..rate_limit import limiter
from ..schemas import SignalIn
from ..security import UserClaims, require_roles

router = APIRouter(tags=["signals"])


@router.post("/signals")
@limiter.limit("100/minute")
def create_signal(payload: SignalIn, _: UserClaims = Depends(require_roles(["operator"]))) -> dict:
    signal_id = str(uuid.uuid4())
    execute(
        """
        INSERT INTO signal_history (
            signal_id, ts, signal_type, severity, target_node_ids, reason, metadata
        ) VALUES (%s, now(), %s, %s, %s, %s, %s)
        """,
        (
            signal_id,
            payload.signal_type,
            payload.severity,
            Json(payload.target_node_ids),
            payload.reason,
            Json(payload.metadata or {}),
        ),
    )
    return {"signal_id": signal_id}
