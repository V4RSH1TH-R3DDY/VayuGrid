from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from psycopg.types.json import Json

from ..db import execute
from ..rate_limit import limiter
from ..schemas import CriticalLoadFlagIn
from ..security import UserClaims, require_roles

router = APIRouter(tags=["community"])


@router.post("/community/critical-load")
@limiter.limit("100/minute")
def flag_critical_load(
    payload: CriticalLoadFlagIn,
    _: UserClaims = Depends(require_roles(["operator", "community", "homeowner"])),
) -> dict:
    flag_id = str(uuid.uuid4())
    execute(
        """
        INSERT INTO critical_load_flags (
            flag_id, node_id, priority_tier, reason, active, created_at, metadata
        ) VALUES (%s, %s, %s, %s, TRUE, now(), %s)
        """,
        (flag_id, payload.node_id, payload.priority_tier, payload.reason, Json({})),
    )
    return {"flag_id": flag_id}
