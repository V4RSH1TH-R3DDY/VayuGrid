from __future__ import annotations

import secrets

from fastapi import APIRouter, Depends

from ..db import execute
from ..rate_limit import limiter
from ..security import UserClaims, hash_api_key, require_roles

router = APIRouter(tags=["admin"])


@router.post("/admin/nodes/{node_id}/api-key")
@limiter.limit("100/minute")
def create_node_api_key(node_id: int, _: UserClaims = Depends(require_roles(["operator"]))) -> dict:
    api_key = secrets.token_urlsafe(32)
    key_hash = hash_api_key(api_key)
    execute(
        """
        INSERT INTO node_api_keys (node_id, key_hash, active, created_at)
        VALUES (%s, %s, TRUE, now())
        """,
        (node_id, key_hash),
    )
    return {"node_id": node_id, "api_key": api_key}
