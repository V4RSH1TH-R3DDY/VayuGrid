from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Request
from psycopg.types.json import Json

from ..db import execute, fetch_all, fetch_one
from ..rate_limit import limiter
from ..schemas import ConsentIn
from ..security import UserClaims, require_roles

router = APIRouter(tags=["privacy"])


def _encrypt_ip(ip: str) -> str:
    from ..security import Encryption

    try:
        return Encryption.encrypt(ip)
    except Exception:
        return ip  # fall back to plaintext if encryption fails


@router.get("/privacy/consent/{node_id}")
@limiter.limit("100/minute")
def get_consent(
    request: Request,
    node_id: int,
    _: UserClaims = Depends(require_roles(["operator", "homeowner"])),
) -> dict:
    record = fetch_one(
        """
        SELECT node_id, consented, consent_version, categories, updated_at
        FROM household_consents
        WHERE node_id = %s
        ORDER BY updated_at DESC
        LIMIT 1
        """,
        (node_id,),
    )
    return record or {
        "node_id": node_id,
        "consented": False,
        "categories": [],
        "consent_version": "",
    }


@router.post("/privacy/consent/{node_id}")
@limiter.limit("100/minute")
def set_consent(
    node_id: int,
    payload: ConsentIn,
    request: Request,
    _: UserClaims = Depends(require_roles(["operator", "homeowner"])),
) -> dict:
    execute(
        """
        INSERT INTO household_consents (
            node_id, consented, consent_version, categories, ip_address, updated_at
        ) VALUES (%s, %s, %s, %s, %s, now())
        """,
        (
            node_id,
            payload.consented,
            payload.consent_version,
            Json(payload.categories),
            (_encrypt_ip(request.client.host) if request.client and request.client.host else None),
        ),
    )
    return {"status": "recorded"}


@router.get("/privacy/export/{node_id}")
@limiter.limit("20/minute")
def export_household_data(
    request: Request,
    node_id: int,
    _: UserClaims = Depends(require_roles(["operator", "homeowner"])),
) -> dict:
    telemetry = fetch_all(
        "SELECT * FROM node_telemetry WHERE node_id = %s ORDER BY ts",
        (node_id,),
    )
    trades = fetch_all(
        """
        SELECT * FROM trade_records
        WHERE buyer_node_id = %s OR seller_node_id = %s
        ORDER BY ts
        """,
        (node_id, node_id),
    )
    consents = fetch_all(
        "SELECT * FROM household_consents WHERE node_id = %s ORDER BY updated_at",
        (node_id,),
    )
    for record in consents:
        raw_ip = record.get("ip_address")
        if raw_ip:
            try:
                from ..security import Encryption

                record["ip_address"] = Encryption.decrypt(raw_ip)
            except Exception:
                pass  # old unencrypted data — leave as-is
    flags = fetch_all(
        "SELECT * FROM critical_load_flags WHERE node_id = %s ORDER BY created_at",
        (node_id,),
    )
    return {
        "node_id": node_id,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "telemetry": telemetry,
        "trades": trades,
        "consents": consents,
        "critical_load_flags": flags,
    }


@router.post("/privacy/delete/{node_id}")
@limiter.limit("20/minute")
def request_deletion(
    request: Request,
    node_id: int,
    _: UserClaims = Depends(require_roles(["operator", "homeowner"])),
) -> dict:
    request_id = f"del_{node_id}_{int(datetime.now(timezone.utc).timestamp())}"
    scheduled_for = datetime.now(timezone.utc) + timedelta(hours=72)
    execute(
        """
        INSERT INTO data_deletion_requests (
            request_id, node_id, requested_at, scheduled_for, status, metadata
        ) VALUES (%s, %s, now(), %s, %s, %s)
        """,
        (request_id, node_id, scheduled_for, "scheduled", Json({})),
    )
    return {"request_id": request_id, "scheduled_for": scheduled_for.isoformat()}
