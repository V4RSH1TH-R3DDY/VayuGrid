from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from .anomaly import detector
from .breach import notifier
from .config import settings
from .db import fetch_all, pool
from .rate_limit import limiter
from .routers import admin, auth, community, dashboards, nodes, privacy, signals, trading
from .security import UserClaims, decode_access_token, require_roles

app = FastAPI(title="VayuGrid API", version="0.1.0")

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api")
app.include_router(admin.router, prefix="/api")
app.include_router(nodes.router, prefix="/api")
app.include_router(signals.router, prefix="/api")
app.include_router(trading.router, prefix="/api")
app.include_router(privacy.router, prefix="/api")
app.include_router(community.router, prefix="/api")
app.include_router(dashboards.router, prefix="/api")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/security/status")
def security_status(_: UserClaims = Depends(require_roles(["operator"]))) -> dict:
    return {
        "anomaly_detector_trained": detector.is_trained,
        "recent_events": notifier.get_recent_events(limit=20),
    }


def _live_snapshot() -> dict:
    telemetry = fetch_all(
        """
        SELECT ts, node_id, node_type, battery_soc_kwh, solar_output_kw,
               household_load_kw, ev_charge_kw, net_grid_kw, voltage_pu
        FROM node_telemetry
        ORDER BY ts DESC
        LIMIT 50
        """
    )
    trades = fetch_all(
        """
        SELECT trade_id, ts, buyer_node_id, seller_node_id, quantity_kwh,
               cleared_price_inr_per_kwh, status
        FROM trade_records
        ORDER BY ts DESC
        LIMIT 50
        """
    )
    predictions = fetch_all(
        """
        SELECT ts, transformer_id, transformer_loading_pu, max_branch_loading_pu
        FROM transformer_readings
        ORDER BY ts DESC
        LIMIT 50
        """
    )
    gnn_predictions = []
    for row in predictions:
        loading = row.get("transformer_loading_pu") or 0
        overload_probability = 1 / (1 + pow(2.71828, -12 * (loading - 1.0))) if loading else 0
        gnn_predictions.append(
            {
                "ts": row.get("ts"),
                "transformer_id": row.get("transformer_id"),
                "loading_pu": loading,
                "max_branch_loading_pu": row.get("max_branch_loading_pu"),
                "overload_probability": round(overload_probability, 4),
            }
        )
    return {
        "ts": datetime.now(timezone.utc),
        "telemetry": telemetry,
        "trade_flow": trades,
        "gnn_predictions": gnn_predictions,
    }


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket) -> None:
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=1008)
        return
    try:
        decode_access_token(token)
    except Exception:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    try:
        while True:
            snapshot = await asyncio.to_thread(_live_snapshot)
            await websocket.send_json(jsonable_encoder(snapshot))
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        return


async def _breach_notification_loop() -> None:
    while True:
        await asyncio.sleep(900)  # 15 minutes
        try:
            await asyncio.to_thread(notifier.notify_pending)
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning("Breach notification loop error: %s", exc)


@app.on_event("startup")
async def startup() -> None:
    await asyncio.to_thread(pool.open, wait=True, timeout=30)

    # Train anomaly detector from existing data (non-fatal if DB has no data yet)
    try:
        detector.fit_from_db()
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning("Anomaly detector training failed at startup: %s", exc)

    # Start the breach notification background loop
    asyncio.get_event_loop().create_task(_breach_notification_loop())


@app.on_event("shutdown")
def shutdown() -> None:
    pool.close()
