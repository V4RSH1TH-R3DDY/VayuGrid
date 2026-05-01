from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from .config import settings
from .db import fetch_all, pool
from .rate_limit import limiter
from .routers import admin, auth, community, dashboards, nodes, privacy, signals
from .security import decode_access_token

app = FastAPI(title="VayuGrid API", version="0.1.0")

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
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
app.include_router(privacy.router, prefix="/api")
app.include_router(community.router, prefix="/api")
app.include_router(dashboards.router, prefix="/api")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


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


@app.on_event("shutdown")
def shutdown() -> None:
    pool.close()
