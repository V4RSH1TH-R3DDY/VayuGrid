from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    node_id: int | None = None


class TelemetryIn(BaseModel):
    ts: datetime
    node_id: int
    node_type: str
    battery_soc_kwh: float | None = None
    battery_power_kw: float | None = None
    solar_output_kw: float | None = None
    household_load_kw: float | None = None
    ev_charge_kw: float | None = None
    net_grid_kw: float | None = None
    voltage_pu: float | None = None
    metadata: Dict[str, Any] | None = None


class TransformerReadingIn(BaseModel):
    ts: datetime
    transformer_id: str
    feeder_total_kw: float
    transformer_loading_pu: float
    max_branch_loading_pu: float
    hottest_spot_temp_c: float
    aging_acceleration: float
    grid_available: bool
    islanding_triggered: bool
    maintenance_mode: bool
    metadata: Dict[str, Any] | None = None


class TradeRecordIn(BaseModel):
    trade_id: str
    ts: datetime
    buyer_node_id: int
    seller_node_id: int
    quantity_kwh: float
    cleared_price_inr_per_kwh: float
    status: str
    metadata: Dict[str, Any] | None = None


class SignalIn(BaseModel):
    signal_type: str
    severity: float = Field(ge=0, le=1)
    target_node_ids: List[int]
    reason: str
    metadata: Dict[str, Any] | None = None


class ConsentIn(BaseModel):
    consented: bool
    consent_version: str
    categories: List[str]


class CriticalLoadFlagIn(BaseModel):
    node_id: int
    priority_tier: str = Field(default="medical")
    reason: str = Field(default="Critical medical equipment")
