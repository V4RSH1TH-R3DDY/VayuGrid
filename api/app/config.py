from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List


def _load_users() -> Dict[str, Dict[str, Any]]:
    raw = os.getenv("DASHBOARD_USERS_JSON")
    if raw:
        return json.loads(raw)
    return {
        "tony": {"password": "operator", "role": "operator"},
        "reggie": {"password": "homeowner", "role": "homeowner", "node_id": 1},
        "luigi": {"password": "community", "role": "community"},
    }


@dataclass
class Settings:
    database_url: str = os.getenv(
        "DATABASE_URL", "postgresql://vayugrid:vayugrid@localhost:5432/vayugrid"
    )
    jwt_secret: str = os.getenv("JWT_SECRET", "change-me")
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "480"))
    cors_origins: List[str] = field(
        default_factory=lambda: [
            origin.strip()
            for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
            if origin.strip()
        ]
    )
    utility_rate_inr_per_kwh: float = float(os.getenv("UTILITY_RATE_INR_PER_KWH", "8"))
    net_metering_rate_inr_per_kwh: float = float(os.getenv("NET_METERING_RATE_INR_PER_KWH", "3"))
    dashboard_users: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.dashboard_users = _load_users()


settings = Settings()
