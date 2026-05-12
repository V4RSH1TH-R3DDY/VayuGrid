from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from api.app.main import app
from api.app.security import create_access_token

client = TestClient(app)


def _token_for_user(username: str, role: str, node_id: int | None = None) -> str:
    data: dict = {"sub": username, "role": role}
    if node_id is not None:
        data["node_id"] = node_id
    return create_access_token(data)


_OP_TOKEN = _token_for_user("tony", "operator")


def test_health() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


class TestAuth:
    def test_login_operator_success(self) -> None:
        resp = client.post(
            "/api/auth/login",
            json={"username": "tony", "password": "operator"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["token_type"] == "bearer"
        assert body["role"] == "operator"
        assert body["node_id"] is None
        assert len(body["access_token"]) > 0

    def test_login_homeowner_success(self) -> None:
        resp = client.post(
            "/api/auth/login",
            json={"username": "reggie", "password": "homeowner"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["role"] == "homeowner"
        assert body["node_id"] == 1

    def test_login_invalid_credentials(self) -> None:
        resp = client.post(
            "/api/auth/login",
            json={"username": "tony", "password": "wrong"},
        )
        assert resp.status_code == 401

    def test_login_unknown_user(self) -> None:
        resp = client.post(
            "/api/auth/login",
            json={"username": "nobody", "password": "x"},
        )
        assert resp.status_code == 401

    def test_login_rate_limit(self) -> None:
        for _ in range(20):
            client.post(
                "/api/auth/login",
                json={"username": "x", "password": "y"},
            )
        resp = client.post(
            "/api/auth/login",
            json={"username": "x", "password": "y"},
        )
        assert resp.status_code == 429

    def test_security_status_no_token_returns_401(self) -> None:
        resp = client.get("/api/security/status")
        assert resp.status_code == 401


class TestOperatorDashboard:
    AUTH_HEADER = {"Authorization": f"Bearer {_OP_TOKEN}"}

    @patch("api.app.routers.dashboards.fetch_all", return_value=[])
    @patch("api.app.routers.dashboards.fetch_one", return_value=None)
    def test_operator_overview_empty_db(
        self, mock_fetch_one: MagicMock, mock_fetch_all: MagicMock
    ) -> None:
        resp = client.get("/api/dashboard/operator/overview", headers=self.AUTH_HEADER)
        assert resp.status_code == 200
        body = resp.json()
        assert body["kpis"]["overload_events"] == 0
        assert body["grid_health"] == []

    @patch("api.app.routers.dashboards.fetch_all")
    @patch("api.app.routers.dashboards.fetch_one")
    def test_operator_overview_with_data(
        self, mock_fetch_one: MagicMock, mock_fetch_all: MagicMock
    ) -> None:
        now = datetime.now(timezone.utc)
        mock_fetch_all.side_effect = [
            [{"node_id": 1, "node_type": "home", "battery_soc_kwh": 10.0,
              "solar_output_kw": 3.0, "household_load_kw": 1.5,
              "net_grid_kw": -1.5, "voltage_pu": 0.98, "ts": now}],
            [{"bucket": now, "net_grid_kw": 5.0}],
            [{"ts": now, "transformer_id": "xfmr-1",
              "transformer_loading_pu": 0.85, "max_branch_loading_pu": 0.7}],
            [],
            [{"solar_output_kw": 3.0, "household_load_kw": 1.5,
              "ev_charge_kw": 0.0, "battery_power_kw": 1.0,
              "net_grid_kw": -1.5}],
            [{"bucket": now, "net_grid_kw": 5.0}],
        ]
        mock_fetch_one.side_effect = [
            {"avg_aging": 1.2},
            {"total_kwh": 100.0, "total_value": 800.0},
            {"count": 2},
        ]

        resp = client.get("/api/dashboard/operator/overview", headers=self.AUTH_HEADER)
        assert resp.status_code == 200
        body = resp.json()
        assert body["kpis"]["transformer_aging_rate"] == 1.2
        assert body["kpis"]["p2p_volume_kwh"] == 100.0
        assert body["kpis"]["overload_events"] == 2
        assert len(body["risk_timeline"]) == 1
        assert body["risk_timeline"][0]["transformer_id"] == "xfmr-1"

    @patch("api.app.routers.dashboards.fetch_all", return_value=[])
    @patch("api.app.routers.dashboards.fetch_one", return_value={})
    def test_homeowner_summary_empty(
        self, mock_fetch_one: MagicMock, mock_fetch_all: MagicMock
    ) -> None:
        resp = client.get("/api/dashboard/homeowner/1/summary", headers=self.AUTH_HEADER)
        assert resp.status_code == 200

    def test_homeowner_summary_forbidden_wrong_node(self) -> None:
        token = _token_for_user("reggie", "homeowner", node_id=1)
        resp = client.get(
            "/api/dashboard/homeowner/2/summary",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 403


class TestWebSocket:
    def test_ws_stream_rejects_missing_token(self) -> None:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws/stream", None):
                pass

    def test_ws_stream_rejects_bad_token(self) -> None:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws/stream?token=badtoken", None):
                pass
