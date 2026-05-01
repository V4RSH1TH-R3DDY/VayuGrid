# Phase 6 API Endpoints

Base URL: `http://localhost:8000`

## Auth

- `POST /api/auth/login`
  - Body: `{ "username": "tony", "password": "operator" }`
  - Returns: `{ "access_token": "...", "token_type": "bearer", "role": "operator", "node_id": null }`

## Operator Dashboard

- `GET /api/dashboard/operator/overview`
  - Returns: KPI summary, grid health, duck curve series, risk timeline, signal history.

- `POST /api/signals`
  - Body:
    ```/dev/null/request.json#L1-6
    {
      "signal_type": "THROTTLE",
      "severity": 0.7,
      "target_node_ids": [1, 2, 3],
      "reason": "Manual override"
    }
    ```

## Homeowner Dashboard

- `GET /api/dashboard/homeowner/{node_id}/summary`
  - Returns: Today’s energy flow, EV status, battery health, earnings, live market.

- `GET /api/privacy/consent/{node_id}`
- `POST /api/privacy/consent/{node_id}`
  - Body:
    ```/dev/null/request.json#L1-5
    {
      "consented": true,
      "consent_version": "v1",
      "categories": ["telemetry", "market", "device"]
    }
    ```

- `GET /api/privacy/export/{node_id}`
- `POST /api/privacy/delete/{node_id}`

## Community Dashboard

- `GET /api/dashboard/community/summary`
- `POST /api/community/critical-load`
  - Body:
    ```/dev/null/request.json#L1-5
    {
      "node_id": 12,
      "priority_tier": "medical",
      "reason": "Critical medical equipment"
    }
    ```

## Node Ingestion (API Key)

- `POST /api/nodes/telemetry`
- `POST /api/nodes/transformer-readings`
- `POST /api/nodes/trades`

## Admin

- `POST /api/admin/nodes/{node_id}/api-key`
  - Returns: `{ "node_id": 12, "api_key": "..." }`

## WebSocket

- `GET /ws/stream?token=JWT`
  - Streams live telemetry, trades, and transformer risk every 5 seconds.
