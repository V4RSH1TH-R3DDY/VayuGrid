# Phase 4 Mesh and P2P Trading

Base URL: `http://localhost:8000`

## Runtime

- `matching-engine` runs every `MATCHING_INTERVAL_SECONDS` seconds, default `10`.
- Orders expire after `60` seconds.
- One order cannot exceed `5 kWh`.
- Nodes are rate-limited to `10` orders per minute.
- Island signals suspend the market; resume signals restore normal operation.

## Node Endpoints

- `POST /api/trading/orders`
  - Auth: `X-Api-Key`
  - Body:
    ```json
    {
      "side": "sell",
      "quantity_kwh": 2.0,
      "limit_price_inr_per_kwh": 8.0
    }
    ```

- `POST /api/trading/orders/{order_id}/cancel`
  - Auth: `X-Api-Key`

## Dashboard and Operator Endpoints

- `GET /api/trading/market`
  - Returns market mode, stress score, current floor/cap, and open liquidity.

- `GET /api/trading/orders?status=open`
  - Operator sees all orders.
  - Homeowner sees only their node's orders.

- `POST /api/trading/settle`
  - Operator-only manual settlement trigger.

- `GET /api/trading/ledger/monthly-summary`
  - Returns buyer/seller monthly kWh and INR totals plus latest ledger hash.

- `GET /api/trading/ledger/verify`
  - Operator-only hash-chain integrity check.

## Mesh Contracts

Local development uses RabbitMQ topic routing with the same logical messages the
future libp2p node mesh will gossip:

- `mesh.trade_order`
- `mesh.heartbeat`
- `mesh.state_update`
- `mesh.signal`
- `mesh.settlement`

Message envelopes carry `ttl_hops = 3`, matching the Phase 4 neighborhood gossip
design for up to roughly 200 homes.
