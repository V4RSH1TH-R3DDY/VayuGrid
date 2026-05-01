# VayuGrid

A smart energy grid

## Local Infrastructure

The repo now includes a local AI and infrastructure stack aligned with the team
plan:

- TimescaleDB/PostgreSQL for telemetry and trade history
- Redis for live node state caching
- RabbitMQ for message transport
- MLflow for experiment tracking
- Prometheus and Grafana for observability
- FastAPI backend for dashboards and nodes
- React dashboards for operator, homeowner, and community views

Start everything with:

```bash
docker compose up --build
```

Default local endpoints:

- PostgreSQL: `localhost:5432`
- Redis: `localhost:6379`
- RabbitMQ AMQP: `localhost:5672`
- RabbitMQ management: `http://localhost:15672`
- RabbitMQ Prometheus metrics: `localhost:15692/metrics`
- MLflow: `http://localhost:5000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`
- API: `http://localhost:8000` (OpenAPI docs at `/docs`)
- Dashboards: `http://localhost:5173`

The TimescaleDB container bootstraps the initial telemetry, trade, and signal
tables from `infra/postgres/init/01-timescaledb.sql`.

## Phase 6 API & Dashboards

The FastAPI service exposes dashboard and node endpoints under `/api`, plus a
WebSocket stream at `/ws/stream` (every 5 seconds). See `docs/api_phase6.md` for
endpoint details.

## Phase 4 Mesh & P2P Trading

The repo now includes the Phase 4 trading layer:

- Continuous Double Auction order matching every 10 seconds
- Dynamic GNN-driven market floor/cap state
- Island-mode market suspension
- RabbitMQ mesh message contracts for local development
- Hash-linked Aya Wallet Ledger records

See `docs/api_phase4.md` for endpoint and runtime details.

Default demo users (override with `DASHBOARD_USERS_JSON` in the API container):
- Operator: `tony / operator`
- Homeowner: `reggie / homeowner`
- Community: `luigi / community`

Node API keys can be minted via `POST /api/admin/nodes/{node_id}/api-key` using an
operator JWT.

## Python Tooling

Install the local project and developer tools with:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .[dev]
```

Then run:

```bash
ruff check .
mypy
pytest
```

## AI Data Contracts

The first AI-facing schema module lives in `ai/schemas.py` and defines:

- `NodeState`
- `TradeOrder`
- `GridTelemetry`
- `NeighborhoodSignal`

## Grid Simulator

The main simulator lives under `simulator/` and is ready for minute-resolution
training runs.

- Package guide:
  [simulator/README.md](/home/varshith/VayuGrid/simulator/README.md)
- Scenario presets:
  `scenarios/phase1_debug.json`,
  `scenarios/phase1_default.json`,
  `scenarios/phase1_stress.json`
- Example run:

```bash
/home/varshith/VayuGrid/.venv/bin/python examples/example_instantiation.py
```

## Pecan Data Wire-Up

Use `pecan_data_wireup.py` to convert raw Pecan Street 1-minute files into
simulator-ready household profiles and optionally replace PV with city-specific
NSRDB solar.

### What it produces

- Minute-level output per household with:
  - `timestamp_ist`
  - `home_id`
  - `load_kw`
  - `pv_kw`
  - `ev_kw`
  - `battery_kw`
  - `grid_kw`
  - `source_region`
- Output files:
  - `data/processed/pecan_india/<city>/<year>/pecan_wired_<city>_<year>.csv`
  - `data/processed/pecan_india/<city>/<year>/pecan_wired_<city>_<year>.parquet`
  - `data/processed/pecan_india/<city>/<year>/pecan_wired_<city>_<year>_summary.csv`

### Example

```bash
/home/varshith/VayuGrid/.venv/bin/python pecan_data_wireup.py \
  --city bangalore \
  --year 2019 \
  --source-regions austin,california,newyork \
  --max-homes 150 \
  --target-kwh-per-day 6.5 \
  --replace-solar-with-nsrdb
```

### Background batch run (all target cities)

```bash
for city in bangalore chennai kochi hyderabad delhi; do
  /home/varshith/VayuGrid/.venv/bin/python pecan_data_wireup.py \
    --city "$city" \
    --year 2019 \
    --source-regions austin,california,newyork \
    --max-homes 150 \
    --target-kwh-per-day 6.5 \
    --replace-solar-with-nsrdb
done
```
