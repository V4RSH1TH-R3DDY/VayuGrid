# VayuGrid — Project Progress Report

> **Team:** Coder4not4 | **Hackathon:** AI4India
> **Last updated:** See file commit date

---

## Table of Contents

1. [Overview](#overview)
2. [What Has Been Done](#what-has-been-done)
   - [Phase 0 — Infrastructure & Setup](#phase-0--infrastructure--setup)
   - [Phase 1 — Grid Simulator](#phase-1--grid-simulator)
   - [Phase 4 — Mesh Network & P2P Trading Layer](#phase-4--mesh-network--p2p-trading-layer)
   - [Phase 5 — Security & Privacy (Partial)](#phase-5--security--privacy-partial)
   - [Phase 6 — API Layer & Dashboards](#phase-6--api-layer--dashboards)
   - [Phase 7 — Real Data Integration (Partial)](#phase-7--real-data-integration-partial)
   - [Baseline Protocol & Success Metrics](#baseline-protocol--success-metrics)
   - [Documentation](#documentation)
   - [Tests](#tests)
3. [What Still Needs to Be Done](#what-still-needs-to-be-done)
   - [Phase 2 — CortexCore AI Agent (Not Started)](#phase-2--cortexcore-ai-agent-not-started)
   - [Phase 3 — VayuGNN Neighborhood Brain (Not Started)](#phase-3--vayugnn-neighborhood-brain-not-started)
   - [Baseline Framework (Not Started)](#baseline-framework-not-started)
   - [Phase 7 — Real Data Integration (Remaining)](#phase-7--real-data-integration-remaining)
   - [Phase 8 — Integration & Regression Testing (Not Started)](#phase-8--integration--regression-testing-not-started)
   - [Phase 9 — Deployment (Not Started)](#phase-9--deployment-not-started)
   - [Phase 10 — Scale & Federation (Not Started)](#phase-10--scale--federation-not-started)
   - [Security Hardening (Remaining)](#security-hardening-remaining)
   - [Missing Feature Wiring](#missing-feature-wiring)
4. [At-a-Glance Status Table](#at-a-glance-status-table)
5. [Critical Path & Handoffs](#critical-path--handoffs)

---

## Overview

VayuGrid is a smart residential energy grid system that enables peer-to-peer (P2P) energy trading, AI-driven home energy management, and neighborhood-level grid coordination for Indian urban communities. The project spans three domains:

- **Simulation & Data** (Varshith) — the virtual training environment
- **AI & Machine Learning** (Charithra) — the edge PPO agent (CortexCore) and GNN (VayuGNN)
- **Infrastructure, Mesh & Trading** (Mukul) — everything the system runs on
- **Product, Security & Dashboards** (Anjali) — what users see and what keeps them safe

The build plan has **10 phases** over ~14 weeks. As of now, **Phases 0, 1, 4, and 6 are complete or nearly complete**, with partial work across Phases 5 and 7. Phases 2, 3, 8, 9, and 10 have not been started.

---

## What Has Been Done

### Phase 0 — Infrastructure & Setup

**Status: ✅ Complete**

The full local development environment is containerized and reproducible.

#### Services (Docker Compose)
| Service | Image / Build | Port | Purpose |
|---|---|---|---|
| PostgreSQL + TimescaleDB | `timescale/timescaledb:latest-pg16` | 5432 | Primary time-series store |
| Redis | `redis:7-alpine` | 6379 | Live node-state cache |
| RabbitMQ | `rabbitmq:3.13-management` | 5672 / 15672 / 15692 | Mesh message bus |
| MLflow | Custom build | 5000 | Experiment tracking |
| Prometheus | `prom/prometheus:v2.54.1` | 9090 | Metrics scraping |
| Grafana | `grafana/grafana:11.1.4` | 3000 | Observability dashboards |
| FastAPI backend | Custom build | 8000 | REST API + WebSocket |
| Matching engine | Custom build | — | Standalone settlement service |
| React frontend | Custom build (Vite) | 5173 | User-facing dashboards |

All services start with a single `docker compose up --build`. Health checks are defined for every stateful service. The TimescaleDB container auto-bootstraps the schema from `infra/postgres/init/`.

#### AI Data Contracts (`ai/schemas.py`)
Four core dataclasses are fully defined and drive every downstream interface:

| Dataclass | Key Fields |
|---|---|
| `NodeState` | `battery_soc_kwh`, `solar_output_kw`, `household_load_kw`, `ev_charge_kw`, `ev_hours_to_deadline`, `voltage_pu`, `market_buy/sell_price`, `active_signal`, 15-minute forecasts |
| `TradeOrder` | `order_id`, `node_id`, `side` (BUY/SELL), `quantity_kwh`, `limit_price_inr_per_kwh`, `expires_at`, `status`, `counterparty_node_id`, `cleared_price_inr_per_kwh` |
| `GridTelemetry` | `transformer_id`, `feeder_total_kw`, `transformer_loading_pu`, `hottest_spot_temp_c`, `aging_acceleration`, `islanding_triggered`, `maintenance_mode` |
| `NeighborhoodSignal` | `signal_type` (THROTTLE / PRE\_COOL / ISLAND / RESUME), `severity`, `target_node_ids`, `recommended_price_floor/cap_inr_per_kwh`, `expires_at` |

#### Python Tooling
- **Linting:** `ruff` (line-length 100, targets E/F/I/B rules)
- **Type-checking:** `mypy` (strict, covers all packages)
- **Tests:** `pytest` (test suite under `tests/`)
- **Package:** Installed as an editable package (`pip install -e .[dev]`) with `pyproject.toml`

---

### Phase 1 — Grid Simulator

**Status: ✅ Complete**

The simulator (`simulator/`) is a fully functional minute-resolution residential grid simulator. It is the most critical deliverable in the project — everything AI-related depends on it.

#### Core Engine (`simulator/simulator.py`)
- `GridSimulator` class accepts a `SimulatorConfig` and runs a time-stepped simulation
- Returns a `SimulationResult` with three Parquet-ready DataFrames:
  - `node_timeseries` — per-node, per-minute: battery SoC, solar output, load, EV, voltage, branch flow
  - `transformer_timeseries` — per-minute: loading pu, thermal state, aging, fault events
  - `event_log` — timestamped fault activations and island events
- Supports deterministic random seeds for reproducible training runs

#### Configuration System (`simulator/config.py`)
- `SimulatorConfig` — top-level config with full `validate()` guard
- `NeighborhoodConfig` — num\_homes (10–500), line resistance, ampacity, voltage
- `AdoptionConfig` — solar/battery/EV penetration ratios
- `LoadProfileConfig` — India city profiles, Pecan/NSRDB switches, festival and cricket dates
- `TransformerConfig` — all IEEE C57.91 thermal parameters
- `FaultConfig` — per-event type/start/end/target/params
- Configs load from JSON files via `load_simulator_config()`

#### Electrical Graph Model (`simulator/graph.py`)
- `ResidentialFeederGraph` — radial feeder topology with a single transformer root
- Random radial graph builder: homes connect to recent predecessors (local cluster behavior)
- Battery nodes attached as children of their home nodes
- `compute_network_state()` method propagates branch flows up the tree, then computes:
  - Per-home voltage (pu) with resistive voltage drop
  - Branch loading (pu vs. ampacity)
  - Max branch loading across the feeder

#### Load Profile Library (`simulator/load_profiles.py`)
India-specific synthetic load generation fully implemented for 6 cities:

| City | Base kWh/day | Summer Temp | AC Gain | Evening Peak |
|---|---|---|---|---|
| Bangalore | 6.0 | 32°C | 0.85× | 1.00× |
| Chennai | 7.4 | 37°C | 1.28× | 0.95× |
| Hyderabad | 7.0 | 36°C | 1.18× | 1.02× |
| Delhi | 7.8 | 39°C | 1.35× | 1.08× |
| Kochi | 5.8 | 33°C | 0.92× | 1.04× |
| Mumbai | 6.7 | 34°C | 1.00× | 1.03× |

Features include:
- Morning, afternoon AC, evening cooking/lighting peaks modelled with Gaussian kernels
- Festival spike overlays (configurable dates)
- Cricket match evening surge (weekends + configured dates)
- Weekend baseline uplift (+8%)
- Per-home Gaussian occupancy noise (σ = 0.06) + load noise (σ = 0.08)
- **Pecan Street integration**: reads processed CSV profiles, applies India daily-energy scaling, then passes through India-specific shaping (AC/evening/festival boosts) while preserving total kWh
- **NSRDB solar integration**: loads city-specific GHI CSVs, normalizes, and replaces synthetic PV curves
- **India EV profiles**: arrival-hour modelled via city-specific Gaussian (σ = 1.1 hr), overnight-biased charging with next-day spillover

#### Fault Injection (`simulator/faults.py`)
Four fault types fully implemented via `FaultLibrary` + `FaultEngine`:

| Fault Type | Behaviour |
|---|---|
| `overload` | Multiplies load by ≥1.35× on a random cluster (configurable ratio) |
| `solar_dropout` | Drops PV output by 80% (configurable) on affected homes |
| `grid_outage` | Sets `grid_available = False`, allows islanding |
| `planned_maintenance` | Sets `grid_available = False`, **disables** islanding (critical distinction for AI safety) |

- Target caching ensures the same random cluster is hit throughout an event window
- Events are driven by `FaultConfig` entries in the scenario JSON

#### IEEE Transformer Thermal Model (`simulator/thermal.py`)
Full C57.91 implementation:
- Top-oil temperature rise (exponential lag with `tau_to = 180 min`)
- Hotspot rise (fast lag with `tau_w = 10 min`)
- Aging acceleration factor: `exp(15000/383 − 15000/(T_hs + 273))` — doubles for every ~8°C above reference
- Cumulative loss-of-life hours tracked per run

#### Scenario Presets (`scenarios/`)
| File | Homes | Duration | Notes |
|---|---|---|---|
| `phase1_debug.json` | 10 | 2 hours | Fast iteration |
| `phase1_default.json` | 30 | 24 hours | Standard training episode |
| `phase1_stress.json` | 100 | 48 hours | High-penetration stress test |

#### Handoff Artifacts
- `simulator/README.md` — complete package guide
- `docs/scenario_configuration_guide.md` — JSON field reference
- `docs/output_schema_definitions.md` — all output column definitions
- `examples/example_instantiation.py` — runnable end-to-end example

---

### Phase 4 — Mesh Network & P2P Trading Layer

**Status: ✅ Complete**

The full P2P trading stack is implemented and wired into both the API and the standalone matching service.

#### Order Book (`trading/order_book.py`)
- Maintains sorted bid and ask queues
- Continuous Double Auction: every `settle()` call matches the best bid against the best ask if bid ≥ ask
- Cleared price = midpoint of the two limit prices
- Orders expire after 60 seconds (configurable TTL); expired orders are pruned on each cycle

#### Matching Engine (`trading/engine.py`)
- `MatchingEngine` wraps `OrderBook`, `AyaLedger`, and `MarketState`
- `submit_order()` enforces:
  - Market mode check (`ISLANDED` → reject all orders)
  - Per-node rate limiting: max 10 orders per minute (sliding window via `deque`)
  - Max order size: 5 kWh (anti-manipulation)
  - Price clamping to dynamic floor/cap from `MarketState`
- `settle_once()` runs the match cycle and appends every match to the ledger
- `update_market_state()` — GNN signal drives pricing regime changes

#### Pricing Policy (`trading/pricing.py`)
- `PricingPolicy` translates a `NeighborhoodSignal` into a `MarketState`
- Three regimes:
  - **NORMAL** (severity < 0.55): floor ₹3, cap ₹12
  - **STRESSED** (0.55 ≤ severity < 0.85): floor and cap lifted proportionally
  - **ISLANDED** (ISLAND signal): `accepts_orders = False`, market suspended
  - **RESUME**: resets to normal parameters
- Signal-recommended floor/cap override the defaults if present

#### Aya Ledger (`trading/ledger.py`)
- `AyaLedger` — append-only hash-linked ledger
- Each `LedgerEntry` includes sequence number, trade metadata, `previous_hash`, and `entry_hash` (SHA-256 over canonical JSON)
- `verify()` walks the entire chain and validates hash linkage and sequence ordering
- `GENESIS_HASH` is 64 zero hex digits
- Buyer and seller `Ed25519` signatures are stored per entry

#### Ed25519 Signatures (`trading/signatures.py`)
- `generate_keypair()` — produces raw Ed25519 private/public key pair, base64-encoded
- `sign_payload()` — signs canonical sorted JSON
- `verify_payload()` — verifies with `InvalidSignature` guard

#### DB-Backed Matching (`trading/db_matching.py`)
- `settle_open_orders()` — fetches open orders from PostgreSQL, runs the auction, writes trade records and ledger entries back to the DB
- Used by the standalone `services/matching_engine.py` service (10-second interval)

#### Mesh Message Contracts (`mesh/messages.py`)
- `GossipMessage` — envelope with `MessageKind`, `source_node_id`, `payload`, `created_at`, `ttl_hops` (default 3), `signature`
- `should_forward()` / `forwarded()` — TTL decrement logic
- `TradeOrderPayload`, `HeartbeatPayload` — typed payload schemas for the gossip network
- Five message kinds: `TRADE_ORDER`, `HEARTBEAT`, `STATE_UPDATE`, `SETTLEMENT`, `SIGNAL`

#### RabbitMQ Bus (`mesh/rabbitmq_bus.py`)
- `RabbitMQMeshBus` — async publish adapter using `aio-pika`
- Topic exchange `vayugrid.mesh` with routing key `mesh.<kind>`
- Acts as a **local development stand-in** for the eventual libp2p gossip network
- `encode()` produces deterministic sorted JSON bytes

#### API Endpoints (Trading)
| Endpoint | Method | Auth | Purpose |
|---|---|---|---|
| `/api/trading/market` | GET | operator/homeowner/community | Current market state + liquidity |
| `/api/trading/orders` | POST | node API key | Submit a buy/sell order |
| `/api/trading/orders` | GET | operator/homeowner | List orders (homeowner sees own only) |
| `/api/trading/orders/{id}/cancel` | POST | node API key | Cancel open order |
| `/api/trading/settle` | POST | operator | Manual settlement trigger |
| `/api/trading/ledger/monthly-summary` | GET | operator/community | Monthly billing summary |
| `/api/trading/ledger/verify` | GET | operator | Verify ledger integrity |

---

### Phase 5 — Security & Privacy (Partial)

**Status: 🟡 Partial (~60% complete)**

#### Implemented
- **JWT authentication** — `jose`-backed RS256/HS256 tokens; `create_access_token()` and `decode_access_token()` in `api/app/security.py`
- **Role-based access** — `require_roles(["operator", "homeowner", "community"])` dependency injected on every protected route
- **Node API keys** — SHA-256 hashed keys minted via `POST /api/admin/nodes/{node_id}/api-key`; used to authenticate Vayu-Node order submissions
- **Rate limiting** — SlowAPI middleware with per-route limits (100 req/min for dashboards, 1000/min for node endpoints)
- **DPDP Act compliance endpoints:**
  - `GET /api/privacy/consent/{node_id}` — retrieve consent record
  - `POST /api/privacy/consent/{node_id}` — record consent with IP, version, categories
  - `GET /api/privacy/export/{node_id}` — full data export (telemetry, trades, consents, critical-load flags) as JSON
  - `POST /api/privacy/delete/{node_id}` — schedule data deletion within 72 hours
- **Ed25519 trade signatures** — buyer/seller sign each trade record (implemented in `trading/signatures.py`)
- **CORS middleware** — configurable `cors_origins`

#### Not Yet Implemented
- HashiCorp Vault for key management and 90-day rotation
- Mutual TLS between Vayu-Nodes and the neighborhood server
- AES-256-GCM at-rest database encryption
- Isolation Forest anomaly detector (per-device Safe Mode)
- Wash trade detection in the matching engine
- Replay attack prevention (message timestamp validation window)
- Automated breach notification system (72-hour trigger)
- Database retention policies (auto-delete raw readings after 90 days; keep 15-min aggregates 3 years)
- Data precision anonymization before transmission (bucketing / rounding)

---

### Phase 6 — API Layer & Dashboards

**Status: ✅ Complete**

#### FastAPI Backend (`api/app/`)
- `main.py` — application factory; mounts all routers under `/api`; WebSocket at `/ws/stream`
- `db.py` — psycopg3 connection pool; `fetch_one()`, `fetch_all()`, `execute()` helpers
- `config.py` — Pydantic `Settings` with env-var overrides (DATABASE\_URL, JWT\_SECRET, CORS\_ORIGINS)
- `rate_limit.py` — SlowAPI limiter instance
- `schemas.py` — Pydantic request/response models (TradeOrderIn, SignalIn, ConsentIn, etc.)

**Router summary:**

| Router | Key Endpoints |
|---|---|
| `auth` | `POST /api/auth/login` — returns JWT for username/password |
| `admin` | `POST /api/admin/nodes/{id}/api-key` — mint node key |
| `nodes` | `GET/POST /api/nodes` — node registry |
| `signals` | `POST /api/signals` — broadcast NeighborhoodSignal, update market state |
| `trading` | Full order lifecycle + ledger (see Phase 4) |
| `privacy` | Consent, export, deletion (see Phase 5) |
| `community` | `POST /api/community/critical-load`, community summary |
| `dashboards` | `/api/dashboard/operator/overview`, `/api/dashboard/homeowner/{id}/summary`, `/api/dashboard/community/summary` |

**WebSocket (`/ws/stream`):**
- JWT token required as query param
- Pushes a live snapshot every 5 seconds with: last 50 telemetry rows, last 50 trades, GNN overload probability per transformer
- Transformer overload probability is currently a sigmoid approximation on loading\_pu (placeholder for real GNN)

#### React Frontend (`frontend/src/`)
Built with Vite + TypeScript + Recharts.

**Views:**
- **`Login.tsx`** — Username/password login form; stores JWT in `authStore`; routes to role-appropriate dashboard
- **`OperatorDashboard.tsx`** — KPI grid (curtailment %, peak reduction %, transformer aging, P2P volume kWh/₹, overload events), Grid Health Map (per-node voltage + load + stress badge), Duck Curve Tracker (actual vs. forecast line chart), Risk Timeline (overload probability line chart), Signal History table, Manual Override Panel (broadcast any signal type)
- **`HomeownerDashboard.tsx`** — Today's energy flow stats (solar, consumed, imported, P2P bought/sold, net bill), EV Status (kWh progress bar, deadline, schedule), Battery Health (SoC, capacity, health %), Earnings vs. net-metering baseline, Live Market price chart, Consent & Privacy panel (granular category opt-in, data export button, deletion request button)
- **`CommunityDashboard.tsx`** — Backup hours available (real-time), community savings (today / month / total), Fairness Allocation breakdown by priority tier, Flag Critical Load form (node ID, priority tier, reason)

**Shared infrastructure:**
- `api/client.ts` — `apiFetch()` wrapper with auto-auth header injection; `authStore` for JWT and node ID
- `hooks/useLiveStream.ts` — WebSocket hook; reconnects on close; exposes `status` and latest `snapshot`
- Components: `StatCard`, `StatusBadge`, `ProgressBar`

---

### Phase 7 — Real Data Integration (Partial)

**Status: 🟡 Partial (~40% complete)**

#### Implemented
- **Pecan Street transformation pipeline** (`pecan_data_wireup.py`):
  - Reads raw Pecan Street 1-minute CSV files (multiple source regions: Austin, California, New York)
  - IST timestamp alignment
  - India daily-energy scaling (target kWh/day configurable per city)
  - NSRDB solar replacement (optional `--replace-solar-with-nsrdb` flag)
  - Output: per-city/per-year CSV + Parquet + summary CSV under `data/processed/pecan_india/`
  - Idempotent: deterministic output given same inputs
  - Quality gates enforced: timestamp continuity, <1% missingness, negative load clipping
- **NSRDB data downloader** (`nsrdb_data_download.py`):
  - Downloads Himawari-derived GHI CSVs for target Indian cities
  - Stored under `data/nsrdb_himawari/`
- **NSRDB integration in simulator** (`simulator/load_profiles.py`):
  - `_load_nsrdb_normalized_ghi()` reads city-year CSV and reindexes to simulation time index
  - Solar curve is replaced when `replace_solar_with_nsrdb = True` in config

#### Not Yet Implemented
- **PVlib + SAM solar simulation pipeline** — convert GHI to per-minute AC output using panel specs and city coordinates; cache outputs for repeatable training
- **AIKosh sandbox integration** — formal API request + ingestion of localized Indian smart meter and transformer telemetry
- **Model retraining pipeline** — automated re-run of CortexCore and VayuGNN on augmented real-data dataset with metric validation gate

---

### Baseline Protocol & Success Metrics

**Status: ✅ Documented, ❌ Not yet implemented as runnable code**

`Baseline_Protocol_and_Success_Metrics.md` fully locks:

- **B0** (no control), **B1** (rule-based TOU), **B2** (MPC-lite) — synthetic simulator baselines
- **PB0** (Pecan replay + no control), **PB1** (TOU + self-consumption rules), **PB2** (forecasted MPC-lite) — real-data baselines

Six KPI formulas are locked with exact definitions:
1. Solar curtailment (%)
2. Peak demand reduction (%)
3. Transformer overload events (threshold: >1.2 pu for 5 consecutive timesteps)
4. Cost reduction (%)
5. Island switchover time (seconds)
6. P2P settlement latency (p50/p95/p99)

The experiment matrix covers 5 cities × 3 day types × 3 seasons × 3 scales × 2 penetration sets × 3 fault types × 5 seeds.

---

### Documentation

**Status: ✅ Good coverage for completed phases**

| File | Contents |
|---|---|
| `README.md` | Local setup, all endpoints, quick-start commands |
| `simulator/README.md` | Package guide, config reference, example run |
| `docs/api_phase4.md` | Phase 4 endpoint reference |
| `docs/api_phase6.md` | Phase 6 endpoint reference |
| `docs/output_schema_definitions.md` | All simulator output column definitions |
| `docs/scenario_configuration_guide.md` | JSON scenario field reference |
| `Baseline_Protocol_and_Success_Metrics.md` | Locked KPI definitions and experiment matrix |
| `Varshith_Workstream_README.md` | Varshith's scope, deliverables, and key interfaces |
| `VayuGrid_Team_Plan.md` | Full 14-week team plan with per-person phase breakdown |

---

### Tests

**Status: 🟡 Partial (unit tests only, no integration or load tests)**

| Test file | Coverage |
|---|---|
| `tests/test_ai_schemas.py` | `NodeState`, `TradeOrder`, `GridTelemetry`, `NeighborhoodSignal` dataclasses |
| `tests/test_ledger.py` | `AyaLedger` — append, hash linkage, `verify()` |
| `tests/test_pricing.py` | `PricingPolicy.from_signal()` across all signal types and severity levels |
| `tests/test_simulator.py` | `GridSimulator` instantiation, run, output shape |
| `tests/test_trading_engine.py` | `MatchingEngine` order lifecycle, rate limiting, island mode rejection |

---

---

## What Still Needs to Be Done

### Phase 2 — CortexCore AI Agent (Not Started)

**Status: ❌ Not started**
**Owner:** Charithra (AI) | **Support:** Varshith (scenarios)

This is the edge AI agent that runs on each Vayu-Node and autonomously manages household energy decisions.

#### Agent Design
- [ ] Define full observation vector (battery SoC, solar output, load, voltage, EV state, market depth, GNN signal, time encoding, 15-min forecasts)
- [ ] Define action space (5 continuous controls: battery charge/discharge rate, grid import/export, EV charge rate, bid price, ask price)
- [ ] Implement hard physical constraints as post-processing (C-rate limits, SoC bounds, grid import/export bounds)

#### Reward Function
- [ ] Economic reward: P2P revenue − grid import cost
- [ ] EV deadline penalty (heavy penalty for missing deadline)
- [ ] Battery health penalty (SoC extremes)
- [ ] Grid cooperation reward (response to GNN signals)
- [ ] Solar curtailment penalty

#### PPO Training Pipeline
- [ ] Implement PPO with asymmetric actor-critic (small actor for deployment, large critic for training)
- [ ] Curriculum learning: 4 stages (single-home → P2P market → multi-node → fault injection → real data)
- [ ] MLflow logging for all training runs
- [ ] ONNX export of actor network
- [ ] INT8 quantization (must achieve <5ms inference on Raspberry Pi 5)
- [ ] Observation normalizer (required for correct deployment behavior)

#### Short-Horizon LSTM Forecaster
- [ ] Train LSTM for 15-minute ahead predictions of solar, load, and market price
- [ ] Integrate forecaster outputs into the agent observation vector

#### Training Scenario Suite (Varshith's support)
- [ ] Seasonal variation episodes (summer/monsoon/winter solar curves)
- [ ] Weekday/weekend load diversity
- [ ] EV deadline pressure cases (short vs. long charge windows)
- [ ] Low/medium/high P2P activity episodes
- [ ] Fault-mix episodes (overload, solar dropout, island)
- [ ] Domain randomization:
  - Battery capacity ±20%
  - Solar efficiency degradation 0–15%
  - Load noise ±10%
  - Voltage variation within realistic range
  - Market signal delay 50–500ms

#### Gate Condition
- [ ] PPO must beat both B1 and B2 baselines by ≥20% on total cost on held-out test scenarios

---

### Phase 3 — VayuGNN Neighborhood Brain (Not Started)

**Status: ❌ Not started**
**Owner:** Charithra (AI) | **Support:** Varshith (dataset)

The GNN watches the entire neighborhood graph and predicts transformer overload risk, broadcasting signals to coordinate collective behavior.

#### Model Architecture
- [ ] Heterogeneous Graph Transformer (HGT) for spatial message passing (homes and transformers as different node types)
- [ ] Multi-head temporal self-attention over 12-snapshot history (12-minute window)
- [ ] Output heads: per-minute overload probability (next 30 min), voltage forecast, neighborhood risk score (0–1), 24-hour duck curve load forecast

#### Training
- [ ] Implement Focal Loss (α=0.75, γ=2.0) for imbalanced overload event detection
- [ ] Hard constraint: false positive rate < 1% on held-out test set
- [ ] Time-based train/val/test split only (70/15/15, no shuffle across time windows)

#### Graph Dataset Generator (Varshith's support)
- [ ] Generate graph dataset: 12 historical snapshots → 30-minute target window
- [ ] Automated labeling pipeline for transformer overload events
- [ ] ≥3 months of simulated data across cities and fault profiles

#### Signal Generation
- [ ] Threshold-based signal translator:
  - Risk > 0.85 or voltage < 0.88 pu in 5 minutes → `ISLAND`
  - Risk > 0.5 → `THROTTLE` to top 20% flexible loads
  - Duck curve ramp > 2 kW/min predicted → `PRE_COOL`
  - Risk < 0.1 for 5 minutes → `RESUME`
- [ ] Wire GNN predictions into the API `/ws/stream` (currently using sigmoid placeholder)

#### Fairness Pool Algorithm
- [ ] Priority queue for island-mode resource allocation:
  1. Medical-critical loads (CPAP, oxygen, refrigerated medication)
  2. Refrigeration (food safety)
  3. Basic lighting (one circuit per home)
  4. Communications
  5. Comfort cooling (proportional allocation of remainder)
- [ ] Configurable priority overrides per community
- [ ] Wire into `CommunityDashboard.tsx` Fairness Allocation panel

---

### Baseline Framework (Not Started)

**Status: ❌ Not started**
**Owner:** Varshith

Runnable implementations of the three baseline controllers are needed before any model training can be evaluated.

- [ ] **B0 controller** — No-control: solar serves load, battery off, EV charges immediately at max rate, no P2P
- [ ] **B1 controller** — Rule-based TOU: deterministic battery and EV rules, P2P based on price thresholds vs. grid tariff
- [ ] **B2 controller** — MPC-lite: linear programming, 30-minute rolling horizon, re-optimizes every timestep
- [ ] **PB0 controller** — Pecan replay + no control
- [ ] **PB1 controller** — Pecan TOU + self-consumption rules
- [ ] **PB2 controller** — Pecan forecasted MPC-lite (persistence forecast: last 7-day same-minute median)
- [ ] Automated benchmark runner: runs all controllers across the full experiment matrix (5 cities × 3 day types × 3 seasons × 3 scales × fault scenarios × 5 seeds)
- [ ] Reporting: metric table in the format defined in `Baseline_Protocol_and_Success_Metrics.md`
- [ ] Acceptance gate check: CV < 0.1, controller ordering stable

---

### Phase 7 — Real Data Integration (Remaining)

**Status: 🟡 ~40% complete**

- [ ] **PVlib + SAM solar simulation pipeline** — compute per-minute AC solar output from GHI, panel tilt/azimuth, and inverter specs for each Indian city; cache outputs
- [ ] **AIKosh sandbox integration** — formal API access request; ingestion of smart meter readings and transformer telemetry from Indian distribution grids; transformer-level and weather-linked telemetry
- [ ] **Model retraining validation pipeline** — after ingesting real data, automatically retrain CortexCore and VayuGNN and verify that all Phase 0 KPI targets still hold
- [ ] Charithra: **transfer learning fine-tuning** of both models on real data; re-tune GNN signal thresholds for real-world noise

---

### Phase 8 — Integration & Regression Testing (Not Started)

**Status: ❌ Not started**
**Owner:** Mukul (integration), Varshith (regression), Charithra (model behavior), Anjali (E2E)

#### Integration Tests
- [ ] **Full trade cycle test**: spin up 5 simulated nodes; one sells, one buys; verify match within 500ms, state updates correct, ledger entry valid with Ed25519 signatures
- [ ] **Island sequence test**: simulate grid voltage collapse → GNN detects within 2 prediction intervals (2 min) → ISLAND signal broadcast within 5s → all nodes in island mode within 30s → market suspended → Fairness Pool active → zero trades generated during event

#### Load Tests (Locust)
- [ ] 500 simultaneous nodes sending telemetry — p99 ingestion latency < 100ms
- [ ] 1,000 trade orders/minute — p99 settlement latency < 500ms
- [ ] GNN inference on 500-node graph — < 2 seconds on CPU
- [ ] 100 concurrent WebSocket connections — no visible lag

#### Chaos Engineering
- [ ] Kill neighborhood server mid-operation — nodes continue autonomously and reconnect
- [ ] Corrupt 10% of incoming telemetry — anomaly detection rejects without crashing GNN
- [ ] Flood trade queue with 10,000 orders/min — circuit breaker activates
- [ ] Instantaneous grid voltage drop to zero — island sequence triggers immediately

#### Regression Testing (Varshith)
- [ ] Automated baseline comparison on every PR touching `simulator/` or training pipelines
- [ ] Block/flag merges when any Phase 0 KPI target regresses

#### Model Behavioral Tests (Charithra)
- [ ] Agent always meets EV deadlines given sufficient advance notice
- [ ] Agent responds to THROTTLE within 2 timesteps
- [ ] Agent maintains minimum battery buffer during island events
- [ ] GNN false positive rate < 1% on held-out test set

#### Dashboard E2E Tests (Anjali)
- [ ] All three dashboards tested against staging environment
- [ ] Real-time WebSocket updates arrive within expected latency
- [ ] Data export and deletion endpoints verified
- [ ] Consent flow understandable to a non-technical user
- [ ] Basic accessibility audit

---

### Phase 9 — Deployment (Not Started)

**Status: ❌ Not started**
**Owner:** Mukul

#### Vayu-Node Hardware Stack (per household)
- [ ] Document and source: Raspberry Pi 5 (8GB), 32GB industrial SD, USB3 SSD, Wi-Fi 6 + 4G LTE fallback, RS-485/Modbus RTU adapter, UPS backup, IP54 DIN enclosure

#### Neighborhood Server (per 50–200 homes)
- [ ] Document and source: Intel NUC 13 Pro or equivalent (i5, 32GB RAM, 1TB NVMe)
- [ ] Production Docker Compose configuration

#### OTA Model Update Process
- [ ] Training completes → MLflow registers model → CI runs validation suite
- [ ] Passing model artifact signed with Ed25519 key
- [ ] 5% canary deploy → 24-hour monitoring → full rollout or automatic rollback
- [ ] Mender.io integration for OTA delivery with A/B partition rollback

#### Hardware-in-the-Loop Test
- [ ] 3 physical Raspberry Pi 5 devices for 72 hours with simulated smart meter data via serial
- [ ] Measure: inference latency (<5ms), memory, CPU, thermal behavior
- [ ] Verify OTA updates apply without service interruption

---

### Phase 10 — Scale & Federation (Not Started)

**Status: ❌ Not started**
**Owner:** All

#### Horizontal Scaling (Mukul)
- [ ] RabbitMQ 3-node cluster with quorum queues for HA
- [ ] TimescaleDB continuous aggregates and compression for dashboard query performance
- [ ] Read replica for dashboard queries (write primary reserved for ingestion)
- [ ] Stateless API + matching engine scaled horizontally behind a load balancer

#### Multi-Neighborhood Federation (Mukul)
- [ ] Federation API exposed by each neighborhood server
- [ ] Regional Aggregator: GNN over neighborhood-level summaries (not individual home data)
- [ ] Inter-neighborhood trade proposals via Regional Aggregator
- [ ] Energy flows via utility grid; settlement via Aya Ledger

#### MARL Upgrade (Charithra)
- [ ] Upgrade from single-agent PPO to MAPPO with centralized critic
- [ ] Emergent coordination: duck curve flattening, surplus signaling, EV shift
- [ ] Runtime actors remain local (no inter-agent real-time communication)

#### Federated Learning (Charithra)
- [ ] Flower framework integration for federated GNN training
- [ ] Each neighborhood trains locally; shares only gradients (FedAvg aggregation)
- [ ] Raw telemetry never leaves origin neighborhood (DPDP compliance)

#### Cross-Neighborhood Validation (Varshith)
- [ ] Baseline comparison for each newly onboarded neighborhood within first deployment week
- [ ] Flag low-performing neighborhoods for localized fine-tuning

#### Regional Operator Dashboard (Anjali)
- [ ] Extend Tony's dashboard with a regional map: all federated neighborhoods, risk levels, inter-neighborhood trade flows, aggregate KPIs
- [ ] Utility distribution planning view

---

### Security Hardening (Remaining)

**Status: ❌ Not started**

- [ ] **HashiCorp Vault** — self-hosted, for encryption key management with 90-day rotation
- [ ] **AES-256-GCM at-rest encryption** for all database-stored personal data
- [ ] **Mutual TLS** between every Vayu-Node and the neighborhood server
- [ ] **Isolation Forest anomaly detector** — trained on normal operation; triggers per-device Safe Mode for physically impossible readings
- [ ] **Wash trade detection** in the matching engine (same node appearing as both buyer and seller)
- [ ] **Replay attack prevention** — timestamp validation window: reject any message older than 30 seconds
- [ ] **DDoS circuit breaker** — halt new order acceptance when queue depth exceeds threshold
- [ ] **Automated breach notification system** — detect security incidents and trigger alerts within 72 hours
- [ ] **Database retention automation** — raw 1-min readings → delete after 90 days; 15-min aggregates → keep 3 years; monthly anonymized summaries → keep indefinitely
- [ ] **Data precision anonymization** — battery level rounded to 2 decimal places, load demand bucketed into 10 ranges before transmission

---

### Missing Feature Wiring

These features are partially present but not fully connected end-to-end:

- [ ] **Real GNN predictions in WebSocket stream** — currently using a sigmoid approximation on `transformer_loading_pu`; needs the actual VayuGNN model inference
- [ ] **Fairness Pool activation** — shown in `CommunityDashboard.tsx` but not driven by a real island-mode trigger from the GNN or market suspension event
- [ ] **libp2p peer discovery** — `mesh/rabbitmq_bus.py` notes this explicitly as a placeholder; actual libp2p gossip with mDNS discovery and 6-peer fan-out needs to be implemented for hardware deployments
- [ ] **LSTM forecaster wiring** — `NodeState.forecast_solar_kw_15m`, `forecast_load_kw_15m`, `forecast_price_inr_per_kwh_15m` fields exist in the schema but no forecaster populates them yet
- [ ] **Grafana dashboards** — the Grafana container is provisioned but no VayuGrid-specific dashboards have been created for simulator KPIs, trade flow, or transformer health

---

## At-a-Glance Status Table

| Phase | Description | Status | Owner |
|---|---|---|---|
| 0 | Infrastructure, data contracts, tooling | ✅ Complete | Mukul / All |
| 1 | Grid Simulator (full minute-resolution engine) | ✅ Complete | Varshith |
| 2 | CortexCore PPO Agent | ❌ Not started | Charithra |
| 3 | VayuGNN Neighborhood Brain | ❌ Not started | Charithra |
| 4 | Mesh network & P2P trading layer | ✅ Complete | Mukul |
| 5 | Security & Privacy (DPDP) | 🟡 ~60% | Anjali |
| 6 | API layer & three dashboards | ✅ Complete | Anjali |
| 7 | Real data integration (Pecan + NSRDB) | 🟡 ~40% | Varshith |
| 7b | PVlib/SAM + AIKosh integration | ❌ Not started | Varshith |
| 8 | Integration, load & regression testing | ❌ Not started | All |
| 9 | Hardware deployment + OTA updates | ❌ Not started | Mukul |
| 10 | Scale, federation, MARL, federated learning | ❌ Not started | All |

**Baseline framework** (B0/B1/B2/PB0/PB1/PB2): ✅ Documented | ❌ Not implemented as code

---

## Critical Path & Handoffs

The following dependencies must be resolved in order. Delays here cascade across the entire project.

```
Phase 1 (Simulator) ─────────────────────────────────────────────────── DONE
    │
    ├──► Phase 2 (CortexCore) ──────────────────────────────────────── BLOCKED
    │         Needs: simulator, baseline B0/B1/B2 for gate check
    │
    ├──► Phase 3 (VayuGNN) ─────────────────────────────────────────── BLOCKED
    │         Needs: simulator, graph dataset generator
    │         Needs: Phase 2 complete for NeighborhoodSignal schema (done: in ai/schemas.py)
    │
    ├──► Baseline Framework (B0/B1/B2) ─────────────────────────────── BLOCKED
    │         Needs: simulator (DONE), then implement controllers
    │
    └──► Phase 7b (PVlib + AIKosh) ─────────────────────────────────── BLOCKED
              Needs: Pecan pipeline (DONE), then PVlib + AIKosh access

Phase 3 (VayuGNN) ───────────────────────────────────────────────────── BLOCKED
    │
    └──► Real GNN in WebSocket stream ──────────────────────────────── BLOCKED
    └──► Fairness Pool live activation ─────────────────────────────── BLOCKED
    └──► Phase 8 island sequence test ──────────────────────────────── BLOCKED

Phase 2 + Phase 3 both done ─────────────────────────────────────────── then:
    │
    ├──► Phase 8 (Integration + load + chaos testing)
    ├──► Phase 9 (Hardware deployment + OTA)
    └──► Phase 10 (Scale, MARL, federated learning)
```

**Key handoffs still pending:**
1. **Varshith → Charithra** (Baseline framework code) — B0/B1/B2 implementations needed before Phase 2 gate check
2. **Charithra → all** (trained CortexCore model + normalizer + benchmarks) — needed before Phase 8
3. **Charithra → all** (VayuGNN model + signal thresholds) — needed to wire real predictions into API and trigger real Fairness Pool
4. **Varshith → Charithra** (3+ months of GNN graph dataset) — needed to start VayuGNN training
5. **Mukul → Anjali** (federation API spec) — needed before regional dashboard in Phase 10
