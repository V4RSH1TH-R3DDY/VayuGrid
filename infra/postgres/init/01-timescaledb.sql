CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS node_telemetry (
    ts TIMESTAMPTZ NOT NULL,
    node_id INTEGER NOT NULL,
    node_type TEXT NOT NULL,
    battery_soc_kwh DOUBLE PRECISION,
    battery_power_kw DOUBLE PRECISION,
    solar_output_kw DOUBLE PRECISION,
    household_load_kw DOUBLE PRECISION,
    ev_charge_kw DOUBLE PRECISION,
    net_grid_kw DOUBLE PRECISION,
    voltage_pu DOUBLE PRECISION,
    metadata JSONB DEFAULT '{}'::jsonb,
    PRIMARY KEY (ts, node_id)
);

SELECT create_hypertable('node_telemetry', 'ts', if_not_exists => TRUE);
ALTER TABLE node_telemetry SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'node_id'
);
SELECT add_compression_policy('node_telemetry', INTERVAL '7 days', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS transformer_readings (
    ts TIMESTAMPTZ NOT NULL,
    transformer_id TEXT NOT NULL,
    feeder_total_kw DOUBLE PRECISION NOT NULL,
    transformer_loading_pu DOUBLE PRECISION NOT NULL,
    max_branch_loading_pu DOUBLE PRECISION NOT NULL,
    hottest_spot_temp_c DOUBLE PRECISION NOT NULL,
    aging_acceleration DOUBLE PRECISION NOT NULL,
    grid_available BOOLEAN NOT NULL,
    islanding_triggered BOOLEAN NOT NULL,
    maintenance_mode BOOLEAN NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    PRIMARY KEY (ts, transformer_id)
);

SELECT create_hypertable('transformer_readings', 'ts', if_not_exists => TRUE);
ALTER TABLE transformer_readings SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'transformer_id'
);
SELECT add_compression_policy('transformer_readings', INTERVAL '7 days', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS trade_records (
    trade_id TEXT PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL,
    buyer_node_id INTEGER NOT NULL,
    seller_node_id INTEGER NOT NULL,
    quantity_kwh DOUBLE PRECISION NOT NULL,
    cleared_price_inr_per_kwh DOUBLE PRECISION NOT NULL,
    status TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb
);

SELECT create_hypertable('trade_records', 'ts', if_not_exists => TRUE, migrate_data => TRUE);
ALTER TABLE trade_records SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'buyer_node_id,seller_node_id'
);
SELECT add_compression_policy('trade_records', INTERVAL '7 days', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS trade_orders (
    order_id TEXT PRIMARY KEY,
    node_id INTEGER NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
    quantity_kwh DOUBLE PRECISION NOT NULL CHECK (quantity_kwh > 0),
    remaining_kwh DOUBLE PRECISION NOT NULL CHECK (remaining_kwh >= 0),
    limit_price_inr_per_kwh DOUBLE PRECISION NOT NULL CHECK (limit_price_inr_per_kwh >= 0),
    status TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    matched_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS trade_orders_open_idx
    ON trade_orders (status, side, limit_price_inr_per_kwh, created_at)
    WHERE status = 'open';
CREATE INDEX IF NOT EXISTS trade_orders_node_created_idx ON trade_orders (node_id, created_at DESC);

CREATE TABLE IF NOT EXISTS market_state (
    id BOOLEAN PRIMARY KEY DEFAULT TRUE,
    mode TEXT NOT NULL,
    stress_score DOUBLE PRECISION NOT NULL,
    price_floor_inr_per_kwh DOUBLE PRECISION NOT NULL,
    price_cap_inr_per_kwh DOUBLE PRECISION NOT NULL,
    accepts_orders BOOLEAN NOT NULL,
    reason TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    CHECK (id)
);

INSERT INTO market_state (
    id, mode, stress_score, price_floor_inr_per_kwh, price_cap_inr_per_kwh,
    accepts_orders, reason, updated_at
) VALUES (TRUE, 'normal', 0, 3, 12, TRUE, 'normal', now())
ON CONFLICT (id) DO NOTHING;

CREATE TABLE IF NOT EXISTS market_events (
    event_id TEXT PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL,
    event_type TEXT NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS market_events_type_ts_idx ON market_events (event_type, ts DESC);

CREATE TABLE IF NOT EXISTS ledger_entries (
    sequence BIGSERIAL PRIMARY KEY,
    trade_id TEXT NOT NULL UNIQUE,
    ts TIMESTAMPTZ NOT NULL,
    buyer_node_id INTEGER NOT NULL,
    seller_node_id INTEGER NOT NULL,
    quantity_kwh DOUBLE PRECISION NOT NULL,
    cleared_price_inr_per_kwh DOUBLE PRECISION NOT NULL,
    previous_hash TEXT NOT NULL,
    entry_hash TEXT NOT NULL,
    buyer_signature TEXT,
    seller_signature TEXT,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS ledger_entries_month_idx ON ledger_entries (ts DESC);

CREATE TABLE IF NOT EXISTS signal_history (
    signal_id TEXT PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL,
    signal_type TEXT NOT NULL,
    severity DOUBLE PRECISION NOT NULL,
    target_node_ids JSONB NOT NULL,
    reason TEXT NOT NULL,
    recommended_price_floor_inr_per_kwh DOUBLE PRECISION,
    recommended_price_cap_inr_per_kwh DOUBLE PRECISION,
    expires_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'::jsonb
);

SELECT create_hypertable('signal_history', 'ts', if_not_exists => TRUE, migrate_data => TRUE);
ALTER TABLE signal_history SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'signal_type'
);
SELECT add_compression_policy('signal_history', INTERVAL '7 days', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS node_api_keys (
    node_id INTEGER NOT NULL,
    key_hash TEXT NOT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (node_id, key_hash)
);

CREATE UNIQUE INDEX IF NOT EXISTS node_api_keys_key_hash_idx ON node_api_keys (key_hash);

CREATE TABLE IF NOT EXISTS household_consents (
    node_id INTEGER NOT NULL,
    consented BOOLEAN NOT NULL,
    consent_version TEXT NOT NULL,
    categories JSONB NOT NULL,
    ip_address TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS household_consents_node_idx ON household_consents (node_id);

CREATE TABLE IF NOT EXISTS data_deletion_requests (
    request_id TEXT PRIMARY KEY,
    node_id INTEGER NOT NULL,
    requested_at TIMESTAMPTZ NOT NULL,
    scheduled_for TIMESTAMPTZ NOT NULL,
    status TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS data_deletion_requests_node_idx ON data_deletion_requests (node_id);

CREATE TABLE IF NOT EXISTS critical_load_flags (
    flag_id TEXT PRIMARY KEY,
    node_id INTEGER NOT NULL,
    priority_tier TEXT NOT NULL,
    reason TEXT NOT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS critical_load_flags_node_idx ON critical_load_flags (node_id);

SELECT add_retention_policy('node_telemetry', INTERVAL '90 days', if_not_exists => TRUE);

CREATE MATERIALIZED VIEW IF NOT EXISTS node_telemetry_15min
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('15 minutes', ts) AS bucket,
    node_id,
    AVG(battery_soc_kwh) AS battery_soc_kwh_avg,
    AVG(battery_power_kw) AS battery_power_kw_avg,
    AVG(solar_output_kw) AS solar_output_kw_avg,
    AVG(household_load_kw) AS household_load_kw_avg,
    AVG(ev_charge_kw) AS ev_charge_kw_avg,
    AVG(net_grid_kw) AS net_grid_kw_avg,
    AVG(voltage_pu) AS voltage_pu_avg
FROM node_telemetry
GROUP BY bucket, node_id;

SELECT add_continuous_aggregate_policy(
    'node_telemetry_15min',
    start_offset => INTERVAL '7 days',
    end_offset => INTERVAL '15 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

SELECT add_retention_policy('node_telemetry_15min', INTERVAL '3 years', if_not_exists => TRUE);

CREATE MATERIALIZED VIEW IF NOT EXISTS community_monthly_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 month', ts) AS month,
    SUM(COALESCE(solar_output_kw, 0)) AS solar_kw_sum,
    SUM(COALESCE(household_load_kw, 0)) AS load_kw_sum,
    SUM(COALESCE(net_grid_kw, 0)) AS net_grid_kw_sum
FROM node_telemetry
GROUP BY month;

SELECT add_continuous_aggregate_policy(
    'community_monthly_summary',
    start_offset => INTERVAL '1 year',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day'
);
