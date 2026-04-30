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

CREATE TABLE IF NOT EXISTS signal_history (
    signal_id TEXT PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL,
    signal_type TEXT NOT NULL,
    severity DOUBLE PRECISION NOT NULL,
    target_node_ids JSONB NOT NULL,
    reason TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb
);

SELECT create_hypertable('signal_history', 'ts', if_not_exists => TRUE, migrate_data => TRUE);
ALTER TABLE signal_history SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'signal_type'
);
SELECT add_compression_policy('signal_history', INTERVAL '7 days', if_not_exists => TRUE);
