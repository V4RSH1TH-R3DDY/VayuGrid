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

ALTER TABLE signal_history
    ADD COLUMN IF NOT EXISTS recommended_price_floor_inr_per_kwh DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS recommended_price_cap_inr_per_kwh DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ;
