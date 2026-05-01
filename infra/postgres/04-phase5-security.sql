-- Phase 5 Security Tables

-- Security events log (anomaly alerts, auth failures, breach records)
CREATE TABLE IF NOT EXISTS security_events (
    event_id    TEXT        PRIMARY KEY,
    ts          TIMESTAMPTZ NOT NULL DEFAULT now(),
    event_type  TEXT        NOT NULL,   -- 'anomaly' | 'auth_failure' | 'replay_attack' | 'breach'
    severity    FLOAT       NOT NULL CHECK (severity >= 0.0 AND severity <= 1.0),
    node_id     INTEGER,
    description TEXT        NOT NULL,
    notified_at TIMESTAMPTZ,            -- NULL until notification is dispatched
    metadata    JSONB       NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS security_events_ts_idx
    ON security_events (ts DESC);
CREATE INDEX IF NOT EXISTS security_events_type_ts_idx
    ON security_events (event_type, ts DESC);
CREATE INDEX IF NOT EXISTS security_events_unnotified_idx
    ON security_events (notified_at)
    WHERE notified_at IS NULL;

-- Seen mesh message IDs — used for distributed replay attack prevention
-- Entries older than 60 seconds are periodically purged by the security scheduler
CREATE TABLE IF NOT EXISTS seen_message_ids (
    message_id  TEXT        PRIMARY KEY,
    received_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS seen_message_ids_received_idx
    ON seen_message_ids (received_at);

-- Encryption key rotation audit trail
CREATE TABLE IF NOT EXISTS key_rotation_log (
    rotation_id TEXT        PRIMARY KEY,
    rotated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    key_version INTEGER     NOT NULL,
    rotated_by  TEXT        NOT NULL,
    metadata    JSONB       NOT NULL DEFAULT '{}'::jsonb
);
