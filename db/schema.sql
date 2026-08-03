-- schema.sql
-- Database initialization schema for AI-NIDS.
-- Written to be SQLite-compatible out of the box (used by default via
-- AlertManager) and portable to MySQL with the noted substitutions:
--   INTEGER PRIMARY KEY AUTOINCREMENT  ->  INT AUTO_INCREMENT PRIMARY KEY
--   TEXT                               ->  VARCHAR(255) / TEXT
--   REAL                               ->  DOUBLE

-- ---------------------------------------------------------------------
-- Users & RBAC (FR7): distinct Administrator (full access) and
-- Analyst (read-only) roles.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    username        TEXT NOT NULL UNIQUE,
    password_hash   TEXT NOT NULL,
    role            TEXT NOT NULL CHECK (role IN ('admin', 'analyst')),
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ---------------------------------------------------------------------
-- Alerts (FR4): timestamped alert log including attack category,
-- confidence score, and source/destination IP.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS alerts (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT NOT NULL,
    attack_category   TEXT NOT NULL CHECK (attack_category IN
                        ('Normal', 'DoS', 'Probe', 'R2L', 'U2R')),
    confidence        REAL NOT NULL,
    src_ip            TEXT NOT NULL,
    dst_ip            TEXT NOT NULL,
    src_port          INTEGER DEFAULT 0,
    dst_port          INTEGER DEFAULT 0,
    status            TEXT NOT NULL DEFAULT 'alerted' CHECK (status IN
                        ('alerted', 'review_queue', 'acknowledged'))
);
CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts (timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts (status);

-- ---------------------------------------------------------------------
-- Audit trail (FR9): complete, immutable log of acknowledgements,
-- threshold changes, and retraining events. Application layer only
-- ever INSERTs into this table.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS audit_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    TEXT NOT NULL,
    event_type   TEXT NOT NULL,
    actor        TEXT NOT NULL,
    details      TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log (timestamp);

-- ---------------------------------------------------------------------
-- Model performance metrics snapshot (feeds FR6 dashboard + NF5 report)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_performance (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp     TEXT NOT NULL,
    model_name        TEXT NOT NULL,
    accuracy          REAL,
    precision_macro   REAL,
    recall_macro      REAL,
    f1_macro          REAL,
    false_positive_rate REAL,
    mean_latency_ms   REAL,
    dataset           TEXT
);
