-- Cinewright core database, migration 0001.
-- SQLite with foreign keys on and WAL journal. JSON columns hold contract models (src/cinewright/api/schemas.py).
-- Timestamps are ISO 8601 strings in UTC.

CREATE TABLE hosts (
    id                   TEXT PRIMARY KEY,
    name                 TEXT NOT NULL,
    kind                 TEXT NOT NULL CHECK (kind IN ('local', 'remote_own', 'remote_rented', 'cloud')),
    allowed_data_classes TEXT NOT NULL,              -- JSON array of DataClass
    enrolled_at          TEXT,
    revoked_at           TEXT,
    created_at           TEXT NOT NULL
);

CREATE TABLE hardware_profiles (
    host_id     TEXT PRIMARY KEY REFERENCES hosts (id) ON DELETE CASCADE,
    profile     TEXT NOT NULL,                       -- JSON HardwareProfile
    detected_at TEXT NOT NULL
);

CREATE TABLE workers (
    id             TEXT PRIMARY KEY,
    host_id        TEXT NOT NULL REFERENCES hosts (id) ON DELETE CASCADE,
    engine         TEXT NOT NULL,
    engine_version TEXT,
    models         TEXT NOT NULL,                    -- JSON array of engine model ids
    health         TEXT NOT NULL,                    -- JSON WorkerHealth
    token_hash     TEXT NOT NULL,                    -- SHA-256 of the worker's bearer token
    registered_at  TEXT NOT NULL,
    last_seen_at   TEXT NOT NULL
);

CREATE TABLE installed_models (
    model_id     TEXT NOT NULL,
    quant        TEXT NOT NULL,
    state        TEXT NOT NULL CHECK (state IN ('absent', 'downloading', 'installed', 'failed')),
    size_bytes   INTEGER,
    path         TEXT,
    last_used_at TEXT,
    installed_at TEXT,
    PRIMARY KEY (model_id, quant)
);

CREATE TABLE model_files (
    sha256      TEXT PRIMARY KEY,
    model_id    TEXT NOT NULL,
    quant       TEXT NOT NULL,
    name        TEXT NOT NULL,
    size_bytes  INTEGER,
    verified_at TEXT,
    FOREIGN KEY (model_id, quant) REFERENCES installed_models (model_id, quant) ON DELETE CASCADE
);

CREATE TABLE calibration_results (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    host_id          TEXT NOT NULL REFERENCES hosts (id) ON DELETE CASCADE,
    model_id         TEXT NOT NULL,
    quant            TEXT NOT NULL,
    resolution       TEXT NOT NULL,
    status           TEXT NOT NULL CHECK (status IN ('runs', 'slow', 'wont_run')),
    peak_vram_mb     INTEGER,
    seconds_per_step REAL,
    ram_offload_mb   INTEGER,
    detail           TEXT,
    engine           TEXT NOT NULL,
    engine_version   TEXT,
    measured_at      TEXT NOT NULL,
    UNIQUE (host_id, model_id, quant, resolution)
);

CREATE TABLE agents (
    id                  TEXT PRIMARY KEY,
    name                TEXT NOT NULL UNIQUE,
    token_hash          TEXT NOT NULL,
    quota_jobs_per_day  INTEGER,
    budget_eur_per_month REAL,
    revoked_at          TEXT,
    created_at          TEXT NOT NULL
);

CREATE TABLE jobs (
    id               TEXT PRIMARY KEY,
    state            TEXT NOT NULL CHECK (state IN ('awaiting_approval', 'queued', 'leased', 'running', 'succeeded', 'failed', 'cancelled')),
    priority         TEXT NOT NULL CHECK (priority IN ('interactive', 'batch')),
    capability       TEXT NOT NULL,
    data_class       TEXT NOT NULL CHECK (data_class IN ('public', 'private', 'biometric')),
    request          TEXT NOT NULL,                  -- JSON JobRequest
    route            TEXT,                           -- JSON Route
    seed             INTEGER,                        -- resolved seed
    client           TEXT NOT NULL CHECK (client IN ('ui', 'mcp')),
    agent_id         TEXT REFERENCES agents (id),
    progress         TEXT,                           -- JSON Progress
    eta_seconds      REAL,
    error            TEXT,                           -- JSON JobError
    attempts         INTEGER NOT NULL DEFAULT 0,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    worker_id        TEXT REFERENCES workers (id) ON DELETE SET NULL,
    lease_id         TEXT,
    lease_expires_at TEXT,
    created_at       TEXT NOT NULL,
    started_at       TEXT,
    finished_at      TEXT
);

CREATE INDEX jobs_queue ON jobs (state, priority, created_at);

CREATE TABLE job_events (
    job_id     TEXT NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    seq        INTEGER NOT NULL,
    kind       TEXT NOT NULL CHECK (kind IN ('state', 'progress', 'preview', 'log')),
    data       TEXT NOT NULL,                        -- JSON object
    created_at TEXT NOT NULL,
    PRIMARY KEY (job_id, seq)
);

CREATE TABLE assets (
    sha256     TEXT PRIMARY KEY,
    media_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    data_class TEXT NOT NULL CHECK (data_class IN ('public', 'private', 'biometric')),
    job_id     TEXT REFERENCES jobs (id) ON DELETE SET NULL,
    provenance TEXT NOT NULL,                        -- JSON Provenance
    created_at TEXT NOT NULL
);

CREATE TABLE asset_parents (
    asset_sha256  TEXT NOT NULL REFERENCES assets (sha256) ON DELETE CASCADE,
    parent_sha256 TEXT NOT NULL REFERENCES assets (sha256),
    PRIMARY KEY (asset_sha256, parent_sha256)
);

CREATE TABLE approvals (
    id                 TEXT PRIMARY KEY,
    job_id             TEXT NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    reason             TEXT NOT NULL,
    estimated_cost_eur REAL,
    host_id            TEXT REFERENCES hosts (id),
    state              TEXT NOT NULL CHECK (state IN ('pending', 'approved', 'declined')),
    created_at         TEXT NOT NULL,
    decided_at         TEXT
);

CREATE TABLE settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL                              -- JSON value
);
