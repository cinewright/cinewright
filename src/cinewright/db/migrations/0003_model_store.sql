-- Cinewright core database, migration 0003: the model store (#3).
-- One file in the model store can belong to several installed quantizations: the text encoders, VAEs and other
-- shared engine files a descriptor `requires`. `model_files` (0001) tied each file to exactly one model and
-- quantization and was never written by any release, so `store_files` and its references replace it.

DROP TABLE model_files;

-- Every file the model store holds, at the path Wan2GP looks it up below the store (its first checkpoint root).
-- Paths ignore case, as on Windows. A file is registered only after its SHA-256 and size matched the registry.
CREATE TABLE store_files (
    path        TEXT PRIMARY KEY COLLATE NOCASE,    -- relative to <data dir>/models, `/`-separated
    sha256      TEXT NOT NULL,
    size_bytes  INTEGER NOT NULL,
    verified_at TEXT NOT NULL
);

-- The files each installed quantization needs; the number of references to a path is its reference count.
CREATE TABLE installed_model_files (
    model_id TEXT NOT NULL,
    quant    TEXT NOT NULL,
    path     TEXT NOT NULL COLLATE NOCASE REFERENCES store_files (path),
    PRIMARY KEY (model_id, quant, path),
    FOREIGN KEY (model_id, quant) REFERENCES installed_models (model_id, quant) ON DELETE CASCADE
);

CREATE INDEX installed_model_files_path ON installed_model_files (path);

-- Why the last install of a quantization failed, in one sentence; null unless `state` is `failed`.
ALTER TABLE installed_models ADD COLUMN error TEXT;

-- Text encoder variant of a worker host (Wan2GP's `text_encoder_quantization`); null = not chosen yet, `int8` applies.
ALTER TABLE hosts ADD COLUMN text_encoder_variant TEXT CHECK (text_encoder_variant IN ('bf16', 'int8'));
