-- Cinewright core database, migration 0002: per-job output records.
-- Identical bytes are one asset, whose provenance names the first job that produced them. Each job's own
-- outputs and their provenance live here instead, so a second job with identical bytes keeps its record.

CREATE TABLE job_outputs (
    job_id     TEXT NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    name       TEXT NOT NULL,                        -- output name of the upload; the last upload of a name wins
    sha256     TEXT NOT NULL,
    media_type TEXT NOT NULL,                        -- the upload's checked Content-Type
    size_bytes INTEGER NOT NULL,
    position   INTEGER,                              -- index in the job's completion; null unless the job succeeded with it
    provenance TEXT,                                 -- JSON Provenance of this job's output; null until the job completes
    created_at TEXT NOT NULL,                        -- time of the upload
    PRIMARY KEY (job_id, name),
    UNIQUE (job_id, position)
);

-- Backfill for jobs that already succeeded. The 0001 core recorded each upload as a log event carrying the
-- key `output` ({name, sha256, media_type, size_bytes}) and the completion's output hashes in the terminal
-- state event (`outputs`), but no names there. The k-th occurrence of a hash in `outputs` gets the k-th
-- uploaded name with that hash, so the job lists the same outputs as before. One exception: a completion
-- that listed one name twice (never sent by the shipped worker) has one row for that name, so the job lists
-- that output once. A row takes the asset's provenance when the asset names that job; otherwise (identical
-- bytes of an earlier job) it stays null.
INSERT INTO job_outputs (job_id, name, sha256, media_type, size_bytes, position, provenance, created_at)
WITH records AS (
    SELECT e.job_id,
           e.seq,
           e.created_at,
           json_extract(e.data, '$.output.name') AS name,
           json_extract(e.data, '$.output.sha256') AS sha256,
           json_extract(e.data, '$.output.media_type') AS media_type,
           json_extract(e.data, '$.output.size_bytes') AS size_bytes
    FROM job_events e
    JOIN jobs j ON j.id = e.job_id
    WHERE j.state = 'succeeded'
      AND e.kind = 'log'
      AND json_type(e.data, '$.output.name') = 'text'
      AND json_type(e.data, '$.output.sha256') = 'text'
      AND json_type(e.data, '$.output.media_type') = 'text'
      AND json_type(e.data, '$.output.size_bytes') = 'integer'
),
latest AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY job_id, name ORDER BY seq DESC) AS newest
    FROM records
),
uploads AS (
    SELECT job_id, name, sha256, media_type, size_bytes, created_at,
           ROW_NUMBER() OVER (PARTITION BY job_id, sha256 ORDER BY seq) AS nth
    FROM latest
    WHERE newest = 1
),
listed AS (
    SELECT j.id AS job_id,
           o.key AS position,
           o.value AS sha256,
           ROW_NUMBER() OVER (PARTITION BY j.id, o.value ORDER BY o.key) AS nth
    FROM jobs j
    JOIN job_events e ON e.job_id = j.id
    JOIN json_each(e.data, '$.outputs') o
    WHERE j.state = 'succeeded'
      AND e.kind = 'state'
      AND e.seq = (SELECT MAX(s.seq) FROM job_events s WHERE s.job_id = j.id AND s.kind = 'state')
)
SELECT u.job_id,
       u.name,
       u.sha256,
       u.media_type,
       u.size_bytes,
       l.position,
       CASE
           WHEN l.position IS NOT NULL
           THEN (SELECT a.provenance FROM assets a WHERE a.sha256 = u.sha256 AND a.job_id = u.job_id)
       END,
       u.created_at
FROM uploads u
LEFT JOIN listed l ON l.job_id = u.job_id AND l.sha256 = u.sha256 AND l.nth = u.nth;
