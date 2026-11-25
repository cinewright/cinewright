"""Migration 0002: a database written by a 0001 core keeps the outputs of its succeeded jobs.

The 0001 core recorded each output upload as a log event carrying the reserved `output` key and the
completion's output hashes in the terminal state event. The rows below have exactly that shape.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime

from conftest import fake_detect

from cinewright.api.schemas import JobRequest, Provenance, ProvenanceAsset, Route
from cinewright.core.core import Core
from cinewright.core.registry import Registry
from cinewright.db.database import Database, bundled_migrations

X, Y, Z, W = ("1" * 64, "2" * 64, "3" * 64, "4" * 64)
AT = "2026-11-20T10:00:00.000000+00:00"


def upload_record(name: str, sha256: str, media_type: str = "image/png", size_bytes: int = 10) -> dict:
    """The log event the 0001 core's `store_output` wrote."""
    return {
        "message": f"Uploaded output {name}",
        "output": {"name": name, "sha256": sha256, "media_type": media_type, "size_bytes": size_bytes},
    }


def provenance(job_id: str, sha256: str, media_type: str = "image/png") -> str:
    return Provenance(
        asset=ProvenanceAsset(sha256=sha256, media_type=media_type, size_bytes=10),
        prompt=f"prompt of {job_id}",
        seed=7,
        job_id=job_id,
        created_at=datetime(2026, 11, 20, 10, tzinfo=UTC),
    ).model_dump_json(by_alias=True)


def old_job(conn, job_id: str, state: str, events: list[tuple[str, dict]]) -> None:
    request = JobRequest.model_validate({"capability": "t2i", "prompt": f"prompt of {job_id}"})
    route = Route(model_id="test-image", quant="bf16", host_id="local", engine="wan2gp")
    conn.execute(
        "INSERT INTO jobs (id, state, priority, capability, data_class, request, route, seed, client, created_at, finished_at) "
        "VALUES (?, ?, 'interactive', 't2i', 'private', ?, ?, 7, 'ui', ?, ?)",
        (job_id, state, request.model_dump_json(), route.model_dump_json(), AT, AT),
    )
    for seq, (kind, data) in enumerate(events, start=1):
        conn.execute(
            "INSERT INTO job_events (job_id, seq, kind, data, created_at) VALUES (?, ?, ?, ?, ?)",
            (job_id, seq, kind, json.dumps(data), AT),
        )


def old_asset(conn, sha256: str, job_id: str, media_type: str = "image/png") -> None:
    conn.execute(
        "INSERT INTO assets (sha256, media_type, size_bytes, data_class, job_id, provenance, created_at) "
        "VALUES (?, ?, 10, 'private', ?, ?, ?)",
        (sha256, media_type, job_id, provenance(job_id, sha256, media_type), AT),
    )


def build_0001_database(path, tmp_path) -> None:
    only_0001 = tmp_path / "migrations-0001"
    only_0001.mkdir()
    shutil.copy(bundled_migrations() / "0001_initial.sql", only_0001)
    db = Database(path)
    try:
        assert db.migrate(only_0001) == ["0001_initial"]
        started = [("state", {"state": "queued"}), ("state", {"state": "leased", "worker_id": "w"}), ("state", {"state": "running"})]
        with db.transaction() as conn:
            # Two outputs in completion order Y, Z; frame.png was uploaded twice (the last upload wins), extra.png
            # was uploaded but not completed, and one record is malformed.
            old_job(
                conn,
                "job-a",
                "succeeded",
                [
                    *started,
                    ("log", upload_record("frame.png", X)),
                    ("log", upload_record("clip.mp4", Y, "video/mp4")),
                    ("log", {"message": "model loaded"}),
                    ("log", upload_record("frame.png", Z)),
                    ("log", upload_record("extra.png", W)),
                    ("log", {"message": "Uploaded output bad", "output": {"name": "bad"}}),
                    ("state", {"state": "succeeded", "outputs": [Y, Z]}),
                ],
            )
            old_asset(conn, Y, "job-a", "video/mp4")
            old_asset(conn, Z, "job-a")
            # Identical bytes as job A, under two names; the asset keeps job A's record.
            old_job(
                conn,
                "job-b",
                "succeeded",
                [
                    *started,
                    ("log", upload_record("clip.mp4", Y, "video/mp4")),
                    ("log", upload_record("copy.mp4", Y, "video/mp4")),
                    ("state", {"state": "succeeded", "outputs": [Y, Y]}),
                ],
            )
            old_job(conn, "job-c", "failed", [*started, ("log", upload_record("frame.png", X)), ("state", {"state": "failed"})])
    finally:
        db.close()


def outputs_before(path, job_id: str) -> list[str]:
    """What the 0001 core listed as `Job.outputs`: the hashes of the terminal state event of a succeeded job."""
    db = Database(path)
    try:
        row = db.query_one(
            "SELECT jobs.state, (SELECT data FROM job_events e WHERE e.job_id = jobs.id AND e.kind = 'state' ORDER BY e.seq DESC LIMIT 1) "
            "AS last_state FROM jobs WHERE id = ?",
            (job_id,),
        )
        return json.loads(row["last_state"]).get("outputs", []) if row["state"] == "succeeded" else []
    finally:
        db.close()


def test_a_0001_database_keeps_the_outputs_of_its_succeeded_jobs(config, registry_dir, clock, tmp_path):
    path = config.paths.db_file
    build_0001_database(path, tmp_path)
    before = {job_id: outputs_before(path, job_id) for job_id in ("job-a", "job-b", "job-c")}
    assert before == {"job-a": [Y, Z], "job-b": [Y, Y], "job-c": []}

    core = Core(config, registry=Registry.load(registry_dir), detect=fake_detect, clock=clock)
    try:
        assert "0002_job_outputs" in {row["version"] for row in core.db.query("SELECT version FROM schema_migrations")}
        assert {job_id: core.get_job(job_id).outputs for job_id in before} == before
        assert {job.id: job.outputs for job in core.list_jobs(None, None, 10).items} == before

        rows = {(row["job_id"], row["name"]): row for row in core.db.query("SELECT * FROM job_outputs")}
        assert {key: (row["sha256"], row["media_type"], row["position"]) for key, row in rows.items()} == {
            ("job-a", "clip.mp4"): (Y, "video/mp4", 0),
            ("job-a", "frame.png"): (Z, "image/png", 1),
            ("job-a", "extra.png"): (W, "image/png", None),
            ("job-b", "clip.mp4"): (Y, "video/mp4", 0),
            ("job-b", "copy.mp4"): (Y, "video/mp4", 1),
        }
        assert all(row["size_bytes"] == 10 and row["created_at"] == AT for row in rows.values())
        # A backfilled row takes the asset's provenance only when the asset names that job.
        assert {key: row["provenance"] for key, row in rows.items()} == {
            ("job-a", "clip.mp4"): provenance("job-a", Y, "video/mp4"),
            ("job-a", "frame.png"): provenance("job-a", Z),
            ("job-a", "extra.png"): None,
            ("job-b", "clip.mp4"): None,
            ("job-b", "copy.mp4"): None,
        }
        assert core.get_asset(Y).job_id == "job-a"  # the asset routes answer as before
    finally:
        core.close()


def test_a_completion_that_repeated_a_name_lists_that_output_once(config, registry_dir, clock, tmp_path):
    # Only a worker other than the shipped one could send this completion; one row cannot hold two positions.
    path = config.paths.db_file
    build_0001_database(path, tmp_path)
    db = Database(path)
    try:
        with db.transaction() as conn:
            completed = ("state", {"state": "succeeded", "outputs": [X, X]})
            old_job(conn, "job-d", "succeeded", [("state", {"state": "running"}), ("log", upload_record("a.png", X)), completed])
            old_asset(conn, X, "job-d")
    finally:
        db.close()
    assert outputs_before(path, "job-d") == [X, X]

    core = Core(config, registry=Registry.load(registry_dir), detect=fake_detect, clock=clock)
    try:
        assert core.get_job("job-d").outputs == [X]
        rows = core.db.query("SELECT name, sha256, position, provenance FROM job_outputs WHERE job_id = 'job-d'")
        assert [tuple(row) for row in rows] == [("a.png", X, 0, provenance("job-d", X))]
    finally:
        core.close()
