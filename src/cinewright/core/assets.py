"""Asset store: content-addressed blobs (`assets/<sha[:2]>/<sha>`) and one provenance record per asset.

Identical bytes are one asset. When a second upload or job produces them again, the first
provenance record stays and the data class becomes the stricter of the two.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from cinewright.api.schemas import (
    Asset,
    ClientKind,
    DataClass,
    HostKind,
    JobRequest,
    ModelDescriptor,
    Provenance,
    ProvenanceAsset,
    ProvenanceClient,
    ProvenanceHost,
    ProvenanceModel,
    Route,
)
from cinewright.core.clock import from_iso, to_iso
from cinewright.core.router import strictest
from cinewright.paths import DataPaths

UPLOAD_PREFIX = ".upload-"  # temporary files of uploads in progress, directly in the asset directory


@dataclass(frozen=True)
class StagedBlob:
    """A complete upload in a temporary file of the asset directory, not yet placed under its hash."""

    path: Path
    sha256: str
    size_bytes: int


class BlobWriter:
    """Writes an upload to a temporary file in the asset directory and hashes it on the way.

    The calls block on disk I/O; the API layer makes them off the event loop. `discard` removes the
    file unless `BlobStore.place` has moved it, and may be called any number of times.
    """

    def __init__(self, directory: Path) -> None:
        fd, name = tempfile.mkstemp(dir=directory, prefix=UPLOAD_PREFIX)
        self.path = Path(name)
        self._file = os.fdopen(fd, "wb")
        self._sha256 = hashlib.sha256()
        self.size_bytes = 0

    def write(self, data: bytes) -> None:
        self._sha256.update(data)
        self._file.write(data)
        self.size_bytes += len(data)

    def finish(self) -> StagedBlob:
        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()  # Windows cannot move or delete a file that is still open
        return StagedBlob(self.path, self._sha256.hexdigest(), self.size_bytes)

    def discard(self) -> None:
        try:
            self._file.close()
        except OSError:
            pass  # a buffered tail that cannot be flushed (disk full) goes with the file; the caller has the real error
        finally:
            self.path.unlink(missing_ok=True)  # close() releases the file even when its flush fails


class BlobStore:
    def __init__(self, paths: DataPaths) -> None:
        self.paths = paths

    def path(self, sha256: str) -> Path:
        return self.paths.asset_file(sha256)

    def exists(self, sha256: str) -> bool:
        return self.path(sha256).is_file()

    def writer(self) -> BlobWriter:
        """A temporary file for a new upload; its hash, and so its place, is known only at the end."""
        return BlobWriter(self.paths.assets_dir)

    def place(self, staged: StagedBlob) -> str:
        """Move a staged upload atomically to `assets/<sha[:2]>/<sha>` and return its SHA-256.

        An existing blob is left untouched. The temporary file is gone afterwards in every case.
        """
        source = staged.path
        if source.parent.resolve() != self.paths.assets_dir.resolve() or not source.name.startswith(UPLOAD_PREFIX):
            raise ValueError(f"{source} is not a staged upload of this asset store.")
        target = self.path(staged.sha256)
        try:
            if target.is_file():
                return staged.sha256
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.replace(source, target)
            except PermissionError:
                # Windows refuses to replace a file another thread just created and still holds open.
                if not target.is_file():
                    raise
        finally:
            source.unlink(missing_ok=True)
        return staged.sha256


def asset_from_row(row: sqlite3.Row) -> Asset:
    return Asset(
        sha256=row["sha256"],
        media_type=row["media_type"],
        size_bytes=row["size_bytes"],
        data_class=DataClass(row["data_class"]),
        job_id=row["job_id"],
        created_at=from_iso(row["created_at"]),
        provenance=Provenance.model_validate_json(row["provenance"]),
    )


def record_asset(
    conn: sqlite3.Connection, provenance: Provenance, data_class: DataClass, job_id: str | None, created_at: datetime
) -> sqlite3.Row:
    """Insert the asset record, or keep the existing one and tighten its data class."""
    sha256 = provenance.asset.sha256
    existing = conn.execute("SELECT * FROM assets WHERE sha256 = ?", (sha256,)).fetchone()
    if existing is None:
        conn.execute(
            "INSERT INTO assets (sha256, media_type, size_bytes, data_class, job_id, provenance, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                sha256,
                provenance.asset.media_type,
                provenance.asset.size_bytes,
                data_class.value,
                job_id,
                provenance.model_dump_json(by_alias=True),
                to_iso(created_at),
            ),
        )
        for parent in provenance.parents:
            conn.execute("INSERT OR IGNORE INTO asset_parents (asset_sha256, parent_sha256) VALUES (?, ?)", (sha256, parent))
    else:
        tighter = strictest([DataClass(existing["data_class"]), data_class])
        conn.execute("UPDATE assets SET data_class = ? WHERE sha256 = ?", (tighter.value, sha256))
    return conn.execute("SELECT * FROM assets WHERE sha256 = ?", (sha256,)).fetchone()


def input_provenance(sha256: str, media_type: str, size_bytes: int, created_at: datetime) -> Provenance:
    return Provenance(asset=ProvenanceAsset(sha256=sha256, media_type=media_type, size_bytes=size_bytes), created_at=created_at)


def output_provenance(
    *,
    asset: ProvenanceAsset,
    job_id: str,
    request: JobRequest,
    route: Route,
    model: ModelDescriptor | None,
    engine_version: str | None,
    seed: int | None,
    host: sqlite3.Row,
    gpu: str | None,
    client: ClientKind,
    agent: str | None,
    created_at: datetime,
) -> Provenance:
    """Provenance of a job output, built from the job, its route, the registry entry and the host."""
    files: list[str] = []
    if model is not None:
        quant = next((q for q in model.quantizations if q.id == route.quant), None)
        files = [f.sha256 for f in quant.files] if quant else []
    return Provenance(
        asset=asset,
        model=ProvenanceModel(
            id=route.model_id,
            name=model.name if model else route.model_id,
            quant=route.quant,
            engine=route.engine,
            engine_version=engine_version,
            file_sha256=files,
        ),
        recipe=request.recipe,
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        seed=seed,
        host=ProvenanceHost(id=host["id"], name=host["name"], kind=HostKind(host["kind"]), gpu=gpu),
        created_at=created_at,
        parents=list(request.inputs),
        job_id=job_id,
        client=ProvenanceClient(kind=client, agent=agent),
    )
