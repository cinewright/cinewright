"""Blob store: streamed uploads are hashed on the way, placed atomically and never left behind as temporary files."""

from __future__ import annotations

import errno
import hashlib
import io
import os
from pathlib import Path

import pytest

from cinewright.core import assets
from cinewright.core.assets import BlobStore, StagedBlob
from cinewright.paths import DataPaths


@pytest.fixture
def blobs(tmp_path: Path) -> BlobStore:
    paths = DataPaths(tmp_path / "data")
    paths.ensure()
    return BlobStore(paths)


def temporary_files(blobs: BlobStore) -> list[Path]:
    return list(blobs.paths.assets_dir.rglob(f"{assets.UPLOAD_PREFIX}*"))


def stage(blobs: BlobStore, *chunks: bytes) -> StagedBlob:
    writer = blobs.writer()
    for chunk in chunks:
        writer.write(chunk)
    return writer.finish()


def test_a_streamed_upload_is_hashed_on_the_way_and_placed_under_its_hash(blobs):
    chunks = [os.urandom(1000), b"", os.urandom(70_000), b"tail"]
    writer = blobs.writer()
    assert writer.path.parent == blobs.paths.assets_dir  # the shard is known only once the hash is
    for chunk in chunks:
        writer.write(chunk)
    staged = writer.finish()
    data = b"".join(chunks)
    assert (staged.sha256, staged.size_bytes) == (hashlib.sha256(data).hexdigest(), len(data))
    assert staged.path.read_bytes() == data

    assert blobs.place(staged) == staged.sha256
    target = blobs.paths.assets_dir / staged.sha256[:2] / staged.sha256
    assert blobs.path(staged.sha256) == target
    assert target.read_bytes() == data
    assert not temporary_files(blobs)


def test_placing_leaves_an_existing_blob_untouched(blobs):
    first = stage(blobs, b"same bytes")
    sha256 = blobs.place(first)
    before = blobs.path(sha256).stat()
    second = stage(blobs, b"same bytes")
    assert blobs.place(second) == sha256
    after = blobs.path(sha256).stat()
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    assert not temporary_files(blobs)


def test_placing_tolerates_the_windows_refusal_when_the_same_bytes_just_arrived(blobs, monkeypatch):
    staged = stage(blobs, b"raced bytes")
    target = blobs.path(staged.sha256)

    def raced(source, destination):  # another upload of the same bytes created the target and still holds it open
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"raced bytes")
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(assets.os, "replace", raced)
    assert blobs.place(staged) == staged.sha256
    assert target.read_bytes() == b"raced bytes"
    assert not temporary_files(blobs)


def test_a_refused_move_without_a_blob_in_place_is_an_error_and_leaves_no_file(blobs, monkeypatch):
    staged = stage(blobs, b"refused bytes")

    def refused(source, destination):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(assets.os, "replace", refused)
    with pytest.raises(PermissionError):
        blobs.place(staged)
    assert not blobs.exists(staged.sha256)
    assert not temporary_files(blobs)


def test_only_staged_uploads_of_the_store_are_placed(blobs, tmp_path):
    elsewhere = tmp_path / "elsewhere.bin"
    elsewhere.write_bytes(b"not an upload")
    sha256 = hashlib.sha256(b"not an upload").hexdigest()
    unprefixed = blobs.paths.assets_dir / "stray.bin"
    unprefixed.write_bytes(b"not an upload")
    for path in (elsewhere, unprefixed):
        with pytest.raises(ValueError, match="not a staged upload"):
            blobs.place(StagedBlob(path, sha256, 13))
        assert path.read_bytes() == b"not an upload"
    assert not blobs.exists(sha256)


def test_a_discarded_upload_leaves_no_file(blobs):
    writer = blobs.writer()
    writer.write(b"half an upload")
    writer.discard()
    writer.discard()  # idempotent: the API layer discards on every exit
    assert not temporary_files(blobs)

    finished = blobs.writer()
    finished.write(b"complete but refused")
    finished.finish()
    finished.discard()
    assert not temporary_files(blobs)


class FullDisk(io.FileIO):
    """A file on a disk that is full: no write reaches it."""

    def write(self, data):
        raise OSError(errno.ENOSPC, "No space left on device")


def test_an_upload_whose_buffered_tail_cannot_be_flushed_is_still_discarded(blobs, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(assets.os, "fdopen", lambda fd, mode: io.BufferedWriter(FullDisk(fd, "w")))
        writer = blobs.writer()
    writer.write(b"a tail smaller than the write buffer")  # buffered: nothing has reached the disk yet
    with pytest.raises(OSError, match="No space"):
        writer.finish()
    writer.discard()  # the flush fails again; the API layer discards in a finally, where a raise would hide the real error
    writer.discard()
    assert not temporary_files(blobs)
