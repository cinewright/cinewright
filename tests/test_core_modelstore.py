"""Model store (#3): planning, hash-checked downloads, placement where Wan2GP looks files up, shared files, quota.

The fixture registry is the bundled one with small stand-in files: the real descriptors' file names, subfolders
and URLs, but each file holds a few bytes of its own, pinned by their own SHA-256 and size. A fake hub serves them
through an `httpx.MockTransport` with Range requests, and answers weight files with a redirect to the CDN host as
Hugging Face does for files in Git LFS. Nothing touches the network.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pytest
from conftest import TEST_MODELS, FakeClock

from cinewright.core.clock import to_iso
from cinewright.core.modelstore import (
    STAGING_DIR,
    AllowedHosts,
    InstallProgress,
    ModelStore,
    ModelStoreError,
    check_file_type,
    format_size,
    merge_wgp_config,
    plan_files,
    store_path,
    wan2gp_defines,
)
from cinewright.core.registry import Registry
from cinewright.db.database import Database, bundled_migrations
from cinewright.paths import DataPaths

CDN = "us.aws.cdn.hf.co"
HUB = "huggingface.co"
MVP_MODEL_TYPES = (
    "z_image",
    "ltx2_22B_distilled",
    "ltx2_22B_distilled_gguf_q4_k_m",
    "ltx2_22B_distilled_gguf_q6_k",
    "ltx2_22B_distilled_gguf_q8_0",
    "t2v_2_2",
    "i2v_2_2",
)
# Z-Image Turbo int8 with the int8 text encoder: the transformer and the root-level shared files in the store root,
# the Qwen3 text encoder with its tokenizer and config files in Qwen3/ (upstream files_locator.locate_file).
Z_IMAGE_INT8_FILES = {
    "ZImageTurbo_quanto_bf16_int8.safetensors",
    "Qwen3/qwen3_quanto_bf16_int8.safetensors",
    "Qwen3/tokenizer.json",
    "Qwen3/tokenizer_config.json",
    "Qwen3/vocab.json",
    "Qwen3/config.json",
    "Qwen3/merges.txt",
    "ZImageTurbo_VAE_bf16.safetensors",
    "ZImageTurbo_VAE_bf16_config.json",
    "ZImageTurbo_scheduler_config.json",
}
WAN_T2V_INT8 = ("wan2.2_text2video_14B_high_quanto_mbf16_int8.safetensors", "wan2.2_text2video_14B_low_quanto_mbf16_int8.safetensors")
WAN_I2V_INT8 = ("wan2.2_image2video_14B_high_quanto_mbf16_int8.safetensors", "wan2.2_image2video_14B_low_quanto_mbf16_int8.safetensors")
UMT5_INT8 = "umt5-xxl/models_t5_umt5-xxl-enc-quanto_int8.safetensors"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fixture_registry() -> tuple[Registry, dict[str, bytes]]:
    """The bundled registry with a small stand-in per file; returns it and the stand-ins by URL."""
    real = Registry.load()
    contents: dict[str, bytes] = {}

    def small(file: Any, path: str) -> Any:
        data = f"stand-in for {path}\n".encode() * 3
        contents[file.url] = data
        return file.model_copy(update={"sha256": sha256(data), "size_bytes": len(data)})

    models = [
        model.model_copy(
            update={"quantizations": [q.model_copy(update={"files": [small(f, f.name) for f in q.files]}) for q in model.quantizations]}
        )
        for model in real.all()
    ]
    shared = [component.model_copy(update={"files": [small(f, f.path) for f in component.files]}) for component in real.components()]
    return Registry(models, shared), contents


class InterruptedStream(httpx.SyncByteStream):
    """A response body that breaks off after `data`, like a dropped connection."""

    def __init__(self, data: bytes) -> None:
        self.data = data

    def __iter__(self):
        yield self.data
        raise httpx.ReadError("connection lost")


class FakeHub:
    """Serves the stand-ins by URL with Range support; weight files redirect to `redirect_to` first, as LFS files do.

    `cut[url] = n` breaks the next download of `url` off after n bytes.
    """

    def __init__(self, contents: dict[str, bytes]) -> None:
        self.files = dict(contents)
        self.requests: list[tuple[str, str | None]] = []
        self.cut: dict[str, int] = {}
        self.redirect_to = CDN

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        ranged = request.headers.get("range")
        self.requests.append((url, ranged))
        if request.url.host == HUB and request.url.path.endswith((".safetensors", ".gguf")):
            return httpx.Response(302, headers={"location": f"https://{self.redirect_to}/lfs{request.url.path}"})
        original = f"https://{HUB}{request.url.path.removeprefix('/lfs')}" if request.url.host == CDN else url
        body = self.files.get(original)
        if body is None:
            return httpx.Response(404)
        start = int(ranged.removeprefix("bytes=").split("-")[0]) if ranged else 0
        headers = {"content-range": f"bytes {start}-{len(body) - 1}/{len(body)}"} if ranged else {}
        data = body[start:]
        if original in self.cut:
            return httpx.Response(206 if ranged else 200, headers=headers, stream=InterruptedStream(data[: self.cut.pop(original)]))
        return httpx.Response(206 if ranged else 200, headers=headers, content=data)

    def served(self, url: str) -> int:
        """Requests that reached the file itself (not only its redirect)."""
        path = httpx.URL(url).path
        return sum(1 for requested, _ in self.requests if httpx.URL(requested).host == CDN and httpx.URL(requested).path == f"/lfs{path}")


class Env:
    def __init__(self, tmp_path: Path) -> None:
        self.registry, self.contents = fixture_registry()
        self.hub = FakeHub(self.contents)
        self.paths = DataPaths(tmp_path / "data")
        self.paths.ensure()
        self.db = Database.open(self.paths.db_file)
        self.db.execute(
            "INSERT INTO hosts (id, name, kind, allowed_data_classes, created_at) VALUES ('local', 'Local', 'local', '[]', ?)",
            ("2026-11-25T00:00:00.000000+00:00",),
        )
        self.clock = FakeClock()
        self.free = 10**12
        self.quota: int | None = None
        self.engine_root: Path | None = None
        self.store = self.new_store()

    def new_store(self, **kwargs: Any) -> ModelStore:
        return ModelStore(
            self.db,
            self.registry,
            self.paths,
            quota_bytes=lambda: self.quota,
            free_bytes=lambda path: self.free,
            transport=httpx.MockTransport(self.hub.handler),
            engine_root=lambda: self.engine_root,
            clock=self.clock,
            **kwargs,
        )

    @property
    def root(self) -> Path:
        return self.paths.models_dir

    def placed(self) -> set[str]:
        """Files in the store outside the download folder, as store paths."""
        return {p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file() and STAGING_DIR not in p.parts}

    def url(self, model_or_component: str, path: str) -> str:
        model = self.registry.get(model_or_component)
        if model is not None:
            return next(f.url for q in model.quantizations for f in q.files if f.name == path)
        component = self.registry.component(model_or_component)
        assert component is not None
        return next(f.url for f in component.files if f.path == path)

    def size(self, model_id: str, quant: str, variant: str = "int8") -> int:
        return sum(f.size_bytes for f in plan_files(self.registry, model_id, quant, variant))

    def rows(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        return [dict(row) for row in self.db.query(sql, params)]

    def add_job(self, model_id: str, quant: str, state: str = "queued", host_id: str = "local") -> str:
        job_id = uuid4().hex
        route = json.dumps({"model_id": model_id, "quant": quant, "host_id": host_id, "engine": "wan2gp", "reasons": []})
        self.db.execute(
            "INSERT INTO jobs (id, state, priority, capability, data_class, request, route, client, created_at) "
            "VALUES (?, ?, 'interactive', 't2v', 'private', '{}', ?, 'ui', ?)",
            (job_id, state, route, to_iso(self.clock())),
        )
        return job_id

    def write_config(self, config: dict[str, Any]) -> None:
        self.store.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.store.config_path.write_bytes(json.dumps(config, indent=4).encode())

    def config(self) -> dict[str, Any]:
        return json.loads(self.store.config_path.read_text(encoding="utf-8"))


@pytest.fixture
def env(tmp_path: Path) -> Env:
    environment = Env(tmp_path)
    yield environment
    environment.db.close()


# ---------------------------------------------------------------------- install


def test_installs_z_image_int8_where_wan2gp_looks_and_merges_wgp_config(env):
    env.write_config(
        {"attention_mode": "sdpa", "checkpoints_paths": ["ckpts", "."], "text_encoder_quantization": "bf16", "save_path": "outputs"}
    )
    seen: list[InstallProgress] = []

    env.store.install("z-image-turbo", "int8", progress=seen.append)

    assert env.placed() == Z_IMAGE_INT8_FILES
    for file in plan_files(env.registry, "z-image-turbo", "int8", "int8"):
        assert (env.root / file.path).read_bytes() == env.contents[file.url]
    assert not any((env.root / STAGING_DIR).glob("*"))
    assert env.config() == {
        "attention_mode": "sdpa",
        "checkpoints_paths": [str(env.root.resolve()), "ckpts", "."],
        "text_encoder_quantization": "int8",
        "save_path": "outputs",
    }
    [row] = env.rows("SELECT * FROM installed_models")
    assert (row["model_id"], row["quant"], row["state"], row["error"]) == ("z-image-turbo", "int8", "installed", None)
    assert row["size_bytes"] == env.size("z-image-turbo", "int8") and row["last_used_at"] == row["installed_at"]
    assert {r["path"] for r in env.rows("SELECT path FROM installed_model_files")} == Z_IMAGE_INT8_FILES
    assert {r["path"] for r in env.rows("SELECT path FROM store_files")} == Z_IMAGE_INT8_FILES
    # The first install on the host stores the variant it installed.
    assert env.rows("SELECT text_encoder_variant FROM hosts")[0]["text_encoder_variant"] == "int8"
    # Weights came through the CDN redirect; progress ends at the total.
    assert env.hub.served(env.url("z-image-turbo", "ZImageTurbo_quanto_bf16_int8.safetensors")) == 1
    assert seen[-1].done_bytes == seen[-1].total_bytes == env.size("z-image-turbo", "int8")


def test_a_hash_mismatch_names_the_file_and_registers_nothing(env):
    env.write_config({"attention_mode": "sdpa", "checkpoints_paths": ["ckpts", "."], "text_encoder_quantization": "bf16"})
    url = env.url("z-image-shared", "Qwen3/tokenizer.json")
    good = env.hub.files[url]
    env.hub.files[url] = good[:-1] + b"!"  # same size, other bytes

    with pytest.raises(ModelStoreError) as failure:
        env.store.install("z-image-turbo", "int8")

    assert failure.value.code == "hash_mismatch"
    assert "Qwen3/tokenizer.json" in failure.value.message
    assert env.placed() == set()
    assert env.rows("SELECT * FROM store_files") == [] and env.rows("SELECT * FROM installed_model_files") == []
    [row] = env.rows("SELECT state, error FROM installed_models")
    assert row["state"] == "failed" and "Qwen3/tokenizer.json" in row["error"]
    assert env.config()["text_encoder_quantization"] == "bf16"  # not registered in Wan2GP either
    assert env.rows("SELECT text_encoder_variant FROM hosts")[0]["text_encoder_variant"] is None

    # A retry reuses the verified downloads and fetches only what failed.
    env.hub.files[url] = good
    env.store.install("z-image-turbo", "int8")
    assert env.placed() == Z_IMAGE_INT8_FILES
    assert env.hub.served(env.url("z-image-turbo", "ZImageTurbo_quanto_bf16_int8.safetensors")) == 1


def test_an_interrupted_download_resumes_from_its_part_file(env):
    url = env.url("z-image-turbo", "ZImageTurbo_quanto_bf16_int8.safetensors")
    env.hub.cut[url] = 20

    with pytest.raises(ModelStoreError) as failure:
        env.store.install("z-image-turbo", "int8")
    assert failure.value.code == "download_failed" and "ZImageTurbo_quanto_bf16_int8.safetensors" in failure.value.message
    assert env.placed() == set() and env.rows("SELECT * FROM store_files") == []

    env.store.install("z-image-turbo", "int8")
    assert env.placed() == Z_IMAGE_INT8_FILES
    assert [ranged for requested, ranged in env.hub.requests if httpx.URL(requested).host == CDN and requested.endswith(url[-40:])] == [
        None,
        "bytes=20-",
    ]


def test_a_redirect_to_a_host_outside_the_registry_is_refused(env):
    env.hub.redirect_to = "cdn.example.com"

    with pytest.raises(ModelStoreError) as failure:
        env.store.install("z-image-turbo", "int8")

    assert failure.value.code == "download_failed" and f"refused https://{env.hub.redirect_to}," in failure.value.message
    assert not any(httpx.URL(url).host == "cdn.example.com" for url, _ in env.hub.requests)
    assert env.placed() == set() and env.rows("SELECT * FROM installed_model_files") == []


@pytest.mark.parametrize(
    ("url", "allowed"),
    [
        ("https://huggingface.co/a/resolve/b/c.json", True),
        (f"https://{CDN}/x", True),
        ("http://huggingface.co/a/resolve/b/c.json", False),
        ("https://huggingface.co:8443/a", False),
        ("https://user@huggingface.co/a", False),
        ("https://huggingface.co.example.com/a", False),
        ("https://example.com/huggingface.co/a", False),
        ("https://cdn-lfs.example.com/a", False),
    ],
)
def test_downloads_go_only_over_https_to_the_listed_hosts(url, allowed):
    transport = AllowedHosts(httpx.MockTransport(lambda request: httpx.Response(200)), frozenset({HUB, CDN}))
    with httpx.Client(transport=transport) as client:
        if allowed:
            assert client.get(url).status_code == 200
        else:
            with pytest.raises(httpx.ConnectError):
                client.get(url)


def test_an_install_that_does_not_fit_fails_before_any_byte_is_downloaded(env):
    needed = env.size("z-image-turbo", "int8")  # shared files included
    env.free = 100

    with pytest.raises(ModelStoreError) as failure:
        env.store.install("z-image-turbo", "int8")

    assert failure.value.code == "not_enough_space"
    assert format_size(needed - 100) in failure.value.message
    assert failure.value.message.endswith(".") and ". " not in failure.value.message  # one sentence
    assert env.hub.requests == []
    assert env.rows("SELECT * FROM installed_models") == [] and env.placed() == set()


def test_the_free_space_check_counts_only_what_is_still_missing(env):
    env.store.install("wan-2.2-t2v-14b", "int8")
    shared = env.size("wan-2.2-i2v-14b", "int8") - sum(len(env.contents[env.url("wan-2.2-i2v-14b", name)]) for name in WAN_I2V_INT8)
    env.free = env.size("wan-2.2-i2v-14b", "int8") - shared  # room for the two transformer files only

    env.store.install("wan-2.2-i2v-14b", "int8")
    assert {"wan-2.2-i2v-14b", "wan-2.2-t2v-14b"} == {r["model_id"] for r in env.rows("SELECT model_id FROM installed_models")}


def test_the_host_variant_picks_the_text_encoder_files(env):
    env.db.execute("UPDATE hosts SET text_encoder_variant = 'bf16'")
    env.write_config({"attention_mode": "sdpa", "checkpoints_paths": ["ckpts", "."], "text_encoder_quantization": "int8"})

    env.store.install("z-image-turbo", "int8")

    assert env.placed() == (Z_IMAGE_INT8_FILES - {"Qwen3/qwen3_quanto_bf16_int8.safetensors"}) | {"Qwen3/qwen3_bf16.safetensors"}
    assert env.config()["text_encoder_quantization"] == "bf16"


def test_a_cancelled_install_registers_nothing_and_keeps_its_downloads(env):
    calls = iter([False, False, True])
    with pytest.raises(ModelStoreError) as failure:
        env.store.install("z-image-turbo", "int8", cancelled=lambda: next(calls, True))
    assert failure.value.code == "cancelled"
    assert env.rows("SELECT * FROM installed_models") == [] and env.placed() == set()
    assert any((env.root / STAGING_DIR).iterdir())


def test_an_install_without_a_pinned_size_is_refused(env):
    env.registry = Registry(TEST_MODELS)  # test-video pins no size
    store = env.new_store()
    with pytest.raises(ModelStoreError) as failure:
        store.prepare_install("test-video", "bf16")
    assert failure.value.code == "size_missing" and "v.safetensors" in failure.value.message
    assert env.hub.requests == []


def test_a_second_install_of_the_same_quantization_waits_for_the_first(env):
    install = env.store.prepare_install("z-image-turbo", "int8")
    with pytest.raises(ModelStoreError) as failure:
        env.store.prepare_install("z-image-turbo", "int8")
    assert failure.value.code == "installing"
    with pytest.raises(ModelStoreError) as failure:
        env.store.remove("z-image-turbo", "int8")
    assert failure.value.code == "installing"
    env.store.run_install(install)
    assert env.placed() == Z_IMAGE_INT8_FILES


def test_an_install_left_downloading_by_the_last_start_is_failed(env):
    env.store.prepare_install("z-image-turbo", "int8")
    env.new_store()  # the next core start
    [row] = env.rows("SELECT state, error FROM installed_models")
    assert row["state"] == "failed" and "closed" in row["error"]


# ---------------------------------------------------------------------- remove and shared files


def test_removing_one_of_two_models_keeps_their_shared_text_encoder(env):
    env.store.install("wan-2.2-t2v-14b", "int8")
    env.store.install("wan-2.2-i2v-14b", "int8")
    assert UMT5_INT8 in env.placed()

    assert env.store.remove("wan-2.2-t2v-14b", "int8") == [("wan-2.2-t2v-14b", "int8")]

    placed = env.placed()
    assert UMT5_INT8 in placed and set(WAN_I2V_INT8) <= placed and not set(WAN_T2V_INT8) & placed
    assert {r["path"] for r in env.rows("SELECT path FROM store_files")} == placed

    env.store.remove("wan-2.2-i2v-14b")
    assert env.placed() == set() and env.rows("SELECT * FROM store_files") == []
    assert [p for p in env.root.iterdir() if p.name != STAGING_DIR] == []  # empty subfolders go too


def test_a_model_a_job_needs_is_not_removed(env):
    env.store.install("z-image-turbo", "int8")
    for state in ("queued", "leased", "running"):
        job_id = env.add_job("z-image-turbo", "int8", state)
        with pytest.raises(ModelStoreError) as failure:
            env.store.remove("z-image-turbo", "int8")
        assert failure.value.code == "in_use"
        env.db.execute("UPDATE jobs SET state = 'succeeded' WHERE id = ?", (job_id,))
    env.store.remove("z-image-turbo", "int8")
    assert env.placed() == set()


def test_a_shared_file_a_queued_job_needs_stays_after_its_last_model_goes(env):
    env.store.install("wan-2.2-i2v-14b", "int8")
    env.add_job("wan-2.2-t2v-14b", "int8")  # needs the same text encoder, not installed

    env.store.remove("wan-2.2-i2v-14b", "int8")

    assert UMT5_INT8 in env.placed() and not set(WAN_I2V_INT8) & env.placed()


def test_removing_an_install_that_is_not_there_is_refused(env):
    with pytest.raises(ModelStoreError) as failure:
        env.store.remove("z-image-turbo", "int8")
    assert failure.value.code == "not_installed"


# ---------------------------------------------------------------------- quota


def install_three(env: Env) -> None:
    """Z-Image, Wan T2V and Wan I2V, all int8; least recently used first: T2V, I2V, Z-Image."""
    for model_id in ("z-image-turbo", "wan-2.2-t2v-14b", "wan-2.2-i2v-14b"):
        env.store.install(model_id, "int8")
    for model_id in ("wan-2.2-t2v-14b", "wan-2.2-i2v-14b", "z-image-turbo"):
        env.clock.advance(60)
        env.store.mark_used(model_id, "int8")


def store_total(env: Env) -> int:
    return env.store.usage().total_bytes


def transformer_bytes(env: Env, model_id: str, names: tuple[str, ...]) -> int:
    return sum(len(env.contents[env.url(model_id, name)]) for name in names)


def test_quota_eviction_removes_the_least_recently_used_model_first(env):
    install_three(env)
    growth = env.size("ltx-2.3-distilled-22b", "int8")
    env.quota = store_total(env) + growth - transformer_bytes(env, "wan-2.2-t2v-14b", WAN_T2V_INT8)

    install = env.store.install("ltx-2.3-distilled-22b", "int8")

    assert install.evicted == (("wan-2.2-t2v-14b", "int8"),)
    installed = {r["model_id"] for r in env.rows("SELECT model_id FROM installed_models")}
    assert installed == {"z-image-turbo", "wan-2.2-i2v-14b", "ltx-2.3-distilled-22b"}
    placed = env.placed()
    assert not set(WAN_T2V_INT8) & placed and UMT5_INT8 in placed  # the shared encoder stays for Wan I2V
    assert store_total(env) <= env.quota


def test_quota_eviction_never_removes_a_model_a_job_needs(env):
    install_three(env)
    env.add_job("wan-2.2-t2v-14b", "int8", "queued")
    growth = env.size("ltx-2.3-distilled-22b", "int8")
    env.quota = store_total(env) + growth - transformer_bytes(env, "wan-2.2-i2v-14b", WAN_I2V_INT8)

    install = env.store.install("ltx-2.3-distilled-22b", "int8")

    assert install.evicted == (("wan-2.2-i2v-14b", "int8"),)
    assert set(WAN_T2V_INT8) <= env.placed()


def test_an_install_the_quota_cannot_hold_is_refused_and_evicts_nothing(env):
    install_three(env)
    for model_id in ("z-image-turbo", "wan-2.2-t2v-14b", "wan-2.2-i2v-14b"):
        env.add_job(model_id, "int8", "running")
    before, requests = env.placed(), len(env.hub.requests)
    env.quota = store_total(env) + 10

    with pytest.raises(ModelStoreError) as failure:
        env.store.install("ltx-2.3-distilled-22b", "int8")

    assert failure.value.code == "quota_exceeded" and "quota" in failure.value.message
    assert env.placed() == before and len(env.hub.requests) == requests
    assert len(env.rows("SELECT * FROM installed_models")) == 3


# ---------------------------------------------------------------------- text encoder variant

QWEN3_INT8 = "Qwen3/qwen3_quanto_bf16_int8.safetensors"
QWEN3_BF16 = "Qwen3/qwen3_bf16.safetensors"


def test_a_variant_switch_installs_the_other_variant_and_frees_the_previous_one(env):
    env.write_config({"attention_mode": "sdpa", "checkpoints_paths": ["ckpts", "."], "text_encoder_quantization": "int8"})
    env.store.install("z-image-turbo", "int8")
    env.store.install("wan-2.2-t2v-14b", "int8")

    switch = env.store.prepare_switch("bf16")
    assert {file.path for file in switch.missing} == {QWEN3_BF16, "umt5-xxl/models_t5_umt5-xxl-enc-bf16.safetensors"}
    assert env.store.run_switch(switch) == "updated"

    # Until the worker restarted and the garbage was collected, both variants are on disk.
    assert {QWEN3_INT8, QWEN3_BF16, UMT5_INT8} <= env.placed()
    assert env.config()["text_encoder_quantization"] == "bf16"
    assert env.rows("SELECT text_encoder_variant FROM hosts")[0]["text_encoder_variant"] == "bf16"
    for model_id in ("z-image-turbo", "wan-2.2-t2v-14b"):
        planned = {f.path for f in plan_files(env.registry, model_id, "int8", "bf16")}
        assert {r["path"] for r in env.rows("SELECT path FROM installed_model_files WHERE model_id = ?", (model_id,))} == planned
        [row] = env.rows("SELECT size_bytes FROM installed_models WHERE model_id = ?", (model_id,))
        assert row["size_bytes"] == env.size(model_id, "int8", "bf16")

    env.store.collect_garbage()
    placed = env.placed()
    assert QWEN3_BF16 in placed and QWEN3_INT8 not in placed and UMT5_INT8 not in placed
    assert {r["path"] for r in env.rows("SELECT path FROM store_files")} == placed


def test_a_failed_variant_switch_changes_nothing(env):
    env.write_config({"attention_mode": "sdpa", "checkpoints_paths": ["ckpts", "."], "text_encoder_quantization": "int8"})
    env.store.install("z-image-turbo", "int8")
    url = env.url("z-image-shared", QWEN3_BF16)
    env.hub.files[url] = env.hub.files[url][:-1] + b"!"

    with pytest.raises(ModelStoreError) as failure:
        env.store.run_switch(env.store.prepare_switch("bf16"))

    assert failure.value.code == "hash_mismatch" and QWEN3_BF16 in failure.value.message
    assert env.rows("SELECT text_encoder_variant FROM hosts")[0]["text_encoder_variant"] == "int8"
    assert env.config()["text_encoder_quantization"] == "int8"
    assert env.placed() == Z_IMAGE_INT8_FILES
    env.store.collect_garbage()
    assert env.placed() == Z_IMAGE_INT8_FILES
    env.store.install("wan-2.2-t2v-14b", "int8")  # the switch no longer blocks installs


def test_a_variant_switch_excludes_installs_removals_and_running_jobs(env):
    env.store.install("z-image-turbo", "int8")
    install = env.store.prepare_install("wan-2.2-t2v-14b", "int8")
    with pytest.raises(ModelStoreError) as failure:
        env.store.prepare_switch("bf16")
    assert failure.value.code == "installing"
    env.store.run_install(install)

    switch = env.store.prepare_switch("bf16")
    for refused in (lambda: env.store.prepare_install("wan-2.2-i2v-14b", "int8"), lambda: env.store.remove("z-image-turbo")):
        with pytest.raises(ModelStoreError) as failure:
            refused()
        assert failure.value.code == "switching"
    with pytest.raises(ModelStoreError) as failure:
        env.store.prepare_switch("int8")
    assert failure.value.code == "switching"
    env.store.run_switch(switch)

    for state in ("leased", "running"):
        job_id = env.add_job("z-image-turbo", "int8", state)
        with pytest.raises(ModelStoreError) as failure:
            env.store.prepare_switch("int8")
        assert failure.value.code == "in_use"
        env.db.execute("UPDATE jobs SET state = 'succeeded' WHERE id = ?", (job_id,))
    env.add_job("z-image-turbo", "int8", "running", host_id="elsewhere")  # another host's job does not block this one
    env.store.run_switch(env.store.prepare_switch("int8"))


def test_usage_counts_each_models_files_once(env):
    env.store.install("wan-2.2-t2v-14b", "int8")
    env.store.install("wan-2.2-t2v-14b", "bf16")
    usage = env.store.usage()
    together = {f.key: f.size_bytes for q in ("int8", "bf16") for f in plan_files(env.registry, "wan-2.2-t2v-14b", q, "int8")}
    assert usage.by_model == {"wan-2.2-t2v-14b": sum(together.values())}
    assert usage.total_bytes == sum(together.values())
    assert sum(usage.installed.values()) > usage.total_bytes  # each quantization counts the shared files


# ---------------------------------------------------------------------- rules


@pytest.mark.parametrize(
    "path",
    [
        "../x.json",
        "a/../../x.json",
        "/abs.json",
        "C:/x.json",
        "C:x.json",
        "a\\b.json",
        "CON.json",
        "sub/nul.txt",
        "a/./b.json",
        "",
        "x./y.json",
        "a/b /c.json",
    ],
)
def test_registry_paths_must_stay_inside_the_store(tmp_path, path):
    with pytest.raises(ModelStoreError) as failure:
        store_path(tmp_path, path)
    assert failure.value.code == "unsafe_path"


def test_registry_paths_inside_the_store_resolve_below_it(tmp_path):
    assert store_path(tmp_path, "Qwen3/tokenizer.json") == tmp_path / "Qwen3" / "tokenizer.json"
    assert store_path(tmp_path, "ZImageTurbo_bf16.safetensors") == tmp_path / "ZImageTurbo_bf16.safetensors"


@pytest.mark.parametrize("name", ["model.bin", "model.pt", "model.pth", "model.pkl", "model.ckpt", "loader.py", "model.onnx", "README"])
def test_only_safetensors_gguf_and_pinned_companions_are_allowed(name):
    for shared in (False, True):
        with pytest.raises(ModelStoreError) as failure:
            check_file_type(name, shared=shared)
        assert failure.value.code == "file_type"


def test_companion_files_are_allowed_only_in_shared_components():
    for name in ("x.safetensors", "x.GGUF"):
        check_file_type(name, shared=False)
        check_file_type(name, shared=True)
    for name in ("tokenizer.json", "merges.txt", "spiece.model"):
        check_file_type(name, shared=True)
        with pytest.raises(ModelStoreError):
            check_file_type(name, shared=False)


def test_every_registry_file_passes_the_rules(tmp_path):
    registry = Registry.load()
    for model in registry.all():
        for quant in model.quantizations:
            for variant in ("bf16", "int8"):
                for file in plan_files(registry, model.id, quant.id, variant):  # checks types and sizes
                    store_path(tmp_path, file.path)


def test_the_plan_takes_the_files_of_one_text_encoder_variant(env):
    int8 = {f.path for f in plan_files(env.registry, "ltx-2.3-distilled-22b", "Q6_K", "int8")}
    bf16 = {f.path for f in plan_files(env.registry, "ltx-2.3-distilled-22b", "Q6_K", "bf16")}
    encoder = "gemma-3-12b-it-qat-q4_0-unquantized/"
    assert int8 - bf16 == {encoder + "gemma-3-12b-it-qat-q4_0-unquantized_quanto_bf16_int8.safetensors"}
    assert bf16 - int8 == {encoder + "gemma-3-12b-it-qat-q4_0-unquantized.safetensors"}
    assert "ltx-2.3-22b-distilled-Q6_K_light.gguf" in int8 and encoder + "tokenizer.model" in int8
    with pytest.raises(ModelStoreError):
        plan_files(env.registry, "ltx-2.3-distilled-22b", "Q6_K", "fp8")
    with pytest.raises(ModelStoreError):
        plan_files(env.registry, "ltx-2.3-distilled-22b", "fp4", "int8")


# ---------------------------------------------------------------------- Wan2GP


def test_merging_wgp_config_keeps_every_other_key(tmp_path):
    config, store = tmp_path / "wgp_config.json", tmp_path / "models"
    assert merge_wgp_config(config, store, "int8") == "missing"
    assert not config.exists()  # Wan2GP writes its full defaults only when the file is missing

    config.write_bytes(
        json.dumps({"checkpoints_paths": ["ckpts", str(store), "."], "profile": 4, "text_encoder_quantization": "bf16"}).encode()
    )
    assert merge_wgp_config(config, store, "int8") == "updated"
    merged = json.loads(config.read_text(encoding="utf-8"))
    assert merged == {"checkpoints_paths": [str(store.resolve()), "ckpts", "."], "profile": 4, "text_encoder_quantization": "int8"}
    assert merge_wgp_config(config, store, "int8") == "unchanged"
    assert b"\r\n" not in config.read_bytes()

    config.write_bytes(json.dumps({"profile": 4}).encode())  # no checkpoints_paths yet
    merge_wgp_config(config, store, "bf16")
    assert json.loads(config.read_text(encoding="utf-8")) == {
        "profile": 4,
        "checkpoints_paths": [str(store.resolve())],
        "text_encoder_quantization": "bf16",
    }

    config.write_bytes(b"[1, 2]")
    with pytest.raises(ModelStoreError) as failure:
        merge_wgp_config(config, store, "int8")
    assert failure.value.code == "config_invalid"


def test_an_install_reports_whether_wan2gp_sees_the_store(env):
    # Without a wgp_config.json Wan2GP has not started yet: the install stands, the merge waits for Wan2GP's file.
    assert env.store.run_install(env.store.prepare_install("z-image-turbo", "int8")) == "missing"
    assert not env.store.config_path.exists()
    env.write_config({"attention_mode": "sdpa", "checkpoints_paths": ["ckpts", "."]})
    assert env.store.register_in_wan2gp() == "updated"
    assert env.store.run_install(env.store.prepare_install("z-image-turbo", "int8")) == "unchanged"  # complete: nothing to download
    assert env.hub.served(env.url("z-image-turbo", "ZImageTurbo_quanto_bf16_int8.safetensors")) == 1


def test_migration_0003_keeps_existing_hosts_without_a_variant(tmp_path):
    only_0002 = tmp_path / "migrations-0002"
    only_0002.mkdir()
    for name in ("0001_initial.sql", "0002_job_outputs.sql"):
        (only_0002 / name).write_bytes((bundled_migrations() / name).read_bytes())
    db = Database(tmp_path / "db" / "old.sqlite3")
    try:
        assert db.migrate(only_0002) == ["0001_initial", "0002_job_outputs"]
        db.execute(
            "INSERT INTO hosts (id, name, kind, allowed_data_classes, created_at) VALUES ('local', 'Local', 'local', '[]', ?)",
            ("2026-11-20T00:00:00.000000+00:00",),
        )
        assert db.migrate() == ["0003_model_store"]
        assert db.query_one("SELECT text_encoder_variant FROM hosts WHERE id = 'local'")["text_encoder_variant"] is None
        db.execute("UPDATE hosts SET text_encoder_variant = 'bf16'")
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("UPDATE hosts SET text_encoder_variant = 'fp8'")
        tables = {row["name"] for row in db.query("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert {"store_files", "installed_model_files"} <= tables and "model_files" not in tables
    finally:
        db.close()


def test_the_mvp_model_types_need_no_finetune_definition(env, tmp_path):
    env.engine_root = tmp_path / "wan2gp"
    (env.engine_root / "defaults").mkdir(parents=True)
    for model_type in MVP_MODEL_TYPES:
        (env.engine_root / "defaults" / f"{model_type}.json").write_text("{}", encoding="utf-8")
    for model in env.registry.all():
        assert all(wan2gp_defines(env.engine_root, binding.model_type) for binding in model.engines)

    env.store.install("z-image-turbo", "int8")

    assert not (env.engine_root / "finetunes").exists()


def test_a_model_type_wan2gp_does_not_define_is_refused_without_an_architecture(env, tmp_path):
    env.engine_root = tmp_path / "wan2gp"
    (env.engine_root / "defaults").mkdir(parents=True)

    with pytest.raises(ModelStoreError) as failure:
        env.store.install("z-image-turbo", "int8")

    assert failure.value.code == "engine_type_unknown" and "z_image" in failure.value.message
    assert env.hub.requests == []


def test_a_finetune_definition_is_written_for_a_model_type_wan2gp_does_not_define(env, tmp_path):
    env.engine_root = tmp_path / "wan2gp"
    (env.engine_root / "defaults").mkdir(parents=True)
    store = env.new_store(architectures={"z_image": "z_image"})

    store.install("z-image-turbo", "int8")

    finetune = env.engine_root / "finetunes" / "z_image.json"
    assert json.loads(finetune.read_text(encoding="utf-8")) == {
        "model": {
            "name": "Z-Image Turbo 6B (z_image)",
            "architecture": "z_image",
            "URLs": ["ZImageTurbo_bf16.safetensors", "ZImageTurbo_quanto_bf16_int8.safetensors"],
        }
    }
    store.remove("z-image-turbo", "int8")
    assert not finetune.exists()

    # Two-expert models would need more than one file per quantization: refused.
    with pytest.raises(ModelStoreError) as failure:
        env.new_store(architectures={"t2v_2_2": "t2v_2_2"}).install("wan-2.2-t2v-14b", "int8")
    assert failure.value.code == "engine_type_unknown"
