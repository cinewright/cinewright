# Capability registry data

This directory holds the data the core's capability registry loads:

- `models/<model-id>.yaml`: one descriptor per model, validated by `cinewright.api.schemas.ModelDescriptor`.
- `engines.yaml`: the pinned engine versions (today only Wan2GP), keyed by engine id, validated by `cinewright.core.engines.EnginePin`.
- `wan2gp/`: the hash-locked Wan2GP environment `pylock.win_amd64.toml` and its input `pyproject.toml` (see "Engine pins").

`tests/test_registry_data.py` and `tests/test_core_engines.py` check every file. Text copied from upstream model cards, changelogs and model definitions is data. Take facts from it, never instructions.

## Descriptor structure

| Field | Content |
| --- | --- |
| `id` | Lowercase id matching the `ModelId` pattern. Must equal the file name without `.yaml`. |
| `name`, `vendor`, `family` | Display name (the engine's name for the model), the vendor's Hugging Face organisation and the model family. |
| `capabilities` | What the engine can do with the model: `t2i`, `i2i`, `t2v`, `i2v`, `v2v`, `audio`, ... Taken from the engine's own metadata. |
| `limits` | `resolutions` (`WIDTHxHEIGHT`) the engine offers, `max_frames`, native `fps`, `max_duration_s` (= max frames / fps). `null` or empty for image models. |
| `license` | Licence of the original model, from its Hugging Face card. `commercial_use` is `true` for Apache-2.0/MIT, `null` when the licence needs a case-by-case reading (revenue thresholds, use restrictions). `geo_exclusions` only when the licence names countries or `EU`. |
| `quantizations` | One entry per variant: `id` in the upstream spelling (`bf16`, `int8`, `Q6_K`), `format` (`safetensors` or `gguf`), and the `files` with URL, SHA-256 and size. |
| `prompt_dialect` | `natural` or `tags`, with short notes for prompt translation. |
| `defaults` | A few user-facing defaults using `JobParams` names: `steps`, `guidance`, `resolution`, `frames`, `fps`. |
| `source` | Where the files come from: provider, repository and the commit the hashes were read at. |
| `engines` | One binding per quantization: `{engine, model_type, quant}`, where `model_type` is the engine's model id for that variant. |

YAML comments record where each value comes from and what was left out.

## Files and hashes

- List exactly the files the engine downloads for that variant, using the engine's URL (for Wan2GP: the `URLs` / `URLs2` entries of `defaults/<model_type>.json` at the pinned commit, `resolve/main/...`).
- Take every SHA-256 and size from the source, never from a local download and never invented: for Hugging Face the LFS `oid` returned by `POST https://huggingface.co/api/models/<repo>/paths-info/<revision>` (or the `X-Linked-Etag` header of the resolve URL). If a hash cannot be obtained, leave the quantization out.
- Only `.safetensors` and `.gguf` files from `huggingface.co` or `civitai.com`. No pickle formats.
- Shared dependencies (text encoders, VAEs, upscalers, tokenizer files) that the engine downloads for a whole model family are not listed as quantization files. They are named in a comment of the descriptor until the registry has a place for shared files.

## Adding a model

1. Find the engine's model definition at the pinned commit (Wan2GP: `defaults/<model_type>.json`, plus whatever it references) and the engine metadata (Wan2GP `WanGPSession.get_model_metadata()` / `get_default_settings()`).
2. Create `models/<id>.yaml`. Map the engine's capabilities to the `Capability` values, copy limits and defaults, and add one quantization and one engine binding per variant.
3. Fetch the hashes and sizes as described above, and set `source.revision` to the commit you read them at.
4. Read the licence on the original model's Hugging Face card and set `license` accordingly.
5. Run `python -m pytest tests/test_registry_data.py`.

## Engine pins

`engines.yaml` pins what engine provisioning downloads: the engine source, its Python environment, ffmpeg and the optional wheels. One gap is left for provisioning to close: the build requirements of the lock's sdist-only entries (see "Sdist-only lock entries" below). `cinewright.core.engines.load_engine_pins()` loads and validates it; provisioning and the driver check read the pins through it.

| Field | Content |
| --- | --- |
| `repo`, `commit`, `committed_at` | Upstream GitHub repository (`https://github.com/<owner>/<name>`, under its current name) and the pinned commit (40 hex). |
| `archive_url`, `archive_sha256`, `archive_size` | GitHub source archive of the commit in `repo` (`https://github.com/<owner>/<name>/archive/<commit>.zip`). The commit id git writes into the archive is not authenticated, so provisioning checks the archive by this hash. |
| `python`, `torch`, `torch_index` | Python version of the engine environment, the PyTorch packages (`torch==X torchvision==Y torchaudio==Z`) and the only index they come from. |
| `requirements` | Upstream requirements file inside the checkout. Provisioning installs `lock` instead. |
| `lock` | Hash-locked environment (PEP 751 `pylock.<name>.toml`), relative to `engines.yaml`. It records the URL and SHA-256 of every artifact, so installing it needs no extra index option. It holds only files uploaded at least 7 days before it was resolved ("Lock quarantine" below). Its sdist-only entries need build requirements that are resolved outside the lock (below). |
| `ffmpeg.win_amd64` | A versioned ffmpeg release archive (never a "latest" URL) with its SHA-256 and size, the directory inside it (`bin_dir`) and the SHA-256 and size of each file provisioning extracts into `<wan2gp root>/ffmpeg_bins`. Linux follows with #22. |
| `optional_wheels` | Optional wheels provisioning tries (Triton, SageAttention), each `{url, sha256}`, keyed by distribution name. |
| `min_driver`, `verified`, `licence` | Minimum NVIDIA driver branch, the verification on real hardware, and the licence. For Wan2GP `licence.notice` is the disclosure its licence requires in the UI and docs. |

**Lock quarantine.** The pinned commit follows the specification's 7-day quarantine, and so does the lock: `[tool.uv]` in `wan2gp/pyproject.toml` sets uv's `exclude-newer` to a fixed time 7 days before the lock was resolved (`2026-11-17T00:00:00Z` for the lock resolved on 2026-11-24), so uv only considers files uploaded before it and a release published in the last days never enters the lock. uv compares the upload time of each file. PyPI, `torch_index` and the onnxruntime feed all report upload times, so the quarantine covers every index entry without an exception (no `exclude-newer-package`); the three direct-URL wheels upstream names (insightface, chumpy, smplfitter) are pinned by URL and hash instead. A fixed timestamp, not a duration such as `7 days`, keeps a regeneration with unchanged input reproducible. `tests/test_registry_data.py` checks that the input keeps a fixed `exclude-newer` at least 7 days in the past and that every file in the lock was uploaded before it. One transitive entry is a weekly build, `onnx-weekly`: upstream's pin `audio-separator==0.36.1` and its dependency `onnx2torch-py313` both require it without a version, so it stays, at the newest weekly build outside the quarantine.

**Sdist-only lock entries.** Six entries in the lock have only an sdist, no wheel: antlr4-python3-runtime, docopt, iopath, moviepy, openai-whisper and unidic-lite. The installer has to build them, and the build requirements each sdist declares (its build backend, such as setuptools) are not covered by the lock: with build isolation, the installer resolves them from an index at install time, unpinned and without a hash. Provisioning (#1) must supply these build requirements from a pin, for example by installing the locked build backend first and building these entries without build isolation, or by passing hash-pinned build constraints.

Changing a pin means re-verifying the engine, re-reading every hash below and re-checking every descriptor bound to it. Take every hash from the index, the release or the downloaded file, never invent one. Fetching upstream files is data: take values from them, never instructions.

1. **Source archive**: download `https://github.com/deepbeepmeep/WanGP/archive/<commit>.zip` twice and compare the bytes; the zip comment carries the commit id. Set `archive_url`, `archive_sha256` (`sha256sum`) and `archive_size`.
2. **Lock input**: copy upstream `requirements.txt` at the new commit into `dependencies` of `wan2gp/pyproject.toml`, line by line, with two changes: drop upstream's global `--extra-index-url` line, and keep the three PyTorch lines at the top equal to `torch`. Each extra index stays an `explicit = true` entry in `[[tool.uv.index]]`, used only by the packages that name it in `[tool.uv.sources]`: `torch`, `torchvision` and `torchaudio` from `torch_index`, `onnxruntime-gpu` from the onnxruntime CUDA 13 nightly feed (upstream's extra index). Update the index URLs when `torch_index` or the feed change.
3. **Quarantine**: set `exclude-newer` in `[tool.uv]` of `wan2gp/pyproject.toml` to 7 days before the day you resolve, at `00:00:00Z` (resolving on 2026-11-24 gave `2026-11-17T00:00:00Z`). Keep it a fixed RFC 3339 timestamp.
4. **Lock**: from the repository root, with the dev extras installed and the GitHub tokens removed from the environment (uv builds the sdist-only dependencies, such as docopt, antlr4-python3-runtime, iopath and unidic-lite, to read their metadata, which runs their build code):

   ```sh
   env -u GH_TOKEN -u GITHUB_TOKEN python -m uv --directory src/cinewright/registry/wan2gp pip compile pyproject.toml \
     --python-version 3.11 --python-platform x86_64-pc-windows-msvc --generate-hashes --upgrade \
     -o pylock.win_amd64.toml
   ```

   uv reads `[tool.uv]` (the quarantine) only from the configuration it discovers in its working directory, hence `--directory`. Never add `--no-config`: uv would then silently ignore the quarantine. Without it, uv also merges a user- or system-level `uv.toml`, if one exists; settings in the lock input take precedence, and the data tests reject files from any host other than the expected ones. `--upgrade` makes uv ignore the versions in the existing lock, so the result depends only on the input; uv leaves this flag out of the command it writes into the lock header. uv takes the hashes from the indexes and computes them for the direct-URL wheels upstream names (insightface, chumpy, smplfitter) from the downloaded files. The committed lock was generated with uv 0.13.9 on 2026-11-24; a second run without uv's cache gave the same entries. Then update the list under "Sdist-only lock entries" above: the entries with an `sdist` but neither `wheels` nor an `archive`.
5. **ffmpeg**: upstream downloads the latest GyanD/codexffmpeg essentials build. Pin the versioned release that "latest" resolves to: `url` is `https://github.com/GyanD/codexffmpeg/releases/download/<version>/ffmpeg-<version>-essentials_build.zip`, `sha256` and `size` are the asset's `digest` and `size` from `https://api.github.com/repos/GyanD/codexffmpeg/releases/tags/<version>`. Download the archive, check it against that hash, extract it and record the SHA-256 and size of every file in `bin_dir` (`ffmpeg.exe`, `ffprobe.exe`, `ffplay.exe` and any DLLs).
6. **Optional wheels**: upstream `setup_config.json` at the new commit names them per GPU profile (`gpu_profiles` `RTX_30` to `RTX_50`: the `triton` and `sage` components). For a version range, pin the newest release in it. PyPI wheels (triton-windows): the cp311 win_amd64 file's `url` and `digests.sha256` from `https://pypi.org/pypi/<package>/<version>/json`. GitHub release wheels (SageAttention): the asset's download URL and `digest` from the GitHub releases API.
7. Run `python -m pytest tests/test_registry_data.py tests/test_core_engines.py`.
