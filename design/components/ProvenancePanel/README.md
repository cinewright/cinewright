# ProvenancePanel

Shows an asset's provenance record and exports it as sidecar JSON.

**Markup:** `<section class="cw-prov">` with `__head` (title, export button) and `__body` as a `<dl>`. Values in monospace; hashes with `class="is-hash"` in `accent-ink`, truncated in the middle, full value in `title`.

Field order as in the specification: model (with quantization and engine version), model hash, recipe with version, seed, LoRAs with weight, host, timestamp (ISO 8601), parent assets. Missing fields stay with "–" so records remain comparable.
