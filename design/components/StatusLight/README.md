# StatusLight

Shows a calibration result or a host's state as a status light with word, shape and measurement.

**Markup:** `<span class="cw-status" data-state="ok|slow|fail">` with the status SVG (14px), the state word and optionally `<span class="cw-status__detail">· measurement</span>`.

- `ok`: filled disc, `status-ok`. "Runs", detail: s/step or ETA.
- `slow`: half disc, `safelight-ink`. "Runs slowly", detail: the reason (offloading, RAM needed).
- `fail`: disc with cross, `status-fail`. "Won't run", detail: what is missing (VRAM needed, architecture unsupported).

The detail is a calibration measurement, never an estimate. Without a measurement the component shows no state, only "Not calibrated" in `ink-muted`.
