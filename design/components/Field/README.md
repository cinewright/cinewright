# Field

Input or select with label and hint.

**Markup:** `<label class="cw-field">` with `__label`, `cw-input` or `cw-select` and an optional `__hint`. For seeds, frames, resolutions and other numbers add `cw-input--mono`. Errors via `data-invalid="true"` on the field; the hint then states the limit ("Max. 121 at 720p").

- Limits come from the selected model's registry entry, never from values hard-coded in the UI.
- Placeholders show an example, never instructions; those go in `__hint`.
