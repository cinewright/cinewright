# PresetTile

Selectable tile for a preset (camera move, style) in the Generate view.

**Markup:** `<button class="cw-preset" aria-pressed="false">` with `__frame` (16:9, 40px icon or preview loop), `__name` and `__meta` (duration · fps in monospace).

- Selection via `aria-pressed="true"`: `accent-deep`, `accent-mark` border, icon in `accent-ink`.
- A preview, when present, replaces the icon and only plays on hover or focus.
- Presets that won't run with the selected model are hidden, not disabled.
