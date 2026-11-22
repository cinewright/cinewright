# Design system

Source of truth for the web UI's look, copied from the Cinewright design system (claude.ai artifact "Cinewright", files under `project/`).
Change the design system first, then copy the changed files here; the web build reads only this folder.

| File | Use |
| --- | --- |
| `BRAND.md` | Brand book: principles, voice and copy, colour, typography, spacing, iconography, motion, logo rules |
| `tokens.json` | Tokens (colour themes dark/light, type, spacing, radius, shadow). `web/scripts/gen-tokens.mjs` turns it into CSS custom properties. |
| `components.css` | Component classes `cw-*` (Button, Field, StatusLight, DataClassTag, ModelCard, PresetTile, JobRow, ProvenancePanel, Banner) |
| `components/<Name>/` | Markup guidelines (`README.md`) and a reference preview (`preview.html`) per component |
| `logos/` | Logo files; use them unchanged (see `logos/README.md`) |
