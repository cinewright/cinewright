Cinewright is a darkroom for moving pictures: a local tool that makes models, hardware, jobs and results visible and verifiable. The design follows from that. The interface is dark so images and clips stay the brightest things on screen. It is dense because users compare measurements. Technical values are set in monospace because people copy them. And there are exactly two signal colours with fixed meanings: violet means "this is your action", safelight amber means "a human decides here".

## Principles

1. **The result is the brightest thing in the room.** Surfaces stay within the `surface` steps; colour belongs to previews, actions and states. No decorative gradients, no background images.
2. **Measurements over promises.** Wherever the system knows a number (VRAM, seconds per step, ETA, cost), the number appears in `value` or `value-lg`. Adjectives like "fast" are replaced by the status light with a measurement next to it.
3. **Amber is reserved for humans.** `safelight` appears only where a person has to look or decide: cost approval, biometric data, amber status. Anything that needs no decision gets no amber.
4. **Every state has a word and a shape.** Colour never carries meaning alone. The status light has a disc, a half disc and a cross; data classes have a globe, a lock and a face frame.
5. **Agents and humans see the same thing.** Whatever an agent starts over MCP shows up in Queue and Library like a human's job, with the agent's name set in `value`.

## Voice and copy

- English is the default language. Model, format and parameter names keep their upstream spelling (`t2v`, `Q5_K_M`, `fp8`, `seed`).
- Sentence case everywhere: buttons, tabs, table headers, titles. No all caps, not even in labels.
- Buttons say what happens, as a verb phrase: "Generate", "Add host", "Cancel job", "Approve · €0.84". No "OK", no "Let's go".
- Messages state cause and consequence in one sentence: "Out of memory at step 12", "Last contact 4 minutes ago. Two jobs are waiting; the queue is moving them to local." No apologies, no exclamation marks, no emoji.
- Address the user directly only where it matters (consent for real faces). Otherwise impersonal and brief.
- Numbers: decimal point, units after a space ("11.8 GB", "4.1 s/step"), currency symbol first ("€0.84"). Durations as `mm:ss`, timestamps in ISO 8601.

## Colour

Dark is the primary theme; Light is worked out to the same standard. Every text pair reaches at least 4.5:1 in both themes; control borders, the focus ring and marks reach at least 3:1.

- **Surfaces:** `surface-000` for the page ground, `surface-100` for cards and rows, `surface-200` for hover and dialogs, `surface-sunken` for inputs, provenance blocks and progress tracks. Depth comes from these steps; cards have no shadow.
- **Lines:** `line` separates structure (card border, table row). `line-strong` outlines anything people operate (input, secondary button).
- **Text:** `ink-strong` for titles, names and key figures, `ink` for body text, `ink-muted` for labels and metadata. `ink-faint` only for placeholders and disabled elements.
- **Violet, the reference colour:** `accent` fills primary buttons and active selection, always with `on-accent`. On dark surfaces `accent` is too weak as text or line; use `accent-ink` (links, active tabs, hashes) and `accent-mark` (progress, borders of selected cards) instead. Selected cards and Private tags sit on `accent-deep`.
- **Safelight:** `safelight` fills the approve button (with `on-safelight`), `safelight-deep` carries approval banners and Biometric tags, `safelight-ink` is the text variant.
- **Status:** `status-ok` (cyan), `status-slow` (alias of `safelight`), `status-fail` (red), with `status-fail-deep` behind errors. Green is deliberately cyan so "runs" and "does not run" stay distinct with red-green colour blindness.
- **Focus:** `focus-ring`, 2px solid, 2px offset, on every focusable element. Never remove it.

At most one primary button per view. Never use violet or amber as fill for large areas outside selection and approval states.

## Typography

Two families, both openly licensed (SIL OFL) and loaded from Google Fonts: Geist for the interface, Geist Mono for anything that is a value. On Windows they fall back to Segoe UI and Cascadia Mono.

- `display` once per view, only in setup steps and empty states.
- `title` for the view title, `heading` for card titles and model names.
- `body` is the default (14/20, deliberately dense), `label` for buttons, fields, tabs and table headers, `caption` for metadata.
- `value` for VRAM, quantization, seeds and parameters, `value-lg` for key figures in cards (ETA, s/step), `hash` for hashes, IDs and paths. Monospace always with tabular figures (`font-variant-numeric: tabular-nums`) so columns don't jump.
- Truncate hashes in the middle (`sha256:9f2c41d0…77b2e41a`); full value in the tooltip and via a copy button.

## Spacing, radii, grid

- 4px grid: `space-1` to `space-8`. Cards and panels pad with `space-4`, table rows with `space-3` vertically, sections sit `space-6` apart, the workspace margin is `space-8`.
- `radius-sm` for tags, fields and code blocks, `radius-md` for buttons, `radius-lg` for cards, panels and dialogs, `radius-pill` only for status dots, progress bars and counters.
- Model, recipe and asset cards sit in a grid with a `space-4` gutter, three to four columns at 1440px.
- `shadow-overlay` only for floating layers (dialog, popover, menu).

## Components

Components are CSS classes prefixed `cw-` in `components/bundle.css`, independent of the UI framework. Each component's README describes the markup the consumer provides.

- **Button** for actions, **Field** for inputs.
- **StatusLight** for the calibration status, **DataClassTag** for data classes and capabilities.
- **ModelCard** and **PresetTile** as cards in the Models and Generate views.
- **JobRow** as a queue row, **ProvenancePanel** for an asset's origin.
- **Banner** for approvals and errors.

## Iconography

Lucide as the icon system (ISC licence): 16px in controls, 12px in tags, 40px in preset tiles, 1.5 stroke, colour always `currentColor` from the parent. The status shapes (filled disc, half disc, disc with cross) are Cinewright's own and belong to the state; never replace them with other icons. No emoji, no coloured illustrations in the interface.

## Motion

Restrained and functional: 120ms for hover and focus, 200ms for dialogs and panels, ease-out. Progress bars move only when real progress is reported; no endless shimmer. `prefers-reduced-motion` turns all transitions off.

## Logo

The logo is a single film frame with its sprocket perforations, set beside the wordmark "Cinewright" in Geist SemiBold. Use the files in the Logos group as they are; never retype the wordmark in a live font, recolour the mark outside the listed variants, or add effects.

- `cinewright-logo-on-dark.svg` on `surface-000` and other dark grounds, `cinewright-logo-on-light.svg` on light grounds.
- `cinewright-mark.svg` where space is tight (app header, About dialog, social avatars), `cinewright-favicon.svg` for 16px and the browser tab.
- `cinewright-mark-white.svg` and `cinewright-mark-black.svg` for single-colour use: photos, print, embossing.
- Clear space around the logo: the width of one perforation rail (a quarter of the mark's width) on every side. Minimum size: mark 16px, full logo 96px wide.
