# ModelCard

Card for a registry model in the Models and Calibration views, with status light, capabilities and measurements.

**Markup:** `<article class="cw-model-card" tabindex="0" aria-selected="false">` with
`__head` (name in `__name`, origin in `__vendor`, a StatusLight without detail on the right),
`__caps` (capability chips `cw-cap`),
`__stats` as a `<dl>` with three values (VRAM, Quant, s/step) and
`__foot` (licence and context in `caption`, a small secondary button on the right).

- One card per model and quantization; the same model in two quantizations appears twice.
- Selection via `aria-selected="true"`: `accent-deep` with an `accent-mark` border.
- Unmeasured values show "–", never an estimate.
- The licence is always visible; models geo-restricted from the EU do not appear.
