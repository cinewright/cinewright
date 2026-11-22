# Button

Triggers an action; five variants, two sizes.

**Markup:** `<button class="cw-btn cw-btn--primary">` with an optional Lucide icon (16px) before the label. Small size with `cw-btn--sm` (26px tall, for cards and table rows).

- `cw-btn--primary`: the one main action of the view (Generate, Calibrate). `accent` with `on-accent`.
- `cw-btn--secondary`: further actions. `surface-100` with a `line-strong` border.
- `cw-btn--ghost`: minor or repeated actions in lists (Cancel, Move up).
- `cw-btn--approve`: human approvals only, with the amount in the label ("Approve · €0.84"). `safelight` with `on-safelight`.
- `cw-btn--danger`: destructive actions (Cancel job, Remove model). Permanent deletion also needs a confirmation dialog.
- `disabled`: the label says what is missing ("No model selected"), never just a greyed-out "Generate".

Don't: place two primary buttons side by side; use `approve` for actions without cost or data consequences.
