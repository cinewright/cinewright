# JobRow

Queue row: one job with preview, origin, progress, ETA and one action.

**Markup:** `<div class="cw-job">` with five children in this order: `__thumb` (56×32, last preview frame or empty), title and `__meta` (model · host · preview/batch), progress (`cw-progress` with `<span style="width:N%">` and `cw-progress__label`), `__eta` in monospace, one action as a small button.

- `cw-progress` without `data-state`: running, fill `accent-mark`. `data-state="waiting"`: waiting, `line-strong`. `data-state="fail"`: aborted, `status-fail`; the label names the cause.
- Set `role="progressbar"` with `aria-valuenow`.
- ETA as `mm:ss`; estimated with a tilde (`~06:30`), unknown as "–".
- Agent jobs name the agent in the meta text.
