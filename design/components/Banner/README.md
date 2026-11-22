# Banner

Message above a view: approval needed (safelight) or error.

**Markup:** `<div class="cw-banner" role="status">` with icon (18px), `__text` (`__title`, `__body`) and `__actions`. Errors: `data-tone="fail"` and `role="alert"`.

- Approval: names the trigger (agent or recipe), target host, estimated cost and data class; actions "Decline" (ghost) and "Approve" (approve).
- Error: cause and consequence in two short sentences, one action that helps.
- Banners stack, at most three visible; the rest collect in the queue.
