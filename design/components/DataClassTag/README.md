# DataClassTag

Marks the data class of a job, asset or cast entry; includes the capability chip `cw-cap`.

**Markup:** `<span class="cw-tag" data-class="public|private|biometric">` with icon (12px) and word. Capabilities: `<span class="cw-cap">t2v</span>` in monospace, spelled as in the registry.

- `public`, "Public": globe, neutral. All workers allowed.
- `private`, "Private": lock, `accent-deep`/`accent-ink`. Local and own remote hosts.
- `biometric`, "Biometric": face frame, `safelight-deep`/`safelight-ink`. Local only, approval per host.

Show the tag wherever the data class affects routing: job details, cast cards, host policies, approval banners.
