// Copy rules from design/BRAND.md: decimal point, units after a space, currency symbol first,
// durations as mm:ss, timestamps in ISO 8601, hashes truncated in the middle.

export const DASH = '–';

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return DASH;
  const total = Math.round(seconds);
  const minutes = Math.floor(total / 60);
  return `${String(minutes).padStart(2, '0')}:${String(total % 60).padStart(2, '0')}`;
}

export function formatMb(mb: number | null | undefined): string {
  if (mb == null) return DASH;
  return `${(mb / 1024).toFixed(1)} GB`;
}

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null) return DASH;
  if (bytes < 1024) return `${bytes} B`;
  const units = ['KB', 'MB', 'GB', 'TB'];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(1)} ${units[unit]}`;
}

export function formatEur(value: number | null | undefined): string {
  if (value == null) return DASH;
  return `€${value.toFixed(2)}`;
}

export function formatSecondsPerStep(value: number | null | undefined): string {
  if (value == null) return DASH;
  return `${value.toFixed(1)} s/step`;
}

/** ISO 8601 as the API sends it, without fractional seconds. */
export function formatTimestamp(iso: string | null | undefined): string {
  if (!iso) return DASH;
  return iso.replace(/(T\d{2}:\d{2}:\d{2})\.\d+/, '$1');
}

export function truncateMiddle(text: string, head = 8, tail = 8): string {
  return text.length <= head + tail + 1 ? text : `${text.slice(0, head)}…${text.slice(-tail)}`;
}

/** `sha256:9f2c41d0…77b2e41a`; pair it with the full value in `title`. */
export function shortHash(hex: string): string {
  return `sha256:${truncateMiddle(hex)}`;
}

export function fullHash(hex: string): string {
  return `sha256:${hex}`;
}
