// A navigator.clipboard stand-in for component tests (jsdom has none). Removed again when the test finishes.

import { onTestFinished } from 'vitest';

/**
 * `ok` records each written text, `refuse` rejects the write as a browser without permission does, `missing` leaves
 * `navigator.clipboard` undefined as outside a secure context. Returns the texts written.
 */
export function stubClipboard(mode: 'ok' | 'refuse' | 'missing' = 'ok'): string[] {
  const writes: string[] = [];
  const clipboard =
    mode === 'missing'
      ? undefined
      : {
          writeText(text: string): Promise<void> {
            if (mode === 'refuse') return Promise.reject(new DOMException('Write permission denied.', 'NotAllowedError'));
            writes.push(text);
            return Promise.resolve();
          },
        };
  Object.defineProperty(window.navigator, 'clipboard', { value: clipboard, configurable: true });
  onTestFinished(() => {
    Reflect.deleteProperty(window.navigator, 'clipboard');
  });
  return writes;
}
