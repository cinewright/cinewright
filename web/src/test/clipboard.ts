// A navigator.clipboard stand-in for component tests (jsdom has none). Removed again when the test finishes.

import { onTestFinished } from 'vitest';

/**
 * `ok` records each written text, `refuse` rejects the write as a browser without permission does, `throw` throws from
 * `writeText` before any promise exists, `no-write` offers a clipboard without `writeText`, `missing` leaves
 * `navigator.clipboard` undefined as outside a secure context. Returns the texts written.
 */
export function stubClipboard(mode: 'ok' | 'refuse' | 'throw' | 'no-write' | 'missing' = 'ok'): string[] {
  const writes: string[] = [];
  let clipboard: object | undefined;
  if (mode === 'no-write') clipboard = {};
  else if (mode !== 'missing')
    clipboard = {
      writeText(text: string): Promise<void> {
        if (mode === 'throw') throw new DOMException('Document is not focused.', 'NotAllowedError');
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
