// A value shown shortened, with its full value in the tooltip and an icon button that copies the full value.
// Every hash or id the UI shortens goes through this component, so none of them is only readable in a tooltip.

import { Check, Copy } from 'lucide-react';
import { useEffect, useState } from 'react';

/** How long the check icon replaces the copy icon after a successful copy. */
export const COPIED_MS = 2000;
export const COPY_FAILED = 'Clipboard unavailable, nothing was copied.';

type CopyState = 'idle' | 'copied' | 'failed';

export interface CopyValueProps {
  /** The shortened value on screen. */
  text: string;
  /** The full value the button copies; also the tooltip of `text`. */
  value: string;
  /** The button's accessible name, saying what it copies: "Copy job id". */
  label: string;
  /** The button's tooltip when it should say more than `label`; defaults to `label`. */
  tooltip?: string;
  /** Classes of the shortened value, for example its type style. */
  className?: string;
}

export function CopyValue({ text, value, label, tooltip, className }: CopyValueProps) {
  const [state, setState] = useState<CopyState>('idle');
  // Each attempt mounts "Copied" or the failure message anew, so a repeated one is announced again, and restarts the
  // check icon's time. Both hold when the clipboard answers before React has rendered the reset to idle.
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    if (state !== 'copied') return undefined;
    const timer = window.setTimeout(() => setState('idle'), COPIED_MS);
    return () => window.clearTimeout(timer);
  }, [state, attempt]);

  const onCopy = () => {
    setState('idle');
    setAttempt((count) => count + 1);
    const clipboard = navigator.clipboard as Clipboard | undefined; // missing outside secure contexts
    if (!clipboard || typeof clipboard.writeText !== 'function') {
      setState('failed');
      return;
    }
    let pending: Promise<void>;
    try {
      pending = clipboard.writeText(value);
    } catch {
      setState('failed');
      return;
    }
    pending.then(
      () => setState('copied'),
      () => setState('failed'),
    );
  };

  const Icon = state === 'copied' ? Check : Copy;
  return (
    <span className="copy-value">
      {/* Value and button share a row that never wraps: the value shortens further and the button stays beside it. */}
      <span className="copy-value__row">
        <span className={['copy-value__text', className].filter(Boolean).join(' ')} title={value}>
          {text}
        </span>
        <button type="button" className="cw-btn cw-btn--ghost cw-btn--sm copy-value__button" aria-label={label} title={tooltip ?? label} onClick={onCopy}>
          <Icon strokeWidth={1.5} aria-hidden="true" />
        </button>
      </span>
      {/* Present before the first copy, so the first "Copied" is read. */}
      <span className="sr-only" role="status">
        <span key={attempt}>{state === 'copied' ? 'Copied' : ''}</span>
      </span>
      {/* Under the row and never wider than it, so nothing beside the value moves. */}
      {state === 'failed' ? (
        <span key={attempt} className="copy-value__error" role="alert">
          {COPY_FAILED}
        </span>
      ) : null}
    </span>
  );
}
