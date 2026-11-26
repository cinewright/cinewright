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
  /** The full value the button copies; also the tooltip of `text` unless `title` is given. */
  value: string;
  /** The button's accessible name, saying what it copies: "Copy job id". */
  label: string;
  /** The button's tooltip when it should say more than `label`; defaults to `label`. */
  tooltip?: string;
  title?: string;
  /** Classes of the shortened value, for example its type style. */
  className?: string;
}

export function CopyValue({ text, value, label, tooltip, title, className }: CopyValueProps) {
  const [state, setState] = useState<CopyState>('idle');
  // A new attempt mounts the message anew, so a second failure is announced again.
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    if (state !== 'copied') return undefined;
    const timer = window.setTimeout(() => setState('idle'), COPIED_MS);
    return () => window.clearTimeout(timer);
  }, [state]);

  const onCopy = () => {
    // Clear first, so a repeated copy changes the status text and is announced again.
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
      <span className={['copy-value__text', className].filter(Boolean).join(' ')} title={title ?? value}>
        {text}
      </span>
      <button type="button" className="cw-btn cw-btn--ghost cw-btn--sm copy-value__button" aria-label={label} title={tooltip ?? label} onClick={onCopy}>
        <Icon strokeWidth={1.5} aria-hidden="true" />
      </button>
      {/* Present before the first copy, so the first "Copied" is read. */}
      <span className="sr-only" role="status">
        {state === 'copied' ? 'Copied' : ''}
      </span>
      {state === 'failed' ? (
        <span key={attempt} className="copy-value__error" role="alert">
          {COPY_FAILED}
        </span>
      ) : null}
    </span>
  );
}
