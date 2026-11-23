import { Copy } from 'lucide-react';
import { useState } from 'react';

import { fullHash, shortHash } from '../lib/format';

/** Hash truncated in the middle, full value in the tooltip and through the copy button. */
export function Hash({ value, copy = false }: { value: string; copy?: boolean }) {
  const [copied, setCopied] = useState(false);
  const full = fullHash(value);
  const onCopy = () => {
    navigator.clipboard
      .writeText(full)
      .then(() => setCopied(true))
      .catch(() => setCopied(false));
  };
  return (
    <span className="hash">
      <span className="text-hash hash__value" title={full}>
        {shortHash(value)}
      </span>
      {copy ? (
        <button type="button" className="cw-btn cw-btn--ghost cw-btn--sm" onClick={onCopy}>
          <Copy strokeWidth={1.5} aria-hidden="true" />
          {copied ? 'Copied' : 'Copy hash'}
        </button>
      ) : null}
    </span>
  );
}
