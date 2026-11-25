// Hash routing (#/queue, #/library/<sha256>), so the core needs no history fallback to serve the UI.

import { useEffect, useState } from 'react';

/** The route's path segments: `#/library/<sha256>` gives `['library', '<sha256>']`. */
export function hashSegments(hash: string = window.location.hash): string[] {
  const path = hash.replace(/^#\/?/, '').split('?')[0] ?? '';
  return path.split('/');
}

/** The route's path segments, updated on every hash change: a view stays mounted while only its sub-path changes. */
export function useHashSegments(): string[] {
  const [segments, setSegments] = useState(() => hashSegments());
  useEffect(() => {
    const onChange = () => setSegments(hashSegments());
    window.addEventListener('hashchange', onChange);
    return () => window.removeEventListener('hashchange', onChange);
  }, []);
  return segments;
}
