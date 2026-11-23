// Error banner and empty state.

import type { ReactNode } from 'react';

import { errorMessage } from '../api/client';
import { FailShape } from './StatusLight';

export function ErrorBanner({ title, error, action }: { title: string; error: unknown; action?: ReactNode }) {
  return (
    <div className="cw-banner" data-tone="fail" role="alert">
      <FailShape />
      <div className="cw-banner__text">
        <p className="cw-banner__title">{title}</p>
        <p className="cw-banner__body">{errorMessage(error)}</p>
      </div>
      {action ? <div className="cw-banner__actions">{action}</div> : null}
    </div>
  );
}

export function RetryButton({ onClick }: { onClick: () => void }) {
  return (
    <button type="button" className="cw-btn cw-btn--secondary cw-btn--sm" onClick={onClick}>
      Try again
    </button>
  );
}

/** `display` is for setup steps and empty states only, once per view. */
export function EmptyState({ title, display = false, children }: { title: string; display?: boolean; children?: ReactNode }) {
  return (
    <div className="empty">
      {display ? <h2 className="empty__display text-display">{title}</h2> : <h2 className="empty__title text-heading">{title}</h2>}
      {children ? <div className="empty__body">{children}</div> : null}
    </div>
  );
}
