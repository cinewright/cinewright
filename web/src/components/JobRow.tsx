// Queue row: preview, origin, progress, ETA and one action (design/components/JobRow).

import type { Job, JobState } from '../api/client';
import { DASH, formatDuration } from '../lib/format';

const TERMINAL: ReadonlySet<JobState> = new Set(['succeeded', 'failed', 'cancelled']);

export function isTerminal(job: Job): boolean {
  return TERMINAL.has(job.state);
}

function progressState(state: JobState): 'waiting' | 'fail' | undefined {
  if (state === 'queued' || state === 'awaiting_approval') return 'waiting';
  if (state === 'failed' || state === 'cancelled') return 'fail';
  return undefined;
}

function stateLabel(job: Job): string {
  const progress = job.progress;
  switch (job.state) {
    case 'awaiting_approval':
      return 'Awaiting approval';
    case 'queued':
      return 'Queued';
    case 'leased':
      return 'Starting';
    case 'running':
      if (progress?.step != null && progress.total_steps != null) return `Step ${progress.step}/${progress.total_steps}`;
      return progress?.phase || 'Running';
    case 'succeeded':
      return 'Done';
    case 'failed':
      return job.error?.message ?? 'Failed';
    case 'cancelled':
      return 'Cancelled';
  }
}

function percent(job: Job): number {
  if (job.state === 'succeeded') return 100;
  return Math.max(0, Math.min(100, job.progress?.percent ?? 0));
}

function eta(job: Job): string {
  if (isTerminal(job) || job.eta_seconds == null) return DASH;
  const value = formatDuration(job.eta_seconds);
  return job.state === 'running' ? value : `~${value}`;
}

function meta(job: Job): string {
  const model = job.route ? `${job.route.model_id} · ${job.route.quant}` : (job.request.model ?? 'model by router');
  const parts = [model, job.route?.host_id ?? 'no host yet', job.request.priority ?? 'interactive'];
  if (job.agent) parts.push(`agent ${job.agent}`);
  return parts.join(' · ');
}

export function JobRow({ job, onCancel, cancelling }: { job: Job; onCancel: (job: Job) => void; cancelling: boolean }) {
  const value = percent(job);
  const label = stateLabel(job);
  return (
    <div className="cw-job">
      <div className="cw-job__thumb" />
      <div className="job__text">
        <div className="cw-job__title job__title" title={job.request.prompt}>
          {job.request.prompt}
        </div>
        <div className="cw-job__meta">{meta(job)}</div>
      </div>
      <div>
        <div
          className="cw-progress"
          data-state={progressState(job.state)}
          role="progressbar"
          aria-label={label}
          aria-valuenow={Math.round(value)}
          aria-valuemin={0}
          aria-valuemax={100}
        >
          <span style={{ width: `${value}%` }} />
        </div>
        <div className="cw-progress__label">
          <span className="job__state" title={label}>
            {label}
          </span>
          <span>{Math.round(value)} %</span>
        </div>
      </div>
      <div className="cw-job__eta">{eta(job)}</div>
      {isTerminal(job) ? (
        <span />
      ) : (
        <button type="button" className="cw-btn cw-btn--ghost cw-btn--sm" onClick={() => onCancel(job)} disabled={cancelling}>
          {cancelling ? 'Cancelling' : 'Cancel'}
        </button>
      )}
    </div>
  );
}
