// The job Generate submitted: progress and preview frames from the app's event stream, and a link that
// opens each output in the Library (#/library/<sha256>) once the job succeeded.

import { useCallback, useEffect, useRef, useState } from 'react';

import {
  UPLOAD_IMAGE_TYPES,
  api,
  isSha256,
  type Job,
  type JobError,
  type JobEvent,
  type JobState,
  type Progress,
} from '../api/client';
import { useEventsConnected, useJobEvents } from '../api/events';
import { truncateMiddle } from '../lib/format';

const JOB_STATES = {
  awaiting_approval: true,
  queued: true,
  leased: true,
  running: true,
  succeeded: true,
  failed: true,
  cancelled: true,
} satisfies Record<JobState, true>;
const TERMINAL: ReadonlySet<JobState> = new Set<JobState>(['succeeded', 'failed', 'cancelled']);
// The core passes preview data through unchecked; render only the image types it accepts for uploads.
const PREVIEW_TYPES: ReadonlySet<string> = new Set(UPLOAD_IMAGE_TYPES);
const BASE64 = /^[A-Za-z0-9+/]+={0,2}$/;
const BUFFER_LIMIT = 200;

export interface TrackedJob {
  id: string;
  state: JobState;
  progress: Progress | null;
  error: JobError | null;
  outputs: string[];
  /** The latest allowed preview frame as a data URL. */
  preview: string | null;
  /** Highest seq of the stored events applied; replayed events at or below it are skipped. */
  lastSeq: number;
}

function fromJob(job: Job): TrackedJob {
  return {
    id: job.id,
    state: job.state,
    progress: job.progress ?? null,
    error: job.error ?? null,
    outputs: job.outputs ?? [],
    preview: null,
    lastSeq: 0,
  };
}

function previewSource(data: Record<string, unknown>): string | null {
  const type = data.media_type;
  const base64 = data.data_base64;
  if (typeof type !== 'string' || !PREVIEW_TYPES.has(type)) return null;
  if (typeof base64 !== 'string' || !BASE64.test(base64)) return null;
  return `data:${type};base64,${base64}`;
}

function readProgress(data: Record<string, unknown>, current: Progress | null): Progress | null {
  if (typeof data.percent !== 'number') return current;
  const num = (value: unknown) => (typeof value === 'number' ? value : null);
  return {
    phase: typeof data.phase === 'string' ? data.phase : (current?.phase ?? 'running'),
    step: num(data.step),
    total_steps: num(data.total_steps),
    percent: Math.max(0, Math.min(100, data.percent)),
  };
}

/** The error of a `failed` or `cancelled` state event: its JobError, else the scheduler's `reason`. */
function readError(state: JobState, data: Record<string, unknown>): JobError | null {
  if (state !== 'failed' && state !== 'cancelled') return null;
  const error = data.error;
  if (typeof error === 'object' && error !== null && typeof (error as Partial<JobError>).message === 'string') {
    const { code, message, retryable } = error as Partial<JobError>;
    return { code: String(code ?? state), message: message as string, retryable: Boolean(retryable) };
  }
  return typeof data.reason === 'string' ? { code: state, message: data.reason, retryable: false } : null;
}

/** Apply one event of the job. Previews are live only and repeat the latest stored seq, so they skip the seq check. */
export function applyEvent(job: TrackedJob, event: JobEvent): TrackedJob {
  if (event.kind === 'preview') {
    const preview = previewSource(event.data);
    return preview ? { ...job, preview } : job;
  }
  if (event.seq <= job.lastSeq) return job; // replayed after a reconnect
  const next = { ...job, lastSeq: event.seq };
  if (event.kind === 'progress') return { ...next, progress: readProgress(event.data, job.progress) };
  if (event.kind === 'state') {
    const raw = event.data.state;
    if (typeof raw !== 'string' || !(raw in JOB_STATES)) return next;
    const state = raw as JobState;
    const outputs = Array.isArray(event.data.outputs) ? event.data.outputs.filter(isSha256) : job.outputs;
    return { ...next, state, outputs, error: readError(state, event.data) };
  }
  return next;
}

/**
 * Follow one submitted job. `begin` before the submit request buffers events of jobs not known yet, because the
 * worker may report before the answer arrives; `track` applies the buffered events of the new job.
 */
export function useTrackedJob() {
  const [job, setJob] = useState<TrackedJob | null>(null);
  const buffering = useRef(false);
  const buffer = useRef<JobEvent[]>([]);
  const current = useRef<TrackedJob | null>(null);
  current.current = job;
  const connected = useEventsConnected();

  useJobEvents((event) => {
    if (current.current?.id === event.job_id) {
      setJob((value) => (value && value.id === event.job_id ? applyEvent(value, event) : value));
    } else if (buffering.current && buffer.current.length < BUFFER_LIMIT) {
      buffer.current.push(event);
    }
  });

  // A stream the provider had to open anew sends no replay; load the job again whenever the stream opens.
  useEffect(() => {
    const tracked = current.current;
    if (!connected || !tracked || TERMINAL.has(tracked.state)) return;
    api
      .job(tracked.id)
      .then((loaded) =>
        setJob((value) => {
          if (!value || value.id !== loaded.id || TERMINAL.has(value.state)) return value;
          return { ...value, state: loaded.state, progress: loaded.progress ?? value.progress, error: loaded.error ?? null, outputs: loaded.outputs ?? [] };
        }),
      )
      .catch(() => undefined); // the next event brings the state
  }, [connected]);

  const begin = useCallback(() => {
    buffering.current = true;
    buffer.current = [];
  }, []);
  const track = useCallback((submitted: Job) => {
    const pending = buffer.current.filter((event) => event.job_id === submitted.id);
    const tracked = pending.reduce(applyEvent, fromJob(submitted));
    buffering.current = false;
    buffer.current = [];
    current.current = tracked; // events that arrive before the next render already belong to this job
    setJob(tracked);
  }, []);
  const abandon = useCallback(() => {
    buffering.current = false;
    buffer.current = [];
  }, []);

  return { job, begin, track, abandon };
}

function stateLabel(job: TrackedJob): string {
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
      return job.error?.message ?? 'Cancelled';
  }
}

function announcement(job: TrackedJob): string {
  if (job.state === 'succeeded') return 'Job done';
  if (job.state === 'failed') return 'Job failed';
  if (job.state === 'cancelled') return 'Job cancelled';
  return '';
}

export function JobPanel({ job }: { job: TrackedJob }) {
  const terminal = TERMINAL.has(job.state);
  const value = job.state === 'succeeded' ? 100 : Math.round(job.progress?.percent ?? 0);
  const label = stateLabel(job);
  const waiting = job.state === 'queued' || job.state === 'awaiting_approval';
  const failed = job.state === 'failed' || job.state === 'cancelled';
  const outputs = job.state === 'succeeded' ? job.outputs : [];
  return (
    <section className="panel stack" aria-labelledby="job-title">
      <div className="panel__head">
        <h2 id="job-title" className="panel__title">
          Job
        </h2>
        <span className="text-caption cw-muted cw-mono" title={job.id}>
          {truncateMiddle(job.id)}
        </span>
      </div>
      {/* Only the outcome is announced; progress and frames change too often. */}
      <p className="sr-only" role="status">
        {announcement(job)}
      </p>
      <div className="asset__frame">
        {job.preview ? (
          <img src={job.preview} alt="Latest preview frame" />
        ) : (
          <span className="text-caption">{terminal ? 'No preview frame' : 'Waiting for a preview frame'}</span>
        )}
      </div>
      <div>
        <div
          className="cw-progress"
          data-state={waiting ? 'waiting' : failed ? 'fail' : undefined}
          role="progressbar"
          aria-label={label}
          aria-valuenow={value}
          aria-valuemin={0}
          aria-valuemax={100}
        >
          <span style={{ width: `${value}%` }} />
        </div>
        <div className="cw-progress__label">
          <span className="job__state" title={label}>
            {label}
          </span>
          <span>{value} %</span>
        </div>
      </div>
      <p className="job-links">
        {outputs.map((sha256, index) => (
          <a key={sha256} href={`#/library/${sha256}`}>
            {outputs.length > 1 ? `Open output ${index + 1} in Library` : 'Open in Library'}
          </a>
        ))}
        <a href="#/queue">Open queue</a>
      </p>
    </section>
  );
}
