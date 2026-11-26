// The job Generate submitted: progress and preview frames from the app's event stream, then the finished result in
// place of the last preview, and a link that opens each output in the Library (#/library/<sha256>) once the job
// succeeded.

import { useCallback, useEffect, useRef, useState } from 'react';

import {
  UPLOAD_IMAGE_TYPES,
  api,
  isAbort,
  isSha256,
  type Asset,
  type Job,
  type JobError,
  type JobEvent,
  type JobState,
  type Progress,
} from '../api/client';
import { useEventsConnected, useJobEvents } from '../api/events';
import { truncateMiddle } from '../lib/format';
import { AssetMedia } from './AssetMedia';
import { ErrorBanner } from './Feedback';

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
  /** Seq of the latest state event applied, so a reload answered after a newer state event does not undo it. */
  lastStateSeq: number;
  /** Seq of the latest progress event applied, so a reload answered after newer progress keeps the live progress. */
  lastProgressSeq: number;
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
    lastStateSeq: 0,
    lastProgressSeq: 0,
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
  if (event.kind === 'progress') {
    const progress = readProgress(event.data, job.progress);
    return progress === job.progress ? next : { ...next, progress, lastProgressSeq: event.seq };
  }
  if (event.kind === 'state') {
    const raw = event.data.state;
    if (typeof raw !== 'string' || !(raw in JOB_STATES)) return next;
    const state = raw as JobState;
    const outputs = Array.isArray(event.data.outputs) ? event.data.outputs.filter(isSha256) : job.outputs;
    return { ...next, state, outputs, error: readError(state, event.data), lastStateSeq: event.seq };
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
  const reloads = useRef(0);
  const connected = useEventsConnected();

  useJobEvents((event) => {
    if (current.current?.id === event.job_id) {
      setJob((value) => (value && value.id === event.job_id ? applyEvent(value, event) : value));
    } else if (buffering.current && buffer.current.length < BUFFER_LIMIT) {
      buffer.current.push(event);
    }
  });

  // A stream the provider had to open anew sends no replay; load the job again whenever the stream opens. Only the
  // latest reload applies, since an older one may answer last. The answer may predate events that arrive while it is
  // on its way: after a state event the live state wins, after a progress event the live progress. Without them the
  // answer is the newest word, also for a job that was queued again.
  useEffect(() => {
    const tracked = current.current;
    if (!connected || !tracked || TERMINAL.has(tracked.state)) return;
    const { lastStateSeq, lastProgressSeq } = tracked;
    const reload = ++reloads.current;
    api
      .job(tracked.id)
      .then((loaded) => {
        if (reload !== reloads.current) return;
        setJob((value) => {
          if (!value || value.id !== loaded.id || TERMINAL.has(value.state) || value.lastStateSeq > lastStateSeq) return value;
          const progress = value.lastProgressSeq > lastProgressSeq ? value.progress : (loaded.progress ?? value.progress);
          return { ...value, state: loaded.state, progress, error: loaded.error ?? null, outputs: loaded.outputs ?? [] };
        });
      })
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
      return 'Failed'; // the cause shows in full below the progress bar
    case 'cancelled':
      return 'Cancelled';
  }
}

/**
 * The job's state in words for a live status region that exists before the job does (a region added with its text
 * is often not read). Progress and frames change too often to announce; a failure speaks through its error banner.
 */
export function jobAnnouncement(job: TrackedJob): string {
  switch (job.state) {
    case 'awaiting_approval':
      return 'Job submitted, awaiting approval';
    case 'queued':
      return 'Job queued';
    case 'leased':
      return 'Job starting';
    case 'running':
      return 'Job running';
    case 'succeeded':
      return 'Job done';
    case 'failed':
      return '';
    case 'cancelled':
      return job.error ? `Job cancelled: ${job.error.message}` : 'Job cancelled';
  }
}

type ResultLookup = { sha256: string; asset: Asset } | { sha256: string; failed: true };

/**
 * The asset record of a succeeded job's first output, for its media type: the asset, `failed` when the lookup failed,
 * or null while it loads and when there is no output. Keyed by the output, so a later job never shows an older result.
 */
function useResult(sha256: string | null): ResultLookup | null {
  const [lookup, setLookup] = useState<ResultLookup | null>(null);
  useEffect(() => {
    if (!sha256) return;
    const controller = new AbortController();
    api
      .asset(sha256, controller.signal)
      .then((asset) => {
        if (!controller.signal.aborted) setLookup({ sha256, asset });
      })
      .catch((error: unknown) => {
        if (!isAbort(error) && !controller.signal.aborted) setLookup({ sha256, failed: true });
      });
    return () => controller.abort();
  }, [sha256]);
  return lookup && lookup.sha256 === sha256 ? lookup : null;
}

export function JobPanel({ job }: { job: TrackedJob }) {
  const terminal = TERMINAL.has(job.state);
  const value = job.state === 'succeeded' ? 100 : Math.round(job.progress?.percent ?? 0);
  const label = stateLabel(job);
  const waiting = job.state === 'queued' || job.state === 'awaiting_approval';
  const failed = job.state === 'failed' || job.state === 'cancelled';
  const outputs = job.state === 'succeeded' ? job.outputs : [];
  // Outputs of a loaded job arrive unchecked; only an asset id builds a path.
  const first = outputs[0];
  const result = useResult(isSha256(first) ? first : null);
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
      {/* The last preview stays while the result loads, and when it does not load. */}
      <div className="asset__frame">
        {result && 'asset' in result ? (
          <AssetMedia asset={result.asset} controls />
        ) : job.preview ? (
          <img src={job.preview} alt="Latest preview frame" />
        ) : (
          <span className="text-caption">{terminal ? 'No preview frame' : 'Waiting for a preview frame'}</span>
        )}
      </div>
      {result && 'failed' in result ? <p className="text-caption cw-muted">The result could not be shown here; it is in the Library.</p> : null}
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
      {job.state === 'failed' ? <ErrorBanner title="The job failed" error={job.error?.message ?? 'The core gave no reason.'} /> : null}
      {job.state === 'cancelled' && job.error ? <p className="job__reason">{job.error.message}</p> : null}
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
