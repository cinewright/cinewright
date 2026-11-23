// Queue: jobs from /jobs, kept live through the event stream, with cancel.

import { useCallback, useRef, useState } from 'react';

import { api, type Job, type JobEvent } from '../api/client';
import { useEventsConnected, useEventsEverConnected, useJobEvents } from '../api/events';
import { EmptyState, ErrorBanner, RetryButton } from '../components/Feedback';
import { JobRow } from '../components/JobRow';
import { Light } from '../components/StatusLight';
import { useApi } from '../lib/useApi';

function sortJobs(jobs: Iterable<Job>): Job[] {
  return [...jobs].sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at));
}

function upsert(jobs: Job[], job: Job): Job[] {
  const map = new Map(jobs.map((j) => [j.id, j]));
  map.set(job.id, job);
  return sortJobs(map.values());
}

/** Merge a progress event into a job; the data shape is engine-defined, so read it defensively. */
function withProgress(job: Job, data: Record<string, unknown>): Job {
  const percent = typeof data.percent === 'number' ? data.percent : null;
  if (percent === null) return job;
  const num = (value: unknown) => (typeof value === 'number' ? value : null);
  return {
    ...job,
    progress: {
      phase: typeof data.phase === 'string' ? data.phase : (job.progress?.phase ?? 'running'),
      step: num(data.step) ?? job.progress?.step ?? null,
      total_steps: num(data.total_steps) ?? job.progress?.total_steps ?? null,
      percent,
    },
    eta_seconds: num(data.eta_seconds) ?? job.eta_seconds ?? null,
  };
}

export function QueueView() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [cursor, setCursor] = useState<string | null>(null);
  const [loadingMore, setLoadingMore] = useState(false);
  const [actionError, setActionError] = useState<unknown>();
  const [cancelling, setCancelling] = useState<ReadonlySet<string>>(new Set());
  const lastSeq = useRef(new Map<string, number>());
  const connected = useEventsConnected();
  const hasConnected = useEventsEverConnected();

  const list = useApi(async (signal) => {
    const page = await api.jobs(undefined, signal);
    setJobs(sortJobs(page.items));
    setCursor(page.next_cursor ?? null);
    return page;
  });

  // One request per job at a time; events arriving meanwhile mark it dirty for one more fetch, so a
  // late response can never overwrite a newer state.
  const inflight = useRef(new Map<string, boolean>());
  const refetch = useCallback(function fetchJob(jobId: string) {
    if (inflight.current.has(jobId)) {
      inflight.current.set(jobId, true);
      return;
    }
    inflight.current.set(jobId, false);
    api
      .job(jobId)
      .then((job) => setJobs((current) => upsert(current, job)))
      .catch(() => undefined) // the next event or a reload brings the job back
      .finally(() => {
        const dirty = inflight.current.get(jobId);
        inflight.current.delete(jobId);
        if (dirty) fetchJob(jobId);
      });
  }, []);

  useJobEvents((event: JobEvent) => {
    const seen = lastSeq.current.get(event.job_id) ?? 0;
    if (event.seq <= seen) return; // replayed after a reconnect
    lastSeq.current.set(event.job_id, event.seq);
    const known = jobs.find((job) => job.id === event.job_id);
    if (!known || event.kind === 'state') {
      refetch(event.job_id);
      return;
    }
    if (event.kind === 'progress') setJobs((current) => current.map((job) => (job.id === event.job_id ? withProgress(job, event.data) : job)));
  });

  // A running job stops at the worker's next check, so the row stays "Cancelling" until its
  // terminal state arrives; the id is released only when the request fails.
  const cancel = (job: Job) => {
    setActionError(undefined);
    setCancelling((current) => new Set(current).add(job.id));
    api
      .cancelJob(job.id)
      .then((updated) => setJobs((current) => upsert(current, updated)))
      .catch((error: unknown) => {
        setActionError(error);
        setCancelling((current) => {
          const next = new Set(current);
          next.delete(job.id);
          return next;
        });
      });
  };

  const loadMore = () => {
    if (!cursor) return;
    setLoadingMore(true);
    api
      .jobs(cursor)
      .then((page) => {
        setJobs((current) => sortJobs(new Map([...current, ...page.items].map((j) => [j.id, j])).values()));
        setCursor(page.next_cursor ?? null);
      })
      .catch(setActionError)
      .finally(() => setLoadingMore(false));
  };

  return (
    <div className="view">
      <header className="view__head">
        <h1 className="view__title">Queue</h1>
        {connected ? (
          <Light state="ok" word="Live" />
        ) : (
          <span className="cw-status cw-muted">{hasConnected ? 'Reconnecting · updates paused' : 'Connecting · updates paused'}</span>
        )}
      </header>

      {list.error ? <ErrorBanner title="Could not load the queue" error={list.error} action={<RetryButton onClick={list.reload} />} /> : null}
      {actionError ? <ErrorBanner title="The request failed" error={actionError} /> : null}

      {!list.loading && !list.error && jobs.length === 0 ? (
        <EmptyState title="No jobs yet">
          <p>
            Jobs from Generate and from agents over MCP appear here. <a href="#/generate">Open Generate</a>
          </p>
        </EmptyState>
      ) : null}

      {jobs.length > 0 ? (
        <div className="list" role="list" aria-label="Jobs">
          {jobs.map((job) => (
            <div role="listitem" key={job.id}>
              <JobRow job={job} onCancel={cancel} cancelling={cancelling.has(job.id)} />
            </div>
          ))}
        </div>
      ) : null}

      {cursor ? (
        <div>
          <button type="button" className="cw-btn cw-btn--secondary" onClick={loadMore} disabled={loadingMore}>
            {loadingMore ? 'Loading' : 'Load more jobs'}
          </button>
        </div>
      ) : null}
    </div>
  );
}
