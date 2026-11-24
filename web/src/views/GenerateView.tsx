// Generate: prompt, capability, optional model pin, data class, priority and seed, with a live
// estimate. The primary button stays disabled with a label that says what is missing.

import { Play } from 'lucide-react';
import { useEffect, useMemo, useState, type FormEvent } from 'react';

import {
  ApiError,
  api,
  isAbort,
  type Capability,
  type DataClass,
  type Estimate,
  type Job,
  type JobRequest,
  type Priority,
} from '../api/client';
import { DATA_CLASSES, DataClassTag } from '../components/DataClassTag';
import { ErrorBanner } from '../components/Feedback';
import { Light } from '../components/StatusLight';
import { DASH, formatDuration, formatEur, truncateMiddle } from '../lib/format';
import { useApi } from '../lib/useApi';

const CAPABILITIES = {
  t2i: true,
  i2i: true,
  t2v: true,
  i2v: true,
  v2v: true,
  audio: true,
  tts: true,
  lora_training: true,
} satisfies Record<Capability, true>;
const ALL_CAPABILITIES = Object.keys(CAPABILITIES) as Capability[];
const DATA_CLASS_ORDER: DataClass[] = ['public', 'private', 'biometric'];
const ESTIMATE_DELAY_MS = 300;

type EstimateState = { status: 'idle' } | { status: 'loading' } | { status: 'done'; estimate: Estimate } | { status: 'error'; error: unknown };

function parseSeed(text: string): number | null | undefined {
  const trimmed = text.trim();
  if (trimmed === '') return null;
  if (!/^\d+$/.test(trimmed)) return undefined;
  const value = Number(trimmed);
  return Number.isSafeInteger(value) ? value : undefined;
}

/** An infeasible problem (409) is an answer, not a failure: show it like `feasible: false`. */
function asEstimate(error: unknown): Estimate | null {
  return error instanceof ApiError && error.infeasible ? { feasible: false, reasons: error.reasons, requires_approval: false } : null;
}

export function GenerateView() {
  const capabilities = useApi((signal) => api.capabilities(signal));
  const models = useApi((signal) => api.models(signal));

  const [prompt, setPrompt] = useState('');
  const [capability, setCapability] = useState<Capability | ''>('');
  const [modelId, setModelId] = useState('');
  const [dataClass, setDataClass] = useState<DataClass>('private');
  const [priority, setPriority] = useState<Priority>('interactive');
  const [seedText, setSeedText] = useState('');
  const [estimate, setEstimate] = useState<EstimateState>({ status: 'idle' });
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<unknown>();
  const [submitted, setSubmitted] = useState<Job>();

  const capabilityOptions = useMemo(() => {
    const offered = new Set((capabilities.data?.items ?? []).map((entry) => entry.capability));
    const list = ALL_CAPABILITIES.filter((cap) => offered.has(cap));
    return list.length > 0 ? list : ALL_CAPABILITIES;
  }, [capabilities.data]);
  const selectedCapability: Capability = capability || capabilityOptions[0] || 't2v';

  const modelOptions = useMemo(
    () => (models.data?.items ?? []).filter((model) => model.capabilities.includes(selectedCapability)),
    [models.data, selectedCapability],
  );
  const pinned = modelOptions.some((model) => model.id === modelId) ? modelId : '';

  const seed = parseSeed(seedText);
  const trimmedPrompt = prompt.trim();
  const request: JobRequest | null = useMemo(() => {
    if (!trimmedPrompt || seed === undefined) return null;
    return {
      capability: selectedCapability,
      prompt: trimmedPrompt,
      model: pinned || null,
      priority,
      data_class: dataClass,
      params: { seed },
    };
  }, [trimmedPrompt, seed, selectedCapability, pinned, priority, dataClass]);
  const requestKey = request ? JSON.stringify(request) : '';

  useEffect(() => {
    if (!request) {
      setEstimate({ status: 'idle' });
      return;
    }
    const controller = new AbortController();
    setEstimate({ status: 'loading' });
    const timer = window.setTimeout(() => {
      api
        .estimate(request, controller.signal)
        .then((result) => setEstimate({ status: 'done', estimate: result }))
        .catch((error: unknown) => {
          if (isAbort(error)) return;
          const infeasible = asEstimate(error);
          setEstimate(infeasible ? { status: 'done', estimate: infeasible } : { status: 'error', error });
        });
    }, ESTIMATE_DELAY_MS);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
    // requestKey captures every field of the request.
  }, [requestKey]);

  let blocker: string | null = null;
  if (!trimmedPrompt) blocker = 'Enter a prompt';
  else if (seed === undefined) blocker = 'Seed must be a whole number';
  else if (submitting) blocker = 'Submitting';
  else if (estimate.status === 'loading' || estimate.status === 'idle') blocker = 'Estimating';
  else if (estimate.status === 'error') blocker = 'Estimate unavailable';
  else if (!estimate.estimate.feasible) blocker = 'No model can run this request';

  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (!request || blocker) return;
    setSubmitting(true);
    setSubmitError(undefined);
    setSubmitted(undefined);
    api
      .submitJob(request)
      .then(setSubmitted)
      .catch((error: unknown) => {
        const infeasible = asEstimate(error);
        if (infeasible) setEstimate({ status: 'done', estimate: infeasible });
        else setSubmitError(error);
      })
      .finally(() => setSubmitting(false));
  };

  return (
    <div className="view">
      <header className="view__head">
        <h1 className="view__title">Generate</h1>
      </header>

      {capabilities.error ? <ErrorBanner title="Could not load capabilities" error={capabilities.error} /> : null}
      {models.error ? <ErrorBanner title="Could not load models" error={models.error} /> : null}

      <div className="generate">
        <form className="panel stack" onSubmit={submit} noValidate>
          <label className="cw-field">
            <span className="cw-field__label">Prompt</span>
            <textarea
              className="cw-input prompt-input"
              value={prompt}
              maxLength={4000}
              rows={4}
              placeholder="Harbour at night, fog, neon reflections"
              onChange={(e) => setPrompt(e.target.value)}
            />
            <span className="cw-field__hint">Translated into the model's prompt dialect.</span>
          </label>

          <div className="form-row">
            <label className="cw-field">
              <span className="cw-field__label">Capability</span>
              <select className="cw-select cw-input--mono" value={selectedCapability} onChange={(e) => setCapability(e.target.value as Capability)}>
                {capabilityOptions.map((cap) => (
                  <option key={cap} value={cap}>
                    {cap}
                  </option>
                ))}
              </select>
            </label>

            <label className="cw-field field-grow">
              <span className="cw-field__label">Model</span>
              <select className="cw-select" value={pinned} onChange={(e) => setModelId(e.target.value)}>
                <option value="">Router chooses</option>
                {modelOptions.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name}
                  </option>
                ))}
              </select>
              <span className="cw-field__hint">Pin a model or let the router pick one.</span>
            </label>
          </div>

          <fieldset className="choice">
            <legend className="cw-field__label">Data class</legend>
            <div className="choice__options">
              {DATA_CLASS_ORDER.map((value) => (
                <label key={value} className="choice__option">
                  <input type="radio" name="data-class" value={value} checked={dataClass === value} onChange={() => setDataClass(value)} />
                  <DataClassTag value={value} />
                </label>
              ))}
            </div>
            <span className="cw-field__hint">{DATA_CLASSES[dataClass].hint}</span>
          </fieldset>

          <div className="form-row">
            <label className="cw-field">
              <span className="cw-field__label">Priority</span>
              <select className="cw-select" value={priority} onChange={(e) => setPriority(e.target.value as Priority)}>
                <option value="interactive">Interactive</option>
                <option value="batch">Batch</option>
              </select>
              <span className="cw-field__hint">Interactive jobs run before batch jobs.</span>
            </label>

            <label className="cw-field" data-invalid={seed === undefined ? 'true' : undefined}>
              <span className="cw-field__label">Seed</span>
              <input
                className="cw-input cw-input--mono seed-input"
                inputMode="numeric"
                value={seedText}
                placeholder="2814033"
                onChange={(e) => setSeedText(e.target.value)}
                aria-invalid={seed === undefined}
              />
              <span className="cw-field__hint">{seed === undefined ? 'Whole number, 0 or more' : 'Empty = random'}</span>
            </label>
          </div>

          {submitError ? <ErrorBanner title="The job was not submitted" error={submitError} /> : null}
          {submitted ? (
            <p className="cw-muted" role="status">
              Queued as job <span className="cw-mono" title={submitted.id}>{truncateMiddle(submitted.id)}</span>.{' '}
              <a href="#/queue">Open queue</a>
            </p>
          ) : null}

          <div>
            <button type="submit" className="cw-btn cw-btn--primary" disabled={blocker !== null}>
              {blocker === null ? <Play strokeWidth={1.5} aria-hidden="true" /> : null}
              {blocker ?? 'Generate'}
            </button>
          </div>
        </form>

        <EstimatePanel state={estimate} hasRequest={request !== null} />
      </div>
    </div>
  );
}

/** Exported for the component tests. */
export function EstimatePanel({ state, hasRequest }: { state: EstimateState; hasRequest: boolean }) {
  // The only live text: the outcome, not the whole panel (cost and ETA change with every keystroke).
  let outcome = '';
  if (state.status === 'error') outcome = 'Estimate unavailable';
  else if (hasRequest && state.status === 'done') outcome = state.estimate.feasible ? 'Estimate ready: can run' : 'Estimate ready: no model can run this request';
  return (
    <section className="panel stack" aria-labelledby="estimate-title">
      <h2 id="estimate-title" className="panel__title">
        Estimate
      </h2>
      <p className="sr-only" role="status">
        {outcome}
      </p>
      {!hasRequest ? <p className="cw-muted">Enter a prompt to get an estimate.</p> : null}
      {hasRequest && state.status === 'loading' ? <p className="cw-muted">Estimating</p> : null}
      {state.status === 'error' ? <ErrorBanner title="Estimate unavailable" error={state.error} /> : null}
      {state.status === 'done' ? <EstimateDetails estimate={state.estimate} /> : null}
    </section>
  );
}

function EstimateDetails({ estimate }: { estimate: Estimate }) {
  const route = estimate.route;
  const reasons = estimate.reasons ?? [];
  return (
    <>
      {estimate.requires_approval ? (
        <div className="cw-banner" role="status">
          <svg viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" aria-hidden="true">
            <path d="M9 2.5 16 15H2z" />
            <path d="M9 7.5v3.5M9 13v.01" strokeLinecap="round" />
          </svg>
          <div className="cw-banner__text">
            <p className="cw-banner__title">Approval needed</p>
            <p className="cw-banner__body">The estimated cost is above the approval threshold; the job waits until someone approves it.</p>
          </div>
        </div>
      ) : null}
      {estimate.feasible ? <Light state="ok" word="Can run" /> : <Light state="fail" word="No model can run this request" />}
      <dl className="kv">
        <dt>Model</dt>
        <dd>{route ? `${route.model_id} · ${route.quant}` : DASH}</dd>
        <dt>Host</dt>
        <dd>{route?.host_id ?? DASH}</dd>
        <dt>Engine</dt>
        <dd>{route?.engine ?? DASH}</dd>
        <dt>Cost</dt>
        <dd>{formatEur(estimate.cost_eur)}</dd>
      </dl>
      <div>
        <div className="text-caption cw-muted">ETA</div>
        <div className="text-value-lg estimate__eta">{formatDuration(estimate.eta_seconds)}</div>
      </div>
      {reasons.length > 0 ? (
        <div>
          <div className="text-caption cw-muted">Router decisions</div>
          <ol className="reasons">
            {reasons.map((reason, index) => (
              <li key={index}>{reason}</li>
            ))}
          </ol>
        </div>
      ) : null}
    </>
  );
}
