// Generate: prompt, capability, optional model pin, input image, resolution, frames and frame rate, data class,
// priority and seed, with a live estimate; then the submitted job's progress, preview frames and Library link.
// The primary button stays disabled with a label that says what is missing.
//
// Field limits come from the pinned model, or from the model the latest estimate routed to (CapabilityEntry.limits).
// A snapped value changes the request, so the estimate, the route and the limits can change again. The request
// therefore holds only what the user set (empty = model default), and a route change adjusts it at most once per
// user edit (see `snapArmed`); a frame count left off the grid blocks the request instead of snapping again.

import { Play } from 'lucide-react';
import { useEffect, useId, useMemo, useRef, useState, type FormEvent } from 'react';

import {
  ApiError,
  api,
  imageTypeProblem,
  isAbort,
  type Asset,
  type Capability,
  type DataClass,
  type Estimate,
  type JobRequest,
  type Priority,
  type Schemas,
} from '../api/client';
import { DATA_CLASSES, DataClassTag } from '../components/DataClassTag';
import { ErrorBanner } from '../components/Feedback';
import { ImageInputField } from '../components/ImageInputField';
import { JobPanel, useTrackedJob } from '../components/JobPanel';
import { Light } from '../components/StatusLight';
import { DASH, formatDuration, formatEur } from '../lib/format';
import { IMAGE_CAPABILITIES, IMAGE_INPUT_CAPABILITIES, describeGrid, frameGrid, onGrid, parseFrames, snapFrames } from '../lib/limits';
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
  const [resolution, setResolution] = useState('');
  const [framesText, setFramesText] = useState('');
  // A number field reports text it cannot read as an empty value; `badInput` tells the two apart.
  const [framesUnreadable, setFramesUnreadable] = useState(false);
  const [fps, setFps] = useState('');
  const [image, setImage] = useState<Asset | null>(null);
  const [uploading, setUploading] = useState<string | null>(null);
  const [imageError, setImageError] = useState<unknown>();
  const [dataClass, setDataClass] = useState<DataClass>('private');
  const [priority, setPriority] = useState<Priority>('interactive');
  const [seedText, setSeedText] = useState('');
  const [estimate, setEstimate] = useState<EstimateState>({ status: 'idle' });
  const [routed, setRouted] = useState<{ capability: Capability; modelId: string } | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<unknown>();
  const tracked = useTrackedJob();
  // True after a user edit; a snap caused by a route change clears it, so that snap cannot start another one.
  const snapArmed = useRef(true);
  const uploadCount = useRef(0);
  const framesLabel = useId();
  const framesHint = useId();

  /** Wrap a setter of a field the user edits. */
  const edit =
    <T,>(set: (value: T) => void) =>
    (value: T) => {
      snapArmed.current = true;
      set(value);
    };

  const capabilityOptions = useMemo(() => {
    const offered = new Set((capabilities.data?.items ?? []).map((entry) => entry.capability));
    const list = ALL_CAPABILITIES.filter((cap) => offered.has(cap));
    return list.length > 0 ? list : ALL_CAPABILITIES;
  }, [capabilities.data]);
  const selectedCapability: Capability = capability || capabilityOptions[0] || 't2v';
  const stillImage = IMAGE_CAPABILITIES.has(selectedCapability);
  const needsImage = IMAGE_INPUT_CAPABILITIES.has(selectedCapability);
  const imageLabel = selectedCapability === 'i2v' ? 'Start image' : 'Input image';

  const modelOptions = useMemo(
    () => (models.data?.items ?? []).filter((model) => model.capabilities.includes(selectedCapability)),
    [models.data, selectedCapability],
  );
  const pinned = modelOptions.some((model) => model.id === modelId) ? modelId : '';

  // The model whose limits the fields offer: the pinned one, else the one the latest estimate routed to.
  const limitsModelId = pinned || (routed?.capability === selectedCapability ? routed.modelId : '');
  const limitsEntry = useMemo(
    () => (capabilities.data?.items ?? []).find((entry) => entry.capability === selectedCapability && entry.model_id === limitsModelId),
    [capabilities.data, selectedCapability, limitsModelId],
  );
  const limits = limitsEntry?.limits ?? null;
  const grid = useMemo(() => (limits ? frameGrid(limits) : null), [limits]);

  const seed = parseSeed(seedText);
  const frames = parseFrames(framesText);
  let framesBlocker: string | null = null; // still images have no frames
  if (!stillImage && (frames === undefined || framesUnreadable)) framesBlocker = 'Frames must be a whole number';
  else if (!stillImage && typeof frames === 'number' && grid && !onGrid(frames, grid)) framesBlocker = `Frames must be ${describeGrid(grid)}`;
  const trimmedPrompt = prompt.trim();

  const request: JobRequest | null = useMemo(() => {
    // A frame count off the grid stops the estimate rather than being snapped again (no route loop).
    if (!trimmedPrompt || seed === undefined || framesBlocker) return null;
    const params: Schemas['JobParams'] = { seed };
    if (resolution) params.resolution = resolution;
    if (!stillImage) {
      if (typeof frames === 'number') params.frames = frames;
      if (fps) params.fps = Number(fps);
    }
    return {
      capability: selectedCapability,
      prompt: trimmedPrompt,
      model: pinned || null,
      priority,
      data_class: dataClass,
      inputs: needsImage && image ? [image.sha256] : [],
      params,
    };
  }, [trimmedPrompt, seed, framesBlocker, resolution, stillImage, frames, fps, selectedCapability, pinned, priority, dataClass, needsImage, image]);
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
        .then((result) => {
          setEstimate({ status: 'done', estimate: result });
          if (result.feasible && result.route) setRouted({ capability: request.capability, modelId: result.route.model_id });
        })
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

  // When the fields' model changes, drop choices it does not offer. Resolution and frame rate fall back to the model
  // default, which every model accepts, so they settle at once. Frames snap to the new grid at most once per user edit.
  const limitsKey = limits ? `${selectedCapability}:${limitsModelId}` : '';
  useEffect(() => {
    if (!limits) return;
    setResolution((value) => (value && !(limits.resolutions ?? []).includes(value) ? '' : value));
    setFps((value) => (value && !(limits.fps ?? []).includes(Number(value)) ? '' : value));
    if (!snapArmed.current || stillImage || !grid) return;
    const value = parseFrames(framesText);
    if (typeof value !== 'number' || onGrid(value, grid)) return;
    snapArmed.current = false;
    setFramesText(String(snapFrames(value, grid)));
    // Only a change of the model (or of its limits) snaps; edits snap when the user leaves the field.
  }, [limitsKey, limits]);

  const commitFrames = () => {
    const value = parseFrames(framesText);
    if (typeof value === 'number' && grid && !onGrid(value, grid)) setFramesText(String(snapFrames(value, grid)));
  };

  const chooseImage = (file: File) => {
    snapArmed.current = true;
    const problem = imageTypeProblem(file);
    if (problem) {
      setImageError(new Error(problem));
      return;
    }
    const count = ++uploadCount.current;
    setImageError(undefined);
    setUploading(file.name);
    api
      .uploadImage(file, dataClass)
      .then((asset) => {
        if (count === uploadCount.current) setImage(asset);
      })
      .catch((error: unknown) => {
        if (count === uploadCount.current) setImageError(error);
      })
      .finally(() => {
        if (count === uploadCount.current) setUploading(null);
      });
  };
  const removeImage = () => {
    snapArmed.current = true;
    setImage(null);
    setImageError(undefined);
  };

  let blocker: string | null = null;
  if (!trimmedPrompt) blocker = 'Enter a prompt';
  else if (seed === undefined) blocker = 'Seed must be a whole number';
  else if (framesBlocker) blocker = framesBlocker;
  else if (needsImage && uploading !== null) blocker = `Uploading the ${imageLabel.toLowerCase()}`;
  else if (needsImage && !image) blocker = `Add ${selectedCapability === 'i2v' ? 'a start' : 'an input'} image`;
  else if (submitting) blocker = 'Submitting';
  else if (estimate.status === 'loading' || estimate.status === 'idle') blocker = 'Estimating';
  else if (estimate.status === 'error') blocker = 'Estimate unavailable';
  else if (!estimate.estimate.feasible) blocker = 'No model can run this request';

  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (!request || blocker) return;
    setSubmitting(true);
    setSubmitError(undefined);
    tracked.begin();
    api
      .submitJob(request)
      .then(tracked.track)
      .catch((error: unknown) => {
        tracked.abandon();
        const infeasible = asEstimate(error);
        if (infeasible) setEstimate({ status: 'done', estimate: infeasible });
        else setSubmitError(error);
      })
      .finally(() => setSubmitting(false));
  };

  const limitsNote = !limitsEntry
    ? 'Pin a model, or enter a prompt so the router picks one; the choices follow its limits.'
    : pinned
      ? `Choices of ${limitsEntry.model_name}.`
      : `Choices of ${limitsEntry.model_name}, the model the router picked.`;

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
              onChange={(e) => edit(setPrompt)(e.target.value)}
            />
            <span className="cw-field__hint">Translated into the model's prompt dialect.</span>
          </label>

          <div className="form-row">
            <label className="cw-field">
              <span className="cw-field__label">Capability</span>
              <select
                className="cw-select cw-input--mono"
                value={selectedCapability}
                onChange={(e) => edit(setCapability)(e.target.value as Capability)}
              >
                {capabilityOptions.map((cap) => (
                  <option key={cap} value={cap}>
                    {cap}
                  </option>
                ))}
              </select>
            </label>

            <label className="cw-field field-grow">
              <span className="cw-field__label">Model</span>
              <select className="cw-select" value={pinned} onChange={(e) => edit(setModelId)(e.target.value)}>
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

          {needsImage ? (
            <ImageInputField label={imageLabel} image={image} uploading={uploading} error={imageError} onChoose={chooseImage} onRemove={removeImage} />
          ) : null}

          <div className="stack stack--tight">
            <div className="form-row">
              <label className="cw-field">
                <span className="cw-field__label">Resolution</span>
                <select
                  className="cw-select cw-input--mono"
                  value={resolution}
                  disabled={!limits}
                  onChange={(e) => edit(setResolution)(e.target.value)}
                >
                  <option value="">Model default</option>
                  {(limits?.resolutions ?? []).map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </select>
              </label>

              {stillImage ? null : (
                <>
                  <label className="cw-field" data-invalid={framesBlocker ? 'true' : undefined}>
                    <span className="cw-field__label" id={framesLabel}>
                      Frames
                    </span>
                    <input
                      aria-labelledby={framesLabel}
                      aria-describedby={framesHint}
                      className="cw-input cw-input--mono number-input"
                      type="number"
                      inputMode="numeric"
                      value={framesText}
                      min={grid?.min}
                      max={grid?.max ?? undefined}
                      step={grid?.step}
                      disabled={!limits}
                      aria-invalid={framesBlocker !== null}
                      onChange={(e) => {
                        setFramesUnreadable(e.target.validity.badInput);
                        edit(setFramesText)(e.target.value);
                      }}
                      onBlur={commitFrames}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') commitFrames();
                      }}
                    />
                    <span className="cw-field__hint" id={framesHint}>
                      {grid ? `${describeGrid(grid)}; empty = model default` : 'Empty = model default'}
                    </span>
                  </label>

                  <label className="cw-field">
                    <span className="cw-field__label">Frame rate</span>
                    <select className="cw-select cw-input--mono" value={fps} disabled={!limits} onChange={(e) => edit(setFps)(e.target.value)}>
                      <option value="">Model default</option>
                      {(limits?.fps ?? []).map((value) => (
                        <option key={value} value={String(value)}>
                          {value} fps
                        </option>
                      ))}
                    </select>
                  </label>
                </>
              )}
            </div>
            <p className="cw-field__hint">{limitsNote}</p>
          </div>

          <fieldset className="choice">
            <legend className="cw-field__label">Data class</legend>
            <div className="choice__options">
              {DATA_CLASS_ORDER.map((value) => (
                <label key={value} className="choice__option">
                  <input type="radio" name="data-class" value={value} checked={dataClass === value} onChange={() => edit(setDataClass)(value)} />
                  <DataClassTag value={value} />
                </label>
              ))}
            </div>
            <span className="cw-field__hint">{DATA_CLASSES[dataClass].hint}</span>
          </fieldset>

          <div className="form-row">
            <label className="cw-field">
              <span className="cw-field__label">Priority</span>
              <select className="cw-select" value={priority} onChange={(e) => edit(setPriority)(e.target.value as Priority)}>
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
                onChange={(e) => edit(setSeedText)(e.target.value)}
                aria-invalid={seed === undefined}
              />
              <span className="cw-field__hint">{seed === undefined ? 'Whole number, 0 or more' : 'Empty = random'}</span>
            </label>
          </div>

          {submitError ? <ErrorBanner title="The job was not submitted" error={submitError} /> : null}

          <div>
            <button type="submit" className="cw-btn cw-btn--primary" disabled={blocker !== null}>
              {blocker === null ? <Play strokeWidth={1.5} aria-hidden="true" /> : null}
              {blocker ?? 'Generate'}
            </button>
          </div>
        </form>

        <div className="stack">
          <EstimatePanel state={estimate} hasRequest={request !== null} />
          {tracked.job ? <JobPanel job={tracked.job} /> : null}
        </div>
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
