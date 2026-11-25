// Setup and calibration: engine provisioning (state from /system, live progress from `engine` events),
// hardware profile with "Detect again", model status lights from /capabilities.

import { RefreshCw } from 'lucide-react';
import { Fragment, useEffect, useRef, useState } from 'react';

import { api, type CalibrationResult, type EngineInfo, type HardwareProfile } from '../api/client';
import { useEventsConnected, useSystemEvents } from '../api/events';
import { CapabilityChip } from '../components/DataClassTag';
import { ErrorBanner, RetryButton } from '../components/Feedback';
import { bestResult } from '../components/ModelCard';
import { Light, PendingLight, StatusLight } from '../components/StatusLight';
import { DASH, formatMb, formatSecondsPerStep, formatTimestamp, truncateMiddle } from '../lib/format';
import { useApi } from '../lib/useApi';

const ENGINE_ID = 'wan2gp';

function EngineStatus({ engine, onRetry, retrying }: { engine: EngineInfo; onRetry: () => void; retrying: boolean }) {
  const progress = engine.progress;
  if (engine.state === 'ready') {
    return (
      <dl className="kv">
        <dt>{engine.name}</dt>
        <dd>
          <Light state="ok" word="Ready" />
        </dd>
        <dt>Version</dt>
        <dd className="cw-mono" title={engine.version ?? undefined}>
          {engine.version ? truncateMiddle(engine.version) : DASH}
        </dd>
      </dl>
    );
  }
  if (engine.state === 'provisioning') {
    const percent = progress?.percent ?? 0;
    const label = progress
      ? [progress.step && progress.total_steps ? `Step ${progress.step} of ${progress.total_steps}` : null, progress.phase].filter(Boolean).join(' · ')
      : 'Starting';
    return (
      <div className="stack">
        <PendingLight word={`Setting up ${engine.name}`} />
        <div>
          <div className="cw-progress" role="progressbar" aria-label={label} aria-valuenow={Math.round(percent)} aria-valuemin={0} aria-valuemax={100}>
            <span style={{ width: `${percent}%` }} />
          </div>
          <div className="cw-progress__label">
            <span>{label}</span>
            <span>{Math.round(percent)} %</span>
          </div>
        </div>
        <p className="text-caption cw-muted">
          Cinewright downloads the engine, its Python environment and ffmpeg once and checks every file against its pinned hash. This
          takes a while; the other views stay usable.
        </p>
      </div>
    );
  }
  if (engine.state === 'failed') {
    const action =
      engine.error?.action === 'retry' ? (
        retrying ? (
          <button type="button" className="cw-btn cw-btn--secondary cw-btn--sm" disabled>
            Starting
          </button>
        ) : (
          <RetryButton onClick={onRetry} />
        )
      ) : undefined;
    return <ErrorBanner title={`${engine.name} setup failed`} error={engine.error?.message ?? 'The setup failed.'} action={action} />;
  }
  return (
    <div className="stack">
      <span className="cw-status cw-muted">Not installed</span>
      <p className="cw-muted">{engine.error?.message ?? `${engine.name} is set up in the background when Cinewright starts.`}</p>
    </div>
  );
}

/** The engine's provisioning: loaded from /system, then kept current by `engine` events (full snapshots) and loaded
 * again whenever the event stream opens. */
export function EnginePanel() {
  const system = useApi((signal) => api.system(signal));
  const [live, setLive] = useState<EngineInfo>();
  const [retrying, setRetrying] = useState(false);
  const [retryError, setRetryError] = useState<unknown>();
  const events = useRef(0);

  useSystemEvents((event) => {
    if (event.kind === 'engine' && event.engine?.id === ENGINE_ID) {
      events.current += 1;
      setLive(event.engine);
    }
  });

  // Engine events are never replayed, so one may have been missed before the stream (re)opened: load the state again.
  const connected = useEventsConnected();
  const wasConnected = useRef(connected);
  useEffect(() => {
    const opened = connected && !wasConnected.current;
    wasConnected.current = connected;
    if (!opened) return;
    const controller = new AbortController();
    const seen = events.current;
    api
      .system(controller.signal)
      .then((info) => {
        const current = info.engines.find((candidate) => candidate.id === ENGINE_ID);
        // An event that arrived meanwhile is newer than this answer.
        if (current && events.current === seen) setLive(current);
      })
      .catch(() => undefined); // the shown state stays; the next event corrects it
    return () => controller.abort();
  }, [connected]);

  const retry = () => {
    const seen = events.current;
    setRetrying(true);
    setRetryError(undefined);
    api
      .provisionEngine(ENGINE_ID)
      // An event that arrived meanwhile is newer than this answer.
      .then((engine) => events.current === seen && setLive(engine))
      .catch(setRetryError)
      .finally(() => setRetrying(false));
  };

  const engine = live ?? system.data?.engines.find((candidate) => candidate.id === ENGINE_ID);
  return (
    <section className="panel" aria-labelledby="engine-title">
      <h2 id="engine-title" className="panel__title">
        Engine
      </h2>
      {system.error && !engine ? (
        <ErrorBanner title="Could not load the engine state" error={system.error} action={<RetryButton onClick={system.reload} />} />
      ) : null}
      {!engine && system.loading ? <p className="cw-muted">Loading engine state</p> : null}
      {engine ? <EngineStatus engine={engine} onRetry={retry} retrying={retrying} /> : null}
      {retryError ? <ErrorBanner title="Could not start the setup again" error={retryError} /> : null}
    </section>
  );
}

export function HardwareList({ hardware }: { hardware: HardwareProfile }) {
  return (
    <dl className="kv">
      <dt>Host</dt>
      <dd>{hardware.host_id}</dd>
      <dt>Operating system</dt>
      <dd>{hardware.os}</dd>
      <dt>CPU</dt>
      <dd>{hardware.cpu ?? DASH}</dd>
      <dt>RAM</dt>
      <dd>{formatMb(hardware.ram_total_mb)}</dd>
      <dt>Free disk</dt>
      <dd>{formatMb(hardware.disk_free_mb)}</dd>
      {hardware.gpus.length === 0 ? (
        <>
          <dt>GPU</dt>
          <dd>No GPU found</dd>
        </>
      ) : (
        hardware.gpus.map((gpu) => (
          <Fragment key={gpu.index}>
            <dt>GPU {gpu.index}</dt>
            <dd>
              {[
                gpu.name,
                `${formatMb(gpu.vram_total_mb)} VRAM`,
                gpu.driver_version && `driver ${gpu.driver_version}`,
                gpu.compute_capability && `compute ${gpu.compute_capability}`,
                gpu.pcie_link,
              ]
                .filter(Boolean)
                .join(' · ')}
            </dd>
          </Fragment>
        ))
      )}
    </dl>
  );
}

function detailFor(results: CalibrationResult[], modelId: string, quant: string): string | null {
  const result = bestResult(results.filter((r) => r.model_id === modelId && r.quant === quant));
  if (!result) return null;
  return result.seconds_per_step != null ? formatSecondsPerStep(result.seconds_per_step) : (result.detail ?? null);
}

export function SetupView() {
  const hardware = useApi((signal) => api.hardware(signal));
  const capabilities = useApi((signal) => api.capabilities(signal));
  const calibration = useApi((signal) => api.calibration(signal));
  const [detecting, setDetecting] = useState(false);
  const [detectError, setDetectError] = useState<unknown>();

  const detect = () => {
    setDetecting(true);
    setDetectError(undefined);
    api
      .detectHardware()
      .then(hardware.setData)
      .catch(setDetectError)
      .finally(() => setDetecting(false));
  };

  const profile = hardware.data;
  const entries = capabilities.data?.items ?? [];
  const results = calibration.data?.items ?? [];

  return (
    <div className="view">
      <header className="view__head">
        <h1 className="view__title">Setup and calibration</h1>
        <button type="button" className="cw-btn cw-btn--primary" disabled>
          Calibration arrives in phase 1
        </button>
      </header>

      <EnginePanel />

      <section className="panel" aria-labelledby="hardware-title">
        <div className="panel__head">
          <h2 id="hardware-title" className="panel__title">
            Hardware
          </h2>
          <button type="button" className="cw-btn cw-btn--secondary cw-btn--sm" onClick={detect} disabled={detecting}>
            <RefreshCw strokeWidth={1.5} aria-hidden="true" />
            {detecting ? 'Detecting' : 'Detect again'}
          </button>
        </div>
        {detectError ? <ErrorBanner title="Hardware detection failed" error={detectError} /> : null}
        {hardware.error && !profile ? <ErrorBanner title="Could not load the hardware profile" error={hardware.error} /> : null}
        {!profile && hardware.loading ? <p className="cw-muted">Loading hardware profile</p> : null}
        {profile ? (
          <div className="stack">
            <HardwareList hardware={profile} />
            {profile.cuda_driver_ok === true ? <Light state="ok" word="Driver supports the GPU" /> : null}
            {profile.cuda_driver_ok === false ? <Light state="fail" word="Driver too old for the GPU" /> : null}
            {(profile.notes ?? []).length > 0 ? (
              <ul className="notes">
                {(profile.notes ?? []).map((note) => (
                  <li key={note}>{note}</li>
                ))}
              </ul>
            ) : null}
            <p className="text-caption cw-muted">
              Detected at <span className="cw-mono">{formatTimestamp(profile.detected_at)}</span>
            </p>
          </div>
        ) : null}
      </section>

      <section className="panel" aria-labelledby="models-title">
        <h2 id="models-title" className="panel__title">
          Model status
        </h2>
        {capabilities.error ? (
          <ErrorBanner title="Could not load capabilities" error={capabilities.error} action={<RetryButton onClick={capabilities.reload} />} />
        ) : null}
        {calibration.error ? <ErrorBanner title="Could not load calibration results" error={calibration.error} /> : null}
        {!capabilities.loading && !capabilities.error && entries.length === 0 ? (
          <p className="cw-muted">No models in the registry.</p>
        ) : null}
        {entries.length > 0 ? (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th scope="col">Model</th>
                  <th scope="col">Quant</th>
                  <th scope="col">Capability</th>
                  <th scope="col">Status</th>
                  <th scope="col">Hosts</th>
                </tr>
              </thead>
              <tbody>
                {entries.map((entry) => (
                  <tr key={`${entry.model_id}/${entry.quant}/${entry.capability}`}>
                    <td className="table__strong">{entry.model_name}</td>
                    <td className="cw-mono">{entry.quant}</td>
                    <td>
                      <CapabilityChip value={entry.capability} />
                    </td>
                    <td>
                      <StatusLight status={entry.status} detail={entry.status ? detailFor(results, entry.model_id, entry.quant) : null} />
                    </td>
                    <td className="cw-mono">{(entry.hosts ?? []).join(', ') || DASH}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </section>
    </div>
  );
}
