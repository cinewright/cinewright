// Setup and calibration: hardware profile with "Detect again", model status lights from /capabilities.

import { RefreshCw } from 'lucide-react';
import { Fragment, useState } from 'react';

import { api, type CalibrationResult, type HardwareProfile } from '../api/client';
import { CapabilityChip } from '../components/DataClassTag';
import { ErrorBanner, RetryButton } from '../components/Feedback';
import { bestResult } from '../components/ModelCard';
import { Light, StatusLight } from '../components/StatusLight';
import { DASH, formatMb, formatSecondsPerStep, formatTimestamp } from '../lib/format';
import { useApi } from '../lib/useApi';

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
