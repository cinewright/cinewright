// Card for one model and quantization (design/components/ModelCard): calibration, install state with live download
// progress, and the install and remove actions. The licence is always visible.

import type { CalibrationResult, CalibrationStatus, ModelInstall, ModelSummary, Schemas } from '../api/client';
import { DASH, formatBytes, formatMb } from '../lib/format';
import { CapabilityChip } from './DataClassTag';
import { Light, PendingLight, StatusLight } from './StatusLight';

const RANK: Record<CalibrationStatus, number> = { runs: 0, slow: 1, wont_run: 2 };

/** The best measured result of a model/quant, or undefined when it was never calibrated. */
export function bestResult(results: CalibrationResult[] | undefined): CalibrationResult | undefined {
  return [...(results ?? [])].sort((a, b) => RANK[a.status] - RANK[b.status])[0];
}

function commercial(license: Schemas['License']): string {
  if (license.commercial_use === true) return 'commercial use';
  if (license.commercial_use === false) return 'non-commercial';
  return 'commercial use unclear';
}

export interface InstallControls {
  onInstall: () => void;
  onRemove: () => void;
  /** Why installing and removing are unavailable right now, as the disabled button's label; null when available. */
  blocker: string | null;
  /** The request this card started that has not answered yet. */
  pending: 'install' | 'remove' | null;
}

function DownloadProgress({ label, live }: { label: string; live: ModelInstall | undefined }) {
  const known = live?.progress != null && live.total_bytes != null;
  const percent = known ? Math.min(100, Math.max(0, live.progress?.percent ?? 0)) : 0;
  const text = known ? `${formatBytes(live.downloaded_bytes ?? 0)} of ${formatBytes(live.total_bytes)}` : 'Waiting for the download';
  return (
    <div>
      <div
        className="cw-progress"
        data-state={known ? undefined : 'waiting'}
        role="progressbar"
        aria-label={`Downloading ${label}`}
        aria-valuenow={Math.round(percent)}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <span style={{ width: `${percent}%` }} />
      </div>
      <div className="cw-progress__label">
        <span>{text}</span>
        <span>{known ? `${Math.round(percent)} %` : DASH}</span>
      </div>
    </div>
  );
}

function InstallState({ state, error, label, live }: { state: Schemas['InstallState']; error: string | null; label: string; live?: ModelInstall }) {
  switch (state) {
    case 'installed':
      return <Light state="ok" word="Installed" />;
    case 'downloading':
      return (
        <div className="stack stack--tight">
          <PendingLight word="Downloading" />
          <DownloadProgress label={label} live={live} />
        </div>
      );
    case 'failed':
      return (
        <div className="stack stack--tight">
          <Light state="fail" word="Install failed" />
          {error ? <p className="text-caption">{error}</p> : null}
        </div>
      );
    case 'absent':
      return <span className="cw-status cw-muted">Not installed</span>;
  }
}

function InstallActions({ state, label, controls }: { state: Schemas['InstallState']; label: string; controls: InstallControls }) {
  const { blocker, pending } = controls;
  const busy = blocker !== null || pending !== null;
  const remove = (
    <button type="button" className="cw-btn cw-btn--danger cw-btn--sm" onClick={controls.onRemove} disabled={busy} aria-label={busy ? undefined : `Remove ${label}`}>
      {pending === 'remove' ? 'Removing' : (blocker ?? 'Remove')}
    </button>
  );
  const install = (word: string) => (
    <button type="button" className="cw-btn cw-btn--secondary cw-btn--sm" onClick={controls.onInstall} disabled={busy} aria-label={busy ? undefined : `${word} ${label}`}>
      {pending === 'install' ? 'Starting' : (blocker ?? word)}
    </button>
  );
  switch (state) {
    case 'absent':
      return install('Install');
    case 'installed':
      return remove;
    case 'failed':
      // Remove also deletes the partial downloads a failed install keeps for the next try.
      return (
        <>
          {remove}
          {install('Try again')}
        </>
      );
    case 'downloading':
      return null;
  }
}

/** `live` is the latest `model` event for this quantization, newer than `quant`; `controls` adds the actions. */
export function ModelCard({
  model,
  quant,
  live,
  controls,
}: {
  model: ModelSummary;
  quant: Schemas['ModelQuantState'];
  live?: ModelInstall;
  controls?: InstallControls;
}) {
  const result = bestResult(quant.calibration);
  const license = model.license;
  const label = `${model.name} ${quant.quant}`;
  const state = live?.install_state ?? quant.install_state;
  const error = state === 'failed' ? (live?.error?.message ?? quant.error ?? null) : null;
  return (
    <article className="cw-model-card" aria-label={label}>
      <div className="cw-model-card__head">
        <div>
          <h3 className="cw-model-card__name">{model.name}</h3>
          <div className="cw-model-card__vendor">{model.vendor}</div>
        </div>
        <StatusLight status={result?.status} />
      </div>
      <div className="cw-model-card__caps">
        {model.capabilities.map((cap) => (
          <CapabilityChip key={cap} value={cap} />
        ))}
      </div>
      <dl className="cw-model-card__stats">
        <div>
          <dt>VRAM</dt>
          <dd>{formatMb(result?.peak_vram_mb)}</dd>
        </div>
        <div>
          <dt>Quant</dt>
          <dd>{quant.quant}</dd>
        </div>
        <div>
          <dt>s/step</dt>
          <dd>{result?.seconds_per_step != null ? result.seconds_per_step.toFixed(1) : DASH}</dd>
        </div>
      </dl>
      <InstallState state={state} error={error} label={label} live={state === 'downloading' ? live : undefined} />
      <div className="cw-model-card__foot">
        <span className="text-caption cw-muted">
          {license.url ? (
            <a href={license.url} target="_blank" rel="noreferrer">
              {license.name}
            </a>
          ) : (
            license.name
          )}
          {` · ${commercial(license)}`}
          {quant.size_bytes != null ? ` · ${formatBytes(quant.size_bytes)}` : ''}
        </span>
        {controls ? (
          <div className="model-card__actions">
            <InstallActions state={state} label={label} controls={controls} />
          </div>
        ) : null}
      </div>
    </article>
  );
}
