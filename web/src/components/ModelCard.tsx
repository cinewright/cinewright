// Card for one model and quantization (design/components/ModelCard). The licence is always visible.

import type { CalibrationResult, CalibrationStatus, ModelSummary, Schemas } from '../api/client';
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

function InstallAction({ quant }: { quant: Schemas['ModelQuantState'] }) {
  switch (quant.install_state) {
    case 'installed':
      return <Light state="ok" word="Installed" />;
    case 'downloading':
      return <PendingLight word="Downloading" />;
    case 'failed':
      return <Light state="fail" word="Install failed" />;
    case 'absent':
      return (
        <button type="button" className="cw-btn cw-btn--secondary cw-btn--sm" disabled>
          Installing arrives in phase 1
        </button>
      );
  }
}

export function ModelCard({ model, quant }: { model: ModelSummary; quant: Schemas['ModelQuantState'] }) {
  const result = bestResult(quant.calibration);
  const license = model.license;
  return (
    <article className="cw-model-card" aria-label={`${model.name} ${quant.quant}`}>
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
        <InstallAction quant={quant} />
      </div>
    </article>
  );
}
