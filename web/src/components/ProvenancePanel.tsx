// An asset's provenance record with sidecar JSON export (design/components/ProvenancePanel).
// The design's fields come first in the specification's order; the rows after Parents are additions of this app.
// Missing values stay "–" so records remain comparable. A parent asset that was deleted keeps its hash, marked as deleted.

import { Download } from 'lucide-react';
import { Fragment, type ReactNode } from 'react';

import { assetProvenanceUrl, type Provenance } from '../api/client';
import { DASH, formatTimestamp, fullHash, shortHash, truncateMiddle } from '../lib/format';

const NONE: ReadonlySet<string> = new Set();

interface Row {
  label: string;
  value: string;
  title?: string;
  hash?: boolean;
}

/** Resolution, frames, fps, steps and guidance the job requested (registry defaults filled in); null for inputs. */
function parameters(params: Provenance['params']): string {
  if (!params) return DASH;
  const parts = [
    params.resolution,
    params.frames != null ? `${params.frames} frames` : null,
    params.fps != null ? `${params.fps} fps` : null,
    params.steps != null ? `${params.steps} steps` : null,
    params.guidance != null ? `guidance ${params.guidance}` : null,
  ];
  return parts.filter(Boolean).join(' · ') || DASH;
}

function rows(p: Provenance, deletedParents: ReadonlySet<string>): Row[] {
  const model = p.model;
  const modelHashes = model?.file_sha256 ?? [];
  const parents = p.parents ?? [];
  const loras = p.loras ?? [];
  const first = modelHashes[0];
  const parent = (hash: string, format: (hex: string) => string) => format(hash) + (deletedParents.has(hash) ? ' (deleted)' : '');
  return [
    {
      label: 'Model',
      value: model ? [model.name, model.quant, [model.engine, model.engine_version].filter(Boolean).join(' ')].join(' · ') : DASH,
    },
    {
      label: 'Model hash',
      value: first ? shortHash(first) + (modelHashes.length > 1 ? ` +${modelHashes.length - 1}` : '') : DASH,
      title: modelHashes.map(fullHash).join('\n') || undefined,
      hash: Boolean(first),
    },
    { label: 'Recipe', value: p.recipe ? `${p.recipe.id}@${p.recipe.version}` : DASH },
    { label: 'Seed', value: p.seed != null ? String(p.seed) : DASH },
    { label: 'LoRAs', value: loras.map((l) => `${l.name} (${l.weight})`).join(' · ') || DASH },
    { label: 'Host', value: p.host ? [p.host.name, p.host.gpu].filter(Boolean).join(' · ') : DASH },
    { label: 'Timestamp', value: formatTimestamp(p.created_at) },
    {
      label: 'Parents',
      value: parents.map((hash) => parent(hash, shortHash)).join(' · ') || DASH,
      title: parents.map((hash) => parent(hash, fullHash)).join('\n') || undefined,
      hash: parents.length > 0,
    },
    { label: 'Parameters', value: parameters(p.params) },
    { label: 'Prompt', value: p.prompt ?? DASH, title: p.prompt ?? undefined },
    { label: 'Job', value: p.job_id ? truncateMiddle(p.job_id) : DASH, title: p.job_id ?? undefined },
    { label: 'Client', value: p.client ? [p.client.kind, p.client.agent].filter(Boolean).join(' · ') : DASH },
  ];
}

export function ProvenancePanel({
  provenance,
  deletedParents = NONE,
  children,
}: {
  provenance: Provenance;
  /** Parent assets known to be deleted; their hashes stay in the Parents row, marked as deleted. */
  deletedParents?: ReadonlySet<string>;
  children?: ReactNode;
}) {
  return (
    <section className="cw-prov" aria-label="Provenance">
      <div className="cw-prov__head">
        <h3 className="cw-prov__title">Provenance</h3>
        <a className="cw-btn cw-btn--ghost cw-btn--sm" href={assetProvenanceUrl(provenance.asset.sha256)} download>
          <Download strokeWidth={1.5} aria-hidden="true" />
          Download sidecar JSON
        </a>
      </div>
      {children}
      <dl className="cw-prov__body">
        {rows(provenance, deletedParents).map((row) => (
          <Fragment key={row.label}>
            <dt>{row.label}</dt>
            <dd className={row.hash ? 'is-hash' : undefined} title={row.title ?? row.value}>
              {row.value}
            </dd>
          </Fragment>
        ))}
      </dl>
    </section>
  );
}
