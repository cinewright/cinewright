// An asset's provenance record with sidecar JSON export (design/components/ProvenancePanel).
// The design's fields come first in the specification's order; the rows after Parents are additions of this app.
// Missing values stay "–" so records remain comparable. Shortened hashes and ids have a copy button (CopyValue).

import { Download } from 'lucide-react';
import { Fragment, type ReactElement, type ReactNode } from 'react';

import { assetProvenanceUrl, type Provenance } from '../api/client';
import { DASH, formatTimestamp, fullHash, shortHash, truncateMiddle } from '../lib/format';
import { CopyValue } from './CopyValue';

interface Row {
  label: string;
  /** Text with itself (or `title`) as tooltip, or shortened values that bring their own tooltip and copy button. */
  value: string | ReactElement;
  title?: string;
  hash?: boolean;
}

/** All model files' hashes behind one button, one per line, so the files after the first are copied too. */
function ModelHashes({ hashes }: { hashes: string[] }) {
  const first = hashes[0] as string;
  const value = hashes.map(fullHash).join('\n');
  if (hashes.length === 1) return <CopyValue text={shortHash(first)} value={value} label="Copy model hash" />;
  const label = `Copy all ${hashes.length} model hashes`;
  return <CopyValue text={`${shortHash(first)} +${hashes.length - 1}`} value={value} label={label} tooltip={`${label}, one per line`} />;
}

function Parents({ parents }: { parents: string[] }) {
  return (
    <span className="copy-values">
      {parents.map((parent, index) => (
        <CopyValue
          key={index} // the same asset may be a parent twice
          text={shortHash(parent)}
          value={fullHash(parent)}
          label={parents.length > 1 ? `Copy parent hash ${index + 1} of ${parents.length}` : 'Copy parent hash'}
        />
      ))}
    </span>
  );
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

function rows(p: Provenance): Row[] {
  const model = p.model;
  const modelHashes = model?.file_sha256 ?? [];
  const parents = p.parents ?? [];
  const loras = p.loras ?? [];
  return [
    {
      label: 'Model',
      value: model ? [model.name, model.quant, [model.engine, model.engine_version].filter(Boolean).join(' ')].join(' · ') : DASH,
    },
    {
      label: 'Model hash',
      value: modelHashes.length > 0 ? <ModelHashes hashes={modelHashes} /> : DASH,
      hash: modelHashes.length > 0,
    },
    { label: 'Recipe', value: p.recipe ? `${p.recipe.id}@${p.recipe.version}` : DASH },
    { label: 'Seed', value: p.seed != null ? String(p.seed) : DASH },
    { label: 'LoRAs', value: loras.map((l) => `${l.name} (${l.weight})`).join(' · ') || DASH },
    { label: 'Host', value: p.host ? [p.host.name, p.host.gpu].filter(Boolean).join(' · ') : DASH },
    { label: 'Timestamp', value: formatTimestamp(p.created_at) },
    {
      label: 'Parents',
      value: parents.length > 0 ? <Parents parents={parents} /> : DASH,
      hash: parents.length > 0,
    },
    { label: 'Parameters', value: parameters(p.params) },
    { label: 'Prompt', value: p.prompt ?? DASH, title: p.prompt ?? undefined },
    { label: 'Job', value: p.job_id ? <CopyValue text={truncateMiddle(p.job_id)} value={p.job_id} label="Copy job id" /> : DASH },
    { label: 'Client', value: p.client ? [p.client.kind, p.client.agent].filter(Boolean).join(' · ') : DASH },
  ];
}

export function ProvenancePanel({ provenance, children }: { provenance: Provenance; children?: ReactNode }) {
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
        {rows(provenance).map((row) => {
          const text = typeof row.value === 'string' ? row.value : null;
          const classes = [row.hash ? 'is-hash' : null, text === null ? 'has-copy' : null].filter(Boolean).join(' ');
          return (
            <Fragment key={row.label}>
              <dt>{row.label}</dt>
              <dd className={classes || undefined} title={text === null ? undefined : (row.title ?? text)}>
                {row.value}
              </dd>
            </Fragment>
          );
        })}
      </dl>
    </section>
  );
}
