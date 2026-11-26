// About: the mark, version from /system, the MIT licence and the WanGP disclosure.

import { X } from 'lucide-react';
import { useEffect, useRef } from 'react';

import { api } from '../api/client';
import { CINEWRIGHT_LICENSE, WANGP_LICENSE_URL, WANGP_NOTICE } from '../lib/notices';
import { DASH, truncateMiddle } from '../lib/format';
import { useApi } from '../lib/useApi';
import { CopyValue } from './CopyValue';
import { ErrorBanner } from './Feedback';

const ENGINE_STATE = {
  not_installed: 'Not installed',
  provisioning: 'Provisioning',
  ready: 'Ready',
  failed: 'Failed',
} as const;

export function AboutDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const ref = useRef<HTMLDialogElement>(null);
  const system = useApi((signal) => (open ? api.system(signal) : Promise.resolve(undefined)), [open]);

  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    if (open && !dialog.open) dialog.showModal();
    if (!open && dialog.open) dialog.close();
  }, [open]);

  const info = system.data;
  return (
    <dialog ref={ref} className="dialog" aria-labelledby="about-title" onClose={onClose}>
      <div className="dialog__head">
        <img src="/logos/cinewright-mark.svg" alt="" width={32} height={32} />
        <h2 id="about-title" className="dialog__title">
          About Cinewright
        </h2>
        <button type="button" className="cw-btn cw-btn--ghost cw-btn--sm" onClick={onClose} aria-label="Close">
          <X strokeWidth={1.5} aria-hidden="true" />
        </button>
      </div>
      <div className="dialog__body stack">
        {system.error ? <ErrorBanner title="Could not load system information" error={system.error} /> : null}
        <dl className="kv">
          <dt>Version</dt>
          <dd>{info?.version ?? DASH}</dd>
          <dt>API version</dt>
          <dd>{info?.api_version ?? DASH}</dd>
          <dt>Listens on</dt>
          <dd>{info?.bind ?? DASH}</dd>
          <dt>Data directory</dt>
          <dd className="kv__wrap">{info?.data_dir ?? DASH}</dd>
          {(info?.engines ?? []).map((engine) => (
            <EngineRow key={engine.id} name={engine.name} version={engine.version ?? null} state={ENGINE_STATE[engine.state]} />
          ))}
        </dl>
        <p>{CINEWRIGHT_LICENSE}</p>
        <p>
          {WANGP_NOTICE}{' '}
          <a href={WANGP_LICENSE_URL} target="_blank" rel="noreferrer">
            WanGP Community License 2.0
          </a>
        </p>
      </div>
    </dialog>
  );
}

function EngineRow({ name, version, state }: { name: string; version: string | null; state: string }) {
  return (
    <>
      <dt>{name}</dt>
      <dd className={version ? 'has-copy' : undefined}>
        {version ? (
          <>
            <CopyValue text={truncateMiddle(version)} value={version} label={`Copy ${name} version`} />
            {' · '}
          </>
        ) : null}
        {state}
      </dd>
    </>
  );
}
