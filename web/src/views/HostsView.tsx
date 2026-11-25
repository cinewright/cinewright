// Hosts: the local host (and later remote ones) with state, worker health, data class policy and hardware.

import { useEffect } from 'react';

import { api, type WorkerHealth } from '../api/client';
import { DataClassTag } from '../components/DataClassTag';
import { ErrorBanner, RetryButton } from '../components/Feedback';
import { Light, PendingLight } from '../components/StatusLight';
import { DASH, formatMb, formatTimestamp } from '../lib/format';
import { useApi } from '../lib/useApi';
import { HardwareList } from './SetupView';

const KIND_LABEL = {
  local: 'Local',
  remote_own: 'Own remote host',
  remote_rented: 'Rented host',
  cloud: 'Cloud',
} as const;

// Worker health changes while the core starts or restarts the worker; the view follows it.
const REFRESH_MS = 5000;

function WorkerState({ health }: { health: WorkerHealth | null | undefined }) {
  if (!health) return <span className="cw-status cw-muted">No worker</span>;
  switch (health.state) {
    case 'starting':
      return <PendingLight word="Starting" />;
    case 'ready':
      return <Light state="ok" word="Ready" />;
    case 'busy':
      return <Light state="ok" word="Busy" />;
    case 'failed':
      return <Light state="fail" word="Failed" />;
  }
}

function vramInUse(health: WorkerHealth | null | undefined): string {
  if (health?.vram_used_mb == null) return DASH;
  const total = health.vram_total_mb != null ? ` of ${formatMb(health.vram_total_mb)}` : '';
  return `${formatMb(health.vram_used_mb)}${total}`;
}

function WorkerList({ health }: { health: WorkerHealth | null | undefined }) {
  return (
    <dl className="kv" aria-label="Worker">
      <dt>Worker</dt>
      <dd>
        <WorkerState health={health} />
      </dd>
      <dt>VRAM in use</dt>
      <dd>{vramInUse(health)}</dd>
      {health?.detail ? (
        <>
          <dt>Detail</dt>
          <dd className="kv__wrap">{health.detail}</dd>
        </>
      ) : null}
    </dl>
  );
}

export function HostsView() {
  const hosts = useApi((signal) => api.hosts(signal));
  const items = hosts.data?.items ?? [];
  const { reload } = hosts;

  useEffect(() => {
    const timer = window.setInterval(reload, REFRESH_MS);
    return () => window.clearInterval(timer);
  }, [reload]);

  return (
    <div className="view">
      <header className="view__head">
        <h1 className="view__title">Hosts</h1>
      </header>

      {hosts.error ? <ErrorBanner title="Could not load hosts" error={hosts.error} action={<RetryButton onClick={hosts.reload} />} /> : null}
      {!hosts.loading && !hosts.error && items.length === 0 ? <p className="cw-muted">No hosts registered.</p> : null}

      <div className="stack">
        {items.map((host) => (
          <section key={host.id} className="panel stack" aria-label={host.name}>
            <div className="panel__head">
              <div>
                <h2 className="panel__title">{host.name}</h2>
                <div className="text-caption cw-muted">
                  {KIND_LABEL[host.kind]} · <span className="cw-mono">{host.id}</span>
                </div>
              </div>
              {host.online ? (
                <Light state="ok" word="Online" />
              ) : (
                <Light state="fail" word="Offline" detail={host.last_seen_at ? `last contact ${formatTimestamp(host.last_seen_at)}` : null} />
              )}
            </div>
            <div className="tags" aria-label="Allowed data classes">
              {host.allowed_data_classes.map((value) => (
                <DataClassTag key={value} value={value} />
              ))}
            </div>
            <WorkerList health={host.worker_health} />
            {host.hardware ? <HardwareList hardware={host.hardware} /> : <p className="cw-muted">No hardware profile yet.</p>}
          </section>
        ))}
      </div>
    </div>
  );
}
