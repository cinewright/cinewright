// Hosts: the local host (and later remote ones) with state, data class policy and hardware.

import { api } from '../api/client';
import { DataClassTag } from '../components/DataClassTag';
import { ErrorBanner, RetryButton } from '../components/Feedback';
import { Light } from '../components/StatusLight';
import { formatTimestamp } from '../lib/format';
import { useApi } from '../lib/useApi';
import { HardwareList } from './SetupView';

const KIND_LABEL = {
  local: 'Local',
  remote_own: 'Own remote host',
  remote_rented: 'Rented host',
  cloud: 'Cloud',
} as const;

export function HostsView() {
  const hosts = useApi((signal) => api.hosts(signal));
  const items = hosts.data?.items ?? [];

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
            {host.hardware ? <HardwareList hardware={host.hardware} /> : <p className="cw-muted">No hardware profile yet.</p>}
          </section>
        ))}
      </div>
    </div>
  );
}
