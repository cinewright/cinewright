// Library: asset grid with media previews and the provenance panel of the selected asset.

import { AudioLines, File } from 'lucide-react';
import { useRef, useState } from 'react';

import { api, assetContentUrl, type Asset } from '../api/client';
import { useJobEvents } from '../api/events';
import { DataClassTag } from '../components/DataClassTag';
import { EmptyState, ErrorBanner, RetryButton } from '../components/Feedback';
import { Hash } from '../components/Hash';
import { ProvenancePanel } from '../components/ProvenancePanel';
import { formatBytes, formatTimestamp } from '../lib/format';
import { useApi } from '../lib/useApi';

const REFRESH_DELAY_MS = 1000;

function merge(current: Asset[], incoming: Asset[]): Asset[] {
  const map = new Map(current.map((a) => [a.sha256, a]));
  for (const asset of incoming) map.set(asset.sha256, asset);
  return [...map.values()].sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at));
}

function Media({ asset, controls }: { asset: Asset; controls: boolean }) {
  const url = assetContentUrl(asset.sha256);
  const label = `${asset.media_type} ${asset.sha256.slice(0, 8)}`;
  if (asset.media_type.startsWith('image/')) return <img src={url} alt={label} loading="lazy" />;
  if (asset.media_type.startsWith('video/'))
    return <video src={url} controls={controls} muted playsInline preload="metadata" aria-label={label} />;
  if (asset.media_type.startsWith('audio/'))
    return controls ? <audio src={url} controls preload="metadata" aria-label={label} /> : <AudioLines strokeWidth={1.5} aria-hidden="true" />;
  return <File strokeWidth={1.5} aria-hidden="true" />;
}

export function LibraryView() {
  const [assets, setAssets] = useState<Asset[]>([]);
  const [cursor, setCursor] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [loadingMore, setLoadingMore] = useState(false);
  const [moreError, setMoreError] = useState<unknown>();
  const refreshTimer = useRef<number>();

  const list = useApi(async (signal) => {
    const page = await api.assets(undefined, signal);
    setAssets((current) => merge(current, page.items));
    setCursor((current) => current ?? page.next_cursor ?? null);
    return page;
  });

  // A job that changes state may have produced outputs; pick up new assets shortly after.
  useJobEvents((event) => {
    if (event.kind !== 'state') return;
    window.clearTimeout(refreshTimer.current);
    refreshTimer.current = window.setTimeout(list.reload, REFRESH_DELAY_MS);
  });

  const loadMore = () => {
    if (!cursor) return;
    setLoadingMore(true);
    setMoreError(undefined);
    api
      .assets(cursor)
      .then((page) => {
        setAssets((current) => merge(current, page.items));
        setCursor(page.next_cursor ?? null);
      })
      .catch(setMoreError)
      .finally(() => setLoadingMore(false));
  };

  const asset = assets.find((a) => a.sha256 === selected);

  return (
    <div className="view">
      <header className="view__head">
        <h1 className="view__title">Library</h1>
      </header>

      {list.error ? <ErrorBanner title="Could not load the library" error={list.error} action={<RetryButton onClick={list.reload} />} /> : null}
      {moreError ? <ErrorBanner title="Could not load more assets" error={moreError} /> : null}

      {!list.loading && !list.error && assets.length === 0 ? (
        <EmptyState title="No assets yet">
          <p>Finished jobs store their outputs here, each with its provenance record.</p>
        </EmptyState>
      ) : null}

      {assets.length > 0 ? (
        <div className="library">
          <div className="stack">
            <div className="asset-grid">
              {assets.map((item) => (
                <button
                  key={item.sha256}
                  type="button"
                  className="asset"
                  aria-pressed={item.sha256 === selected}
                  onClick={() => setSelected(item.sha256 === selected ? null : item.sha256)}
                >
                  <span className="asset__frame">
                    <Media asset={item} controls={false} />
                  </span>
                  <span className="asset__meta">
                    <span className="text-value">
                      {item.media_type} · {formatBytes(item.size_bytes)}
                    </span>
                    <DataClassTag value={item.data_class} />
                  </span>
                  <span className="text-caption cw-muted cw-mono">{formatTimestamp(item.created_at)}</span>
                </button>
              ))}
            </div>
            {cursor ? (
              <div>
                <button type="button" className="cw-btn cw-btn--secondary" onClick={loadMore} disabled={loadingMore}>
                  {loadingMore ? 'Loading' : 'Load more assets'}
                </button>
              </div>
            ) : null}
          </div>

          <aside className="library__detail" aria-label="Selected asset">
            {asset ? (
              <div className="stack">
                <div className="asset__frame asset__frame--large">
                  <Media asset={asset} controls />
                </div>
                <ProvenancePanel provenance={asset.provenance}>
                  <div className="prov__asset">
                    <Hash value={asset.sha256} copy />
                    <DataClassTag value={asset.data_class} />
                  </div>
                </ProvenancePanel>
              </div>
            ) : (
              <p className="cw-muted">Select an asset to see its provenance.</p>
            )}
          </aside>
        </div>
      ) : null}
    </div>
  );
}
