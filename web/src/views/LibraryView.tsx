// Library: asset grid with media previews and the provenance panel of the selected asset.
// `#/library/<sha256>` opens it with that asset selected, loaded by itself when it is not on the first page.

import { AudioLines, File } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';

import { api, assetContentUrl, isSha256, type Asset } from '../api/client';
import { useJobEvents } from '../api/events';
import { DataClassTag } from '../components/DataClassTag';
import { EmptyState, ErrorBanner, RetryButton } from '../components/Feedback';
import { Hash } from '../components/Hash';
import { ProvenancePanel } from '../components/ProvenancePanel';
import { formatBytes, formatTimestamp } from '../lib/format';
import { useHashSegments } from '../lib/route';
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

/** The asset a `#/library/<sha256>` link names: its id, `invalid` for anything but 64 lowercase hex characters, or null. */
function linkedAsset(segments: string[]): string | null {
  const value = segments[0] === 'library' ? segments[1] : undefined;
  if (!value) return null;
  return isSha256(value) ? value : 'invalid';
}

export function LibraryView() {
  const link = linkedAsset(useHashSegments());
  const linkedSha = link === 'invalid' ? null : link;
  const [assets, setAssets] = useState<Asset[]>([]);
  const [cursor, setCursor] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(linkedSha);
  const [loadingMore, setLoadingMore] = useState(false);
  const [moreError, setMoreError] = useState<unknown>();
  const [linkLoading, setLinkLoading] = useState(false);
  const [linkError, setLinkError] = useState<unknown>();
  const requestedLink = useRef<string | null>(null);
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

  // A link selects its asset; one that is not on the first page is loaded by itself (identical bytes keep their
  // first, older record, so a new job's output can sit further down the list).
  useEffect(() => {
    setLinkError(undefined);
    if (linkedSha) setSelected(linkedSha);
  }, [linkedSha]);

  useEffect(() => {
    if (!linkedSha || !list.data || requestedLink.current === linkedSha) return;
    if (assets.some((item) => item.sha256 === linkedSha)) return;
    requestedLink.current = linkedSha;
    setLinkLoading(true);
    api
      .asset(linkedSha)
      .then((item) => setAssets((current) => merge(current, [item])))
      .catch(setLinkError)
      .finally(() => setLinkLoading(false));
  }, [linkedSha, list.data]);

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
      {link === 'invalid' ? (
        <ErrorBanner title="Could not open the linked asset" error="The link names no asset; an asset id is 64 lowercase hex characters." />
      ) : null}
      {linkError ? <ErrorBanner title="Could not open the linked asset" error={linkError} /> : null}

      {!list.loading && !list.error && !linkLoading && assets.length === 0 ? (
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
            ) : linkLoading ? (
              <p className="cw-muted">Loading the linked asset</p>
            ) : (
              <p className="cw-muted">Select an asset to see its provenance.</p>
            )}
          </aside>
        </div>
      ) : null}
    </div>
  );
}
