// Library: asset grid with media previews and the provenance panel of the selected asset.
// The hash is the selection: `#/library/<sha256>` opens it with that asset selected, loaded by itself when it is not on
// the first page, and selecting a card replaces the hash, so links and clicks take one path.
// Delete asks first, as the design requires for deletions; a deleted asset leaves the list and the selection, and keyboard
// focus moves to the card next to it.

import { AudioLines, File, Trash2 } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { api, ApiError, assetContentUrl, isSha256, type Asset } from '../api/client';
import { useJobEvents } from '../api/events';
import { ConfirmDialog } from '../components/ConfirmDialog';
import { DataClassTag } from '../components/DataClassTag';
import { EmptyState, ErrorBanner, RetryButton } from '../components/Feedback';
import { Hash } from '../components/Hash';
import { ProvenancePanel } from '../components/ProvenancePanel';
import { formatBytes, formatTimestamp } from '../lib/format';
import { replaceHash, useHashSegments } from '../lib/route';
import { useApi } from '../lib/useApi';

const REFRESH_DELAY_MS = 1000;

function merge(current: Asset[], incoming: Asset[]): Asset[] {
  const map = new Map(current.map((a) => [a.sha256, a]));
  for (const asset of incoming) map.set(asset.sha256, asset);
  return [...map.values()].sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at));
}

/** What the asset is, in the words of its delete action. */
function kind(asset: Asset): string {
  if (asset.media_type.startsWith('image/')) return 'image';
  if (asset.media_type.startsWith('video/')) return 'clip';
  if (asset.media_type.startsWith('audio/')) return 'audio file';
  return 'asset';
}

function Media({ asset, controls }: { asset: Asset; controls: boolean }) {
  const url = assetContentUrl(asset.sha256);
  const label = `${asset.media_type} ${asset.sha256.slice(0, 8)}`;
  const player = useRef<HTMLMediaElement | null>(null);
  const keep = useCallback((element: HTMLMediaElement | null) => {
    if (element) player.current = element;
  }, []);
  // A removed player may keep downloading, and the core cannot delete a file on Windows while it still streams it.
  // Dropping the source stops the download. Mount one Media per asset (a key), so the source dropped is this asset's;
  // a player still in the page (StrictMode's trial unmount) keeps it.
  useEffect(() => {
    const element = player.current;
    return () => {
      if (!element || element.isConnected) return;
      element.removeAttribute('src');
      element.load();
    };
  }, []);
  if (asset.media_type.startsWith('image/')) return <img src={url} alt={label} loading="lazy" />;
  if (asset.media_type.startsWith('video/'))
    return <video ref={keep} src={url} controls={controls} muted playsInline preload="metadata" aria-label={label} />;
  if (asset.media_type.startsWith('audio/'))
    return controls ? (
      <audio ref={keep} src={url} controls preload="metadata" aria-label={label} />
    ) : (
      <AudioLines strokeWidth={1.5} aria-hidden="true" />
    );
  return <File strokeWidth={1.5} aria-hidden="true" />;
}

/** The asset a `#/library/<sha256>` link names: its id, `invalid` for anything but 64 lowercase hex characters, or null. */
function linkedAsset(segments: string[]): string | null {
  const value = segments[0] === 'library' ? segments[1] : undefined;
  if (!value) return null;
  return isSha256(value) ? value : 'invalid';
}

/** Select an asset, or none, by replacing the hash; the hash change selects it. */
function select(sha256: string | null): void {
  replaceHash(sha256 ? `#/library/${sha256}` : '#/library');
}

export function LibraryView() {
  const link = linkedAsset(useHashSegments());
  const selected = link === 'invalid' ? null : link;
  const [assets, setAssets] = useState<Asset[]>([]);
  const [cursor, setCursor] = useState<string | null>(null);
  const [loadingMore, setLoadingMore] = useState(false);
  const [moreError, setMoreError] = useState<unknown>();
  // The asset being loaded by itself, and the failed load; each counts only while that asset is selected, the failed
  // load only while that asset is also missing, as a list page may bring it in after all.
  const [linkLoading, setLinkLoading] = useState<string | null>(null);
  const [linkError, setLinkError] = useState<{ sha256: string; error: unknown }>();
  const requestedLink = useRef<string | null>(null);
  const refreshTimer = useRef<number>();
  const [confirm, setConfirm] = useState<Asset | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);
  /** A refused deletion, shown with the asset it concerns while that asset stays selected. */
  const [deleteError, setDeleteError] = useState<{ sha256: string; title: string; error: unknown }>();
  /** Assets known to be deleted: deleted here, or a parent the core no longer has (404). */
  const [deleted, setDeleted] = useState<ReadonlySet<string>>(() => new Set());
  /** What the last deletion did, for screen readers. */
  const [announcement, setAnnouncement] = useState('');
  // Each deletion takes the next tick and each request for assets notes the current one, so a response to a request sent
  // before a deletion, which may still list the asset, is told apart from one sent after it, which lists it only if it
  // is back (its bytes uploaded again, or a job that had uploaded them succeeding).
  const tick = useRef(0);
  const deletedAt = useRef(new Map<string, number>());
  /** Where keyboard focus goes once the deletion has ended: the deleted card's place in the grid, or the Delete button. */
  const refocus = useRef<number | 'delete' | null>(null);
  const grid = useRef<HTMLDivElement>(null);
  const heading = useRef<HTMLHeadingElement>(null);
  const deleteButton = useRef<HTMLButtonElement>(null);

  /** The assets of a response to a request sent at tick `sent`, without those deleted after it went out. */
  const fresh = (items: Asset[], sent: number) => items.filter((item) => (deletedAt.current.get(item.sha256) ?? 0) <= sent);

  const list = useApi(async (signal) => {
    const sent = tick.current;
    const page = await api.assets(undefined, signal);
    const items = fresh(page.items, sent);
    setAssets((current) => merge(current, items));
    setCursor((current) => current ?? page.next_cursor ?? null);
    return page;
  });

  // A job that changes state may have produced outputs; pick up new assets shortly after.
  useJobEvents((event) => {
    if (event.kind !== 'state') return;
    window.clearTimeout(refreshTimer.current);
    refreshTimer.current = window.setTimeout(list.reload, REFRESH_DELAY_MS);
  });

  // A selected asset that is not on the first page is loaded by itself (identical bytes keep their first, older
  // record, so a new job's output can sit further down the list). Once per selection: a new list page does not load
  // it again, but selecting it anew, for example by following the same link after another card, does.
  useEffect(() => {
    requestedLink.current = null;
  }, [selected]);

  // A refused deletion concerns the asset that was selected; another selection drops its message.
  useEffect(() => setDeleteError(undefined), [selected]);

  useEffect(() => {
    if (!selected || !list.data || requestedLink.current === selected) return;
    if (assets.some((item) => item.sha256 === selected)) return;
    const sha256 = selected;
    requestedLink.current = sha256;
    setLinkLoading(sha256);
    setLinkError(undefined);
    const sent = tick.current;
    api
      .asset(sha256)
      .then((item) => {
        const items = fresh([item], sent);
        setAssets((current) => merge(current, items));
      })
      .catch((error: unknown) => setLinkError({ sha256, error }))
      .finally(() => setLinkLoading((current) => (current === sha256 ? null : current)));
  }, [selected, list.data]);

  const loadMore = () => {
    if (!cursor) return;
    setLoadingMore(true);
    setMoreError(undefined);
    const sent = tick.current;
    api
      .assets(cursor)
      .then((page) => {
        const items = fresh(page.items, sent);
        setAssets((current) => merge(current, items));
        setCursor(page.next_cursor ?? null);
      })
      .catch(setMoreError)
      .finally(() => setLoadingMore(false));
  };

  const asset = assets.find((a) => a.sha256 === selected);
  // An asset in the list is not deleted (any more), whatever was known of it before.
  const deletedParents = useMemo(() => new Set([...deleted].filter((hash) => !assets.some((item) => item.sha256 === hash))), [deleted, assets]);

  // The provenance of an asset keeps the hashes of its parents after they were deleted. A parent that is not in the list
  // is looked up once per selection; only a 404 marks it as deleted, not a failed request.
  useEffect(() => {
    const unknown = (asset?.provenance.parents ?? []).filter(
      (hash) => isSha256(hash) && !deleted.has(hash) && !assets.some((item) => item.sha256 === hash),
    );
    if (unknown.length === 0) return;
    const controller = new AbortController();
    for (const hash of unknown) {
      api.asset(hash, controller.signal).catch((error: unknown) => {
        if (error instanceof ApiError && error.status === 404) setDeleted((current) => new Set(current).add(hash));
      });
    }
    return () => controller.abort();
  }, [asset?.sha256]);

  /** The asset at `index` in the grid is gone from the core. */
  const gone = (target: Asset, index: number) => {
    deletedAt.current.set(target.sha256, ++tick.current);
    setAssets((current) => current.filter((item) => item.sha256 !== target.sha256));
    setDeleted((current) => new Set(current).add(target.sha256));
    setAnnouncement(`The ${kind(target)} was deleted.`);
    refocus.current = index;
    // The address no longer names it; a selection made while the deletion ran stays.
    if (window.location.hash === `#/library/${target.sha256}`) select(null);
  };

  // Deleting unmounts the asset's players (`deleting`) as the request goes out, which stops their downloads; the core
  // retries for about a second meanwhile, as Windows refuses to delete a file a download still holds open.
  const remove = (target: Asset) => {
    const index = assets.findIndex((item) => item.sha256 === target.sha256);
    setConfirm(null);
    setDeleting(target.sha256);
    setDeleteError(undefined);
    setAnnouncement(''); // so the next announcement changes the live region, even when its text is the same
    api
      .deleteAsset(target.sha256)
      .then(() => gone(target, index))
      .catch((error: unknown) => {
        // Already deleted, for example in another tab: what was asked for is done.
        if (error instanceof ApiError && error.status === 404) return gone(target, index);
        setDeleteError({ sha256: target.sha256, title: `Could not delete the ${kind(target)}`, error });
        refocus.current = 'delete';
      })
      .finally(() => setDeleting(null));
  };

  // The focused Delete button is disabled while the deletion runs and unmounted when it succeeds, which leaves focus on
  // the page itself. Once the deletion has ended, focus goes to the card that took the deleted one's place (else the
  // heading), or back to Delete after a refusal; a user who has moved focus meanwhile keeps it where it is.
  useEffect(() => {
    const target = refocus.current;
    if (target === null || deleting !== null) return;
    refocus.current = null;
    const focused = document.activeElement;
    if (focused !== null && focused !== document.body) return;
    if (target === 'delete') {
      deleteButton.current?.focus();
      return;
    }
    const cards = grid.current?.querySelectorAll<HTMLButtonElement>('.asset') ?? [];
    (cards[Math.min(target, cards.length - 1)] ?? heading.current)?.focus();
  });

  return (
    <div className="view">
      <header className="view__head">
        <h1 className="view__title" ref={heading} tabIndex={-1}>
          Library
        </h1>
      </header>

      <p className="sr-only" role="status">
        {announcement}
      </p>
      {list.error ? <ErrorBanner title="Could not load the library" error={list.error} action={<RetryButton onClick={list.reload} />} /> : null}
      {moreError ? <ErrorBanner title="Could not load more assets" error={moreError} /> : null}
      {link === 'invalid' ? (
        <ErrorBanner title="Could not open the linked asset" error="The link names no asset; an asset id is 64 lowercase hex characters." />
      ) : null}
      {linkError && linkError.sha256 === selected && !asset ? <ErrorBanner title="Could not open the linked asset" error={linkError.error} /> : null}

      {!list.loading && !list.error && linkLoading === null && assets.length === 0 && cursor === null ? (
        <EmptyState title="No assets yet">
          <p>Finished jobs store their outputs here, each with its provenance record.</p>
        </EmptyState>
      ) : null}

      {assets.length > 0 || cursor !== null ? (
        <div className="library">
          <div className="stack">
            <div className="asset-grid" ref={grid}>
              {assets.map((item) => (
                <button
                  key={item.sha256}
                  type="button"
                  className="asset"
                  aria-pressed={item.sha256 === selected}
                  onClick={() => select(item.sha256 === selected ? null : item.sha256)}
                >
                  <span className="asset__frame">{item.sha256 === deleting ? null : <Media asset={item} controls={false} />}</span>
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
                  {asset.sha256 === deleting ? null : <Media key={asset.sha256} asset={asset} controls />}
                </div>
                {deleteError?.sha256 === asset.sha256 ? <ErrorBanner title={deleteError.title} error={deleteError.error} /> : null}
                <div className="library__actions">
                  <button
                    ref={deleteButton}
                    type="button"
                    className="cw-btn cw-btn--danger cw-btn--sm"
                    onClick={() => setConfirm(asset)}
                    disabled={deleting !== null}
                  >
                    <Trash2 strokeWidth={1.5} aria-hidden="true" />
                    {deleting === null ? 'Delete' : deleting === asset.sha256 ? 'Deleting' : 'Another deletion running'}
                  </button>
                </div>
                <ProvenancePanel provenance={asset.provenance} deletedParents={deletedParents}>
                  <div className="prov__asset">
                    <Hash value={asset.sha256} copy />
                    <DataClassTag value={asset.data_class} />
                  </div>
                </ProvenancePanel>
              </div>
            ) : linkLoading !== null && linkLoading === selected ? (
              <p className="cw-muted">Loading the linked asset</p>
            ) : (
              <p className="cw-muted">Select an asset to see its provenance.</p>
            )}
          </aside>
        </div>
      ) : null}

      {confirm ? (
        <ConfirmDialog
          title={`Delete this ${kind(confirm)}?`}
          confirmLabel={`Delete ${kind(confirm)}`}
          onConfirm={() => remove(confirm)}
          onCancel={() => setConfirm(null)}
        >
          <p>Its file ({formatBytes(confirm.size_bytes)}) is deleted from the disk and cannot be restored.</p>
          <p>
            Jobs that made or used it keep their history but no longer list it as an output. Assets made from it keep their provenance; the
            Library marks it there as deleted.
          </p>
        </ConfirmDialog>
      ) : null}
    </div>
  );
}
