// Models: the model store (installed size per model, total, quota, free disk), the text encoder variant of the local
// host, and one card per model and quantization with install, live download progress and remove.
// Install states load from /models and follow `model` events, which are never replayed: the list loads again when an
// install ends or a model is removed, and whenever the event stream (re)opens.

import { useCallback, useEffect, useRef, useState } from 'react';

import { api, type Host, type ModelInstall, type ModelStoreUsage, type ModelSummary, type Schemas } from '../api/client';
import { useEventsConnected, useSystemEvents } from '../api/events';
import { ConfirmDialog } from '../components/ConfirmDialog';
import { ErrorBanner, RetryButton } from '../components/Feedback';
import { ModelCard } from '../components/ModelCard';
import { PendingLight } from '../components/StatusLight';
import { DASH, formatBytes } from '../lib/format';
import { useApi } from '../lib/useApi';

type Quant = Schemas['ModelQuantState'];
type Variant = NonNullable<Host['text_encoder_variant']>;

const VARIANTS: Variant[] = ['int8', 'bf16'];
/** Applies while a host has no variant chosen (WanGP's default). */
const DEFAULT_VARIANT: Variant = 'int8';
/** A text encoder switch sends no events; GET /hosts shows the new variant once it is done. */
const SWITCH_POLL_MS = 5000;
const ENDED: ReadonlySet<Schemas['InstallState']> = new Set(['installed', 'failed', 'absent']);

const keyOf = (modelId: string, quant: string) => `${modelId}/${quant}`;

interface LiveEntry {
  install: ModelInstall;
  /** Order of arrival, to tell events from before and after a reload apart. */
  seq: number;
}

/** GET /models with the latest `model` event per quantization laid over it. */
function useModelStates() {
  const models = useApi((signal) => api.models(signal));
  const [live, setLive] = useState<ReadonlyMap<string, LiveEntry>>(new Map());
  const seq = useRef(0);
  const { setData } = models;

  const refresh = useCallback(async () => {
    const seen = seq.current;
    try {
      const list = await api.models();
      setData(list);
      // Events that arrived meanwhile are newer than this answer.
      setLive((current) => new Map([...current].filter(([, entry]) => entry.seq > seen)));
    } catch {
      // The shown state stays; the next event or reload corrects it.
    }
  }, [setData]);

  useSystemEvents((event) => {
    const install = event.kind === 'model' ? event.model : null;
    if (!install) return;
    seq.current += 1;
    const entry = { install, seq: seq.current };
    setLive((current) => new Map(current).set(keyOf(install.model_id, install.quant), entry));
    // Sizes, the store's totals and free disk change only with the end of an install or a removal.
    if (ENDED.has(install.install_state)) void refresh();
  });

  // An event may have been missed before the stream (re)opened.
  const connected = useEventsConnected();
  const wasConnected = useRef(connected);
  useEffect(() => {
    const opened = connected && !wasConnected.current;
    wasConnected.current = connected;
    if (opened) void refresh();
  }, [connected, refresh]);

  return { models, live, refresh };
}

function StorePanel({ store, models }: { store: ModelStoreUsage | null | undefined; models: ModelSummary[] }) {
  const installed = models.filter((model) => model.installed_bytes > 0);
  return (
    <section className="panel stack" aria-labelledby="store-title">
      <h2 id="store-title" className="panel__title">
        Model store
      </h2>
      <dl className="kv" aria-label="Model store usage">
        <dt>Installed</dt>
        <dd>{store ? formatBytes(store.total_bytes) : DASH}</dd>
        <dt>Quota</dt>
        <dd>{store ? (store.quota_bytes != null ? formatBytes(store.quota_bytes) : 'No quota set') : DASH}</dd>
        <dt>Free disk</dt>
        <dd>{store ? formatBytes(store.free_bytes) : DASH}</dd>
      </dl>
      {installed.length === 0 ? (
        <p className="cw-muted">No model installed yet.</p>
      ) : (
        <div className="table-wrap">
          <table className="table" aria-label="Installed models">
            <thead>
              <tr>
                <th scope="col">Model</th>
                <th scope="col">Quant</th>
                <th scope="col" className="table__num">
                  Size
                </th>
              </tr>
            </thead>
            <tbody>
              {installed.map((model) => (
                <tr key={model.id}>
                  <td className="table__strong">{model.name}</td>
                  <td className="cw-mono">
                    {model.quantizations
                      .filter((quant) => quant.install_state === 'installed')
                      .map((quant) => quant.quant)
                      .join(', ') || DASH}
                  </td>
                  <td className="cw-mono table__num">{formatBytes(model.installed_bytes)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="text-caption cw-muted">
        A text encoder or VAE that several models share is stored once: the total counts it once, each model&apos;s size counts it
        again. The quota is set in Settings; to stay within it, an install first removes the models used least recently.
      </p>
    </section>
  );
}

function TextEncoderPanel({
  host,
  switchingTo,
  installing,
  onSwitch,
  starting,
}: {
  host: Host;
  switchingTo: Variant | null;
  installing: boolean;
  onSwitch: (variant: Variant) => void;
  starting: boolean;
}) {
  const current = host.text_encoder_variant ?? DEFAULT_VARIANT;
  const [choice, setChoice] = useState<Variant>(current);
  // Follow the host once a switch is done, or when another change arrives.
  useEffect(() => setChoice(current), [current]);

  let blocker: string | null = null;
  if (switchingTo) blocker = `Switching to ${switchingTo}`;
  else if (starting) blocker = 'Starting';
  else if (installing) blocker = 'Wait for the install to finish';
  else if (choice === current) blocker = 'Choose another variant';

  return (
    <section className="panel stack" aria-labelledby="encoder-title">
      <div>
        <h2 id="encoder-title" className="panel__title">
          Text encoder
        </h2>
        <div className="text-caption cw-muted">
          {host.name} · <span className="cw-mono">{host.id}</span>
        </div>
      </div>
      <fieldset className="choice" disabled={switchingTo !== null}>
        <legend className="cw-field__label">Variant</legend>
        <div className="choice__options">
          {VARIANTS.map((variant) => (
            <label key={variant} className="choice__option">
              <input type="radio" name="text-encoder-variant" value={variant} checked={choice === variant} onChange={() => setChoice(variant)} />
              <span className="cw-mono">{variant}</span>
              {variant === current ? <span className="text-caption cw-muted">in use</span> : null}
            </label>
          ))}
        </div>
      </fieldset>
      {switchingTo ? (
        <div className="stack stack--tight">
          <PendingLight word={`Switching to ${switchingTo}`} />
          <p className="text-caption cw-muted">
            Cinewright downloads the {switchingTo} files, restarts the worker and then removes the {current} files. The variant changes
            here once the switch is done.
          </p>
        </div>
      ) : null}
      <div className="form-actions">
        <button type="button" className="cw-btn cw-btn--secondary" disabled={blocker !== null} onClick={() => onSwitch(choice)}>
          {blocker ?? `Switch to ${choice}`}
        </button>
      </div>
      <p className="text-caption cw-muted">
        {host.text_encoder_variant == null ? `Not chosen yet, so ${DEFAULT_VARIANT}, WanGP's default, applies. ` : ''}
        One variant serves every model on this computer. int8 takes about half the disk space of bf16. A switch first checks the
        free disk space and is refused while a job runs.
      </p>
    </section>
  );
}

export function ModelsView() {
  const { models, live, refresh } = useModelStates();
  const hosts = useApi((signal) => api.hosts(signal));
  const [pending, setPending] = useState<{ key: string; action: 'install' | 'remove' } | null>(null);
  const [actionError, setActionError] = useState<{ title: string; error: unknown } | null>(null);
  const [confirm, setConfirm] = useState<{ model: ModelSummary; quant: Quant } | null>(null);
  const [switchingTo, setSwitchingTo] = useState<Variant | null>(null);
  const [switchStarting, setSwitchStarting] = useState(false);

  const items = models.data?.items ?? [];
  const local = hosts.data?.items.find((host) => host.kind === 'local');
  const localVariant = local?.text_encoder_variant ?? null;
  const stateOf = (model: ModelSummary, quant: Quant) => live.get(keyOf(model.id, quant.quant))?.install.install_state ?? quant.install_state;
  const installing = items.some((model) => model.quantizations.some((quant) => stateOf(model, quant) === 'downloading'));

  // The switch is done once GET /hosts names the new variant.
  const { reload: reloadHosts } = hosts;
  useEffect(() => {
    if (!switchingTo) return;
    if (localVariant === switchingTo) {
      setSwitchingTo(null);
      void refresh(); // installed sizes changed with the variant
      return;
    }
    const timer = window.setInterval(reloadHosts, SWITCH_POLL_MS);
    return () => window.clearInterval(timer);
  }, [switchingTo, localVariant, reloadHosts, refresh]);

  const install = (model: ModelSummary, quant: Quant) => {
    setPending({ key: keyOf(model.id, quant.quant), action: 'install' });
    setActionError(null);
    api
      .installModel(model.id, quant.quant)
      // The install is checked and marked `downloading` before the answer, so the list shows it at once.
      .then(refresh)
      .catch((error: unknown) => setActionError({ title: `Could not install ${model.name} ${quant.quant}`, error }))
      .finally(() => setPending(null));
  };

  const remove = (model: ModelSummary, quant: Quant) => {
    setConfirm(null);
    setPending({ key: keyOf(model.id, quant.quant), action: 'remove' });
    setActionError(null);
    api
      .removeModel(model.id, quant.quant)
      .then(refresh)
      .catch((error: unknown) => setActionError({ title: `Could not remove ${model.name} ${quant.quant}`, error }))
      .finally(() => setPending(null));
  };

  const switchVariant = (variant: Variant) => {
    if (!local) return;
    setSwitchStarting(true);
    setActionError(null);
    api
      .updateHost(local.id, { text_encoder_variant: variant })
      .then(() => {
        setSwitchingTo(variant);
        reloadHosts();
      })
      .catch((error: unknown) => setActionError({ title: `Could not switch the text encoder to ${variant}`, error }))
      .finally(() => setSwitchStarting(false));
  };

  return (
    <div className="view">
      <header className="view__head">
        <h1 className="view__title">Models</h1>
      </header>

      {actionError ? <ErrorBanner title={actionError.title} error={actionError.error} /> : null}
      {models.error && !models.data ? (
        <ErrorBanner title="Could not load models" error={models.error} action={<RetryButton onClick={models.reload} />} />
      ) : null}
      {hosts.error && !hosts.data ? (
        <ErrorBanner title="Could not load the hosts" error={hosts.error} action={<RetryButton onClick={hosts.reload} />} />
      ) : null}

      <div className="models-overview">
        {models.data ? <StorePanel store={models.data.store} models={items} /> : null}
        {local ? (
          <TextEncoderPanel host={local} switchingTo={switchingTo} installing={installing} onSwitch={switchVariant} starting={switchStarting} />
        ) : null}
      </div>

      {!models.loading && !models.error && items.length === 0 ? <p className="cw-muted">No models in the registry.</p> : null}

      <div className="card-grid">
        {items.flatMap((model) =>
          model.quantizations.map((quant) => {
            const key = keyOf(model.id, quant.quant);
            return (
              <ModelCard
                key={key}
                model={model}
                quant={quant}
                live={live.get(key)?.install}
                controls={{
                  onInstall: () => install(model, quant),
                  onRemove: () => setConfirm({ model, quant }),
                  blocker: switchingTo || switchStarting ? 'Switch in progress' : null,
                  pending: pending?.key === key ? pending.action : null,
                }}
              />
            );
          }),
        )}
      </div>

      {confirm ? (
        <ConfirmDialog
          title={`Remove ${confirm.model.name} ${confirm.quant.quant}?`}
          confirmLabel="Remove model"
          onConfirm={() => remove(confirm.model, confirm.quant)}
          onCancel={() => setConfirm(null)}
        >
          <p>
            Its files are deleted from the model store
            {confirm.quant.size_bytes != null && confirm.quant.install_state === 'installed'
              ? `, which frees up to ${formatBytes(confirm.quant.size_bytes)}`
              : ''}
            . Shared files stay while another installed model or a waiting job needs them. Using the model again means downloading it
            again.
          </p>
        </ConfirmDialog>
      ) : null}
    </div>
  );
}
