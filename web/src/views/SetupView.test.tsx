import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import type { EngineInfo } from '../api/client';
import { EventsProvider } from '../api/events';
import { stubApi } from '../test/api';
import { stubClipboard } from '../test/clipboard';
import { EnginePanel } from './SetupView';

const COMMIT = '38758d3c73366422efcf609abb63cab8385f3552';
const ENGINE: EngineInfo = { id: 'wan2gp', name: 'WanGP (Wan2GP)', version: null, state: 'not_installed', notice: null };
const UV_MISSING =
  "No uv.exe lies in Cinewright's installation folder, so WanGP cannot be set up here (only the installed app brings it); " +
  'reinstall Cinewright if this is the installed app.';

/** An EventSource whose named events the test emits by hand. */
class FakeEventSource {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSED = 2;
  static instances: FakeEventSource[] = [];

  readonly url: string;
  readyState = FakeEventSource.CONNECTING;
  onopen: ((event: Event) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  private readonly listeners = new Map<string, ((event: MessageEvent) => void)[]>();

  constructor(url: string) {
    this.url = url;
    FakeEventSource.instances.push(this);
  }

  addEventListener(kind: string, listener: (event: MessageEvent) => void): void {
    this.listeners.set(kind, [...(this.listeners.get(kind) ?? []), listener]);
  }

  close(): void {
    this.readyState = FakeEventSource.CLOSED;
  }

  emit(kind: string, data: unknown): void {
    const message = new MessageEvent(kind, { data: JSON.stringify(data) });
    for (const listener of this.listeners.get(kind) ?? []) listener(message);
  }
}

function engineEvent(engine: EngineInfo) {
  return { kind: 'engine', engine, created_at: '2026-11-24T12:00:00Z' };
}

/** `initial` is the engine GET /system reports; a function answers with its current result on every request. */
function renderPanel(initial: EngineInfo | (() => EngineInfo), provisioned: EngineInfo) {
  FakeEventSource.instances = [];
  vi.stubGlobal('EventSource', FakeEventSource);
  const engine = typeof initial === 'function' ? initial : () => initial;
  const calls = stubApi({
    'GET /api/v1/system': () => ({ name: 'Cinewright', version: '0.0.1', api_version: '1', data_dir: 'C:/data', bind: '127.0.0.1:8737', engines: [engine()] }),
    'POST /api/v1/engines/wan2gp/provision': () => provisioned,
  });
  render(
    <EventsProvider>
      <EnginePanel />
    </EventsProvider>,
  );
  return { calls, source: FakeEventSource.instances[0] as FakeEventSource };
}

describe('EnginePanel', () => {
  it('follows provisioning progress from engine events until the engine is ready', async () => {
    const { source } = renderPanel(ENGINE, ENGINE);
    await screen.findByText('Not installed');

    act(() =>
      source.emit(
        'engine',
        engineEvent({ ...ENGINE, state: 'provisioning', progress: { phase: 'Downloading WanGP', step: 1, total_steps: 6, percent: 4.2 } }),
      ),
    );
    const bar = screen.getByRole('progressbar', { name: 'Step 1 of 6 · Downloading WanGP' });
    expect(bar.getAttribute('aria-valuenow')).toBe('4');

    act(() => source.emit('engine', engineEvent({ ...ENGINE, state: 'ready', version: COMMIT })));
    expect(screen.getByText('Ready')).toBeTruthy();
    expect(screen.getByTitle(COMMIT).textContent).toBe(`${COMMIT.slice(0, 8)}…${COMMIT.slice(-8)}`);
    expect(screen.queryByRole('progressbar')).toBeNull();

    const writes = stubClipboard();
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Copy WanGP (Wan2GP) version' }));
    });
    expect(writes).toEqual([COMMIT]);
  });

  it('loads the engine state again when the event stream reopens, because a missed engine event is never replayed', async () => {
    const installing: EngineInfo = { ...ENGINE, state: 'provisioning', progress: { phase: 'Installing packages', step: 5, total_steps: 6, percent: 40 } };
    let reported = installing;
    const { calls, source } = renderPanel(() => reported, ENGINE);
    const systemCalls = () => calls.filter((call) => call.method === 'GET' && call.path === '/api/v1/system').length;
    await screen.findByRole('progressbar', { name: 'Step 5 of 6 · Installing packages' });

    act(() => {
      source.readyState = FakeEventSource.OPEN;
      source.onopen?.(new Event('open'));
    });
    await waitFor(() => expect(systemCalls()).toBe(2)); // an event may have come before the stream opened

    // The stream drops; the browser reconnects on its own. Meanwhile setup finishes, and its `ready` event is lost.
    act(() => {
      source.readyState = FakeEventSource.CONNECTING;
      source.onerror?.(new Event('error'));
    });
    reported = { ...ENGINE, state: 'ready', version: COMMIT };
    expect(screen.getByRole('progressbar')).toBeTruthy();

    act(() => {
      source.readyState = FakeEventSource.OPEN;
      source.onopen?.(new Event('open'));
    });
    await screen.findByText('Ready');
    expect(systemCalls()).toBe(3);
    expect(screen.getByTitle(COMMIT)).toBeTruthy();
    expect(screen.queryByRole('progressbar')).toBeNull();
  });

  it('shows a failure with its one action and retries through the API', async () => {
    const failed: EngineInfo = {
      ...ENGINE,
      state: 'failed',
      error: {
        code: 'hash_mismatch',
        message: 'The WanGP download from github.com did not match its pinned SHA-256, so it was deleted and the engine is not installed.',
        action: 'retry',
      },
    };
    const provisioning: EngineInfo = { ...ENGINE, state: 'provisioning', progress: { phase: 'Downloading WanGP', step: 1, total_steps: 6, percent: 0 } };
    const { calls } = renderPanel(failed, provisioning);

    const alert = await screen.findByRole('alert');
    expect(alert.textContent).toContain('WanGP (Wan2GP) setup failed');
    expect(alert.textContent).toContain('did not match its pinned SHA-256');
    fireEvent.click(screen.getByRole('button', { name: 'Try again' }));

    await screen.findByRole('progressbar', { name: 'Step 1 of 6 · Downloading WanGP' });
    expect(calls.filter((call) => call.method === 'POST').map((call) => call.path)).toStrictEqual(['/api/v1/engines/wan2gp/provision']);
    expect(screen.queryByRole('alert')).toBeNull();
  });

  it('offers no retry where retrying cannot help', async () => {
    renderPanel(
      { ...ENGINE, error: { code: 'uv_missing', message: UV_MISSING, action: null } },
      ENGINE,
    );
    await screen.findByText(UV_MISSING);
    expect(screen.queryByRole('button')).toBeNull();
  });
});
