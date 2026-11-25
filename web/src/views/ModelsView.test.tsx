import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';

import type { Host, ModelInstall, ModelStoreUsage, ModelSummary, Schemas } from '../api/client';
import { EventsProvider } from '../api/events';
import { ProblemAnswer, stubApi } from '../test/api';
import { stubEventSource } from '../test/events';
import { ModelsView } from './ModelsView';

const GiB = 1024 ** 3;
const LICENSE = { name: 'Apache-2.0', commercial_use: true, geo_exclusions: [], url: null };

// The two Wan 2.2 models share a text encoder and a VAE: the store's total counts those once, each model's size again.
function wan(id: string, name: string, capability: Schemas['Capability'], installedGiB: number): ModelSummary {
  return {
    id,
    name,
    vendor: 'Wan-AI',
    family: 'wan2.2',
    capabilities: [capability],
    license: LICENSE,
    installed_bytes: installedGiB * GiB,
    quantizations: [
      { quant: 'bf16', install_state: 'absent', size_bytes: 54 * GiB },
      { quant: 'int8', install_state: 'installed', size_bytes: installedGiB * GiB },
    ],
  };
}

function zImage(state: Schemas['InstallState'], error: string | null = null): ModelSummary {
  const installed = state === 'installed';
  return {
    id: 'z-image-turbo',
    name: 'Z-Image Turbo 6B',
    vendor: 'Tongyi-MAI',
    family: 'z-image',
    capabilities: ['t2i'],
    license: LICENSE,
    installed_bytes: installed ? 10.25 * GiB : 0,
    quantizations: [{ quant: 'int8', install_state: state, size_bytes: installed ? 10.25 * GiB : 6 * GiB, error }],
  };
}

const I2V = wan('wan-2.2-i2v-14b', 'Wan 2.2 Image-to-Video 14B', 'i2v', 30.5);
const T2V = wan('wan-2.2-t2v-14b', 'Wan 2.2 Text-to-Video 14B', 't2v', 28);
const STORE: ModelStoreUsage = { total_bytes: 51.5 * GiB, quota_bytes: 200 * GiB, free_bytes: 812.5 * GiB };

const LOCAL: Host = { id: 'local', name: 'This computer', kind: 'local', online: true, allowed_data_classes: ['public', 'private'] };

interface Server {
  models: ModelSummary[];
  store: ModelStoreUsage;
  variant: Host['text_encoder_variant'];
}

function renderView(server: Server, routes: Record<string, (body: unknown) => unknown> = {}) {
  const source = stubEventSource();
  const calls = stubApi({
    'GET /api/v1/models': () => ({ items: server.models, store: server.store }),
    'GET /api/v1/hosts': () => ({ items: [{ ...LOCAL, text_encoder_variant: server.variant }] }),
    ...routes,
  });
  render(
    <EventsProvider>
      <ModelsView />
    </EventsProvider>,
  );
  return { calls, source };
}

function modelEvent(install: ModelInstall) {
  return { kind: 'model', model: install, created_at: '2026-11-25T12:00:00Z' };
}

/** The value next to `term` in the model store's list. */
function storeValue(term: string): string | null {
  const store = screen.getByRole('region', { name: 'Model store' });
  return within(store).getByText(term, { selector: 'dt' }).nextElementSibling?.textContent ?? null;
}

/** Model, quantizations and size of each row of the installed-models table. */
function installedRows(): string[][] {
  const table = screen.getByRole('table', { name: 'Installed models' });
  return within(table)
    .getAllByRole('row')
    .slice(1)
    .map((row) => within(row).getAllByRole('cell').map((cell) => cell.textContent ?? ''));
}

function card(name: string): HTMLElement {
  return screen.getByRole('article', { name });
}

beforeAll(() => {
  // jsdom has no modal dialogs.
  HTMLDialogElement.prototype.showModal = function showModal(this: HTMLDialogElement) {
    this.setAttribute('open', '');
  };
});

afterEach(() => {
  vi.useRealTimers();
});

describe('ModelsView', () => {
  it('shows the installed size per model, the total, the quota and free disk', async () => {
    renderView({ models: [I2V, T2V, zImage('absent')], store: STORE, variant: 'int8' });

    await screen.findByRole('table', { name: 'Installed models' });
    expect(installedRows()).toStrictEqual([
      ['Wan 2.2 Image-to-Video 14B', 'int8', '30.5 GB'],
      ['Wan 2.2 Text-to-Video 14B', 'int8', '28.0 GB'],
    ]);
    // Smaller than 30.5 GB + 28.0 GB: the shared text encoder and VAE count once.
    expect(storeValue('Installed')).toBe('51.5 GB');
    expect(storeValue('Quota')).toBe('200.0 GB');
    expect(storeValue('Free disk')).toBe('812.5 GB');
  });

  it('says so when no quota is set and nothing is installed', async () => {
    renderView({ models: [zImage('absent')], store: { total_bytes: 0, quota_bytes: null, free_bytes: 96 * GiB }, variant: null });

    await screen.findByText('No model installed yet.');
    expect(storeValue('Installed')).toBe('0 B');
    expect(storeValue('Quota')).toBe('No quota set');
    expect(storeValue('Free disk')).toBe('96.0 GB');
    expect(screen.queryByRole('table', { name: 'Installed models' })).toBeNull();
  });

  it('installs a quantization, follows its progress from model events and reloads the store when it ends', async () => {
    const server: Server = { models: [I2V, zImage('absent')], store: STORE, variant: 'int8' };
    const { calls, source } = renderView(server, {
      'POST /api/v1/models/z-image-turbo/install': () => {
        server.models = [I2V, zImage('downloading')];
        return null;
      },
    });
    const modelLoads = () => calls.filter((call) => call.method === 'GET' && call.path === '/api/v1/models').length;

    fireEvent.click(await screen.findByRole('button', { name: 'Install Z-Image Turbo 6B int8' }));
    const bar = await within(card('Z-Image Turbo 6B int8')).findByRole('progressbar', { name: 'Downloading Z-Image Turbo 6B int8' });
    expect(calls.filter((call) => call.method === 'POST')).toStrictEqual([
      { method: 'POST', path: '/api/v1/models/z-image-turbo/install', query: { quant: 'int8' }, body: undefined },
    ]);
    expect(bar.getAttribute('aria-valuenow')).toBe('0');
    expect(within(card('Z-Image Turbo 6B int8')).getByText('Waiting for the download')).toBeTruthy();

    const downloading = { model_id: 'z-image-turbo', quant: 'int8', install_state: 'downloading' } as const;
    act(() =>
      source().emit(
        'model',
        modelEvent({ ...downloading, progress: { phase: 'download', percent: 25 }, downloaded_bytes: 1.5 * GiB, total_bytes: 6 * GiB }),
      ),
    );
    expect(bar.getAttribute('aria-valuenow')).toBe('25');
    expect(within(card('Z-Image Turbo 6B int8')).getByText('1.5 GB of 6.0 GB')).toBeTruthy();
    const loadsWhileDownloading = modelLoads();

    server.models = [I2V, zImage('installed')];
    server.store = { ...STORE, total_bytes: 61.75 * GiB, free_bytes: 806.5 * GiB };
    act(() => source().emit('model', modelEvent({ ...downloading, install_state: 'installed' })));

    await waitFor(() => expect(storeValue('Installed')).toBe('61.8 GB'));
    expect(modelLoads()).toBe(loadsWhileDownloading + 1);
    expect(storeValue('Free disk')).toBe('806.5 GB');
    expect(installedRows()).toContainEqual(['Z-Image Turbo 6B', 'int8', '10.3 GB']);
    expect(within(card('Z-Image Turbo 6B int8')).getByText('Installed')).toBeTruthy();
    expect(within(card('Z-Image Turbo 6B int8')).queryByRole('progressbar')).toBeNull();
    expect(screen.getByRole('button', { name: 'Remove Z-Image Turbo 6B int8' })).toBeTruthy();
  });

  it('shows a refused install with the reason from the core', async () => {
    const shortfall =
      'Installing Z-Image Turbo 6B int8 needs 2.5 GB more free space on the drive of the model store: it downloads 6.0 GB, and only 3.5 GB are free.';
    renderView({ models: [zImage('absent')], store: STORE, variant: 'int8' }, {
      'POST /api/v1/models/z-image-turbo/install': () => new ProblemAnswer(409, 'Conflict', shortfall),
    });

    fireEvent.click(await screen.findByRole('button', { name: 'Install Z-Image Turbo 6B int8' }));
    const alert = await screen.findByRole('alert');
    expect(alert.textContent).toContain('Could not install Z-Image Turbo 6B int8');
    expect(alert.textContent).toContain(shortfall);
    expect(screen.getByRole('button', { name: 'Install Z-Image Turbo 6B int8' })).toBeTruthy();
  });

  it('shows why an install failed and offers to try again or remove its downloads', async () => {
    const mismatch = 'umt5-xxl/models_t5_umt5-xxl-enc-quanto_int8.safetensors did not match its pinned SHA-256, so it was deleted.';
    renderView({ models: [zImage('failed', mismatch)], store: STORE, variant: 'int8' });

    const failed = await screen.findByRole('article', { name: 'Z-Image Turbo 6B int8' });
    await within(failed).findByText(mismatch);
    expect(within(failed).getByText('Install failed')).toBeTruthy();
    expect(within(failed).getByRole('button', { name: 'Try again Z-Image Turbo 6B int8' })).toBeTruthy();
    expect(within(failed).getByRole('button', { name: 'Remove Z-Image Turbo 6B int8' })).toBeTruthy();
  });

  it('removes a quantization only after confirmation', async () => {
    const server: Server = { models: [I2V, T2V], store: STORE, variant: 'int8' };
    const { calls } = renderView(server, {
      'DELETE /api/v1/models/wan-2.2-t2v-14b': () => {
        server.models = [I2V, { ...T2V, installed_bytes: 0, quantizations: T2V.quantizations.map((q) => ({ ...q, install_state: 'absent' as const })) }];
        server.store = { ...STORE, total_bytes: 30.5 * GiB };
        return null;
      },
    });
    const deletes = () => calls.filter((call) => call.method === 'DELETE');

    fireEvent.click(await screen.findByRole('button', { name: 'Remove Wan 2.2 Text-to-Video 14B int8' }));
    let dialog = screen.getByRole('dialog', { name: 'Remove Wan 2.2 Text-to-Video 14B int8?' });
    expect(dialog.textContent).toContain('frees up to 28.0 GB');
    fireEvent.click(within(dialog).getByRole('button', { name: 'Cancel' }));
    expect(screen.queryByRole('dialog')).toBeNull();
    expect(deletes()).toHaveLength(0);

    fireEvent.click(screen.getByRole('button', { name: 'Remove Wan 2.2 Text-to-Video 14B int8' }));
    dialog = screen.getByRole('dialog', { name: 'Remove Wan 2.2 Text-to-Video 14B int8?' });
    fireEvent.click(within(dialog).getByRole('button', { name: 'Remove model' }));

    await waitFor(() => expect(storeValue('Installed')).toBe('30.5 GB'));
    expect(deletes()).toStrictEqual([{ method: 'DELETE', path: '/api/v1/models/wan-2.2-t2v-14b', query: { quant: 'int8' }, body: undefined }]);
    expect(installedRows()).toStrictEqual([['Wan 2.2 Image-to-Video 14B', 'int8', '30.5 GB']]);
    expect(screen.getByRole('button', { name: 'Install Wan 2.2 Text-to-Video 14B int8' })).toBeTruthy();
  });

  it('switches the text encoder of the local host and blocks installs until the host reports the new variant', async () => {
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] });
    const server: Server = { models: [I2V, zImage('absent')], store: STORE, variant: null };
    const { calls } = renderView(server, { 'PATCH /api/v1/hosts/local': () => null });

    const panel = await screen.findByRole('region', { name: 'Text encoder' });
    expect(within(panel).getByText(/Not chosen yet, so int8/)).toBeTruthy();
    expect(within(panel).getByRole('radio', { name: /int8/ })).toHaveProperty('checked', true);
    expect(within(panel).getByRole('button', { name: 'Choose another variant' })).toHaveProperty('disabled', true);

    fireEvent.click(within(panel).getByRole('radio', { name: /bf16/ }));
    fireEvent.click(within(panel).getByRole('button', { name: 'Switch to bf16' }));

    await within(panel).findByRole('button', { name: 'Switching to bf16' });
    expect(calls.filter((call) => call.method === 'PATCH')).toStrictEqual([
      { method: 'PATCH', path: '/api/v1/hosts/local', query: {}, body: { text_encoder_variant: 'bf16' } },
    ]);
    // The core refuses installs and removals during a switch.
    const install = within(card('Z-Image Turbo 6B int8')).getByRole('button', { name: 'Switch in progress' });
    expect(install).toHaveProperty('disabled', true);

    server.variant = 'bf16';
    act(() => {
      vi.advanceTimersByTime(5000);
    });
    await within(panel).findByRole('button', { name: 'Choose another variant' });
    expect(within(panel).getByRole('radio', { name: /bf16/ })).toHaveProperty('checked', true);
    expect(within(panel).queryByText(/Not chosen yet/)).toBeNull();
    expect(screen.getByRole('button', { name: 'Install Z-Image Turbo 6B int8' })).toHaveProperty('disabled', false);
  });

  it('does not offer a switch while a model is being installed', async () => {
    renderView({ models: [I2V, zImage('downloading')], store: STORE, variant: 'int8' });

    const panel = await screen.findByRole('region', { name: 'Text encoder' });
    await within(card('Z-Image Turbo 6B int8')).findByRole('progressbar');
    fireEvent.click(within(panel).getByRole('radio', { name: /bf16/ }));
    expect(within(panel).getByRole('button', { name: 'Wait for the install to finish' })).toHaveProperty('disabled', true);
  });
});
