import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import type { Asset } from '../api/client';
import { EventsProvider } from '../api/events';
import { stubApi, type ApiCall, type Routes } from '../test/api';
import { stubEventSource } from '../test/events';
import { LibraryView } from './LibraryView';

const NEW_SHA = '1'.repeat(64);
const OLD_SHA = '2'.repeat(64);
const LINKED_SHA = '3'.repeat(64);

function asset(sha256: string, createdAt: string, params: Asset['provenance']['params'] = null): Asset {
  return {
    sha256,
    media_type: 'video/mp4',
    size_bytes: 4 * 1024 ** 2,
    data_class: 'private',
    job_id: 'job-1',
    created_at: createdAt,
    provenance: {
      schema: 'cinewright.provenance/1',
      asset: { sha256, media_type: 'video/mp4', size_bytes: 4 * 1024 ** 2 },
      model: { id: 'ltx-2.3-distilled-22b', name: 'LTX-2 2.3 Distilled 22B', quant: 'int8', engine: 'wan2gp', engine_version: '38758d3' },
      prompt: 'Harbour at night',
      seed: 2814033,
      params,
      created_at: createdAt,
      job_id: 'job-1',
    },
  };
}

// An identical output keeps its first, older record, so the job's asset is not on the first page.
const LINKED = asset(LINKED_SHA, '2026-11-01T08:00:00Z', {
  resolution: '1280x704',
  frames: 241,
  fps: 24,
  steps: 8,
});
const FIRST_PAGE = [asset(NEW_SHA, '2026-11-25T12:00:00Z'), asset(OLD_SHA, '2026-11-24T12:00:00Z')];

function renderAt(hash: string, routes: Routes = {}): ApiCall[] {
  window.location.hash = hash;
  stubEventSource();
  const calls = stubApi({
    'GET /api/v1/assets': () => ({ items: FIRST_PAGE, next_cursor: 'page-2' }),
    [`GET /api/v1/assets/${LINKED_SHA}`]: () => LINKED,
    ...routes,
  });
  render(
    <EventsProvider>
      <LibraryView />
    </EventsProvider>,
  );
  return calls;
}

/** The value of `term` in the provenance panel of the selected asset. */
function provenance(term: string): string | null {
  const detail = screen.getByRole('complementary', { name: 'Selected asset' });
  return within(detail).getByText(term, { selector: 'dt' }).nextElementSibling?.textContent ?? null;
}

/** The asset card that shows `timestamp` (the provenance panel may show it too). */
function card(timestamp: string): HTMLElement {
  const button = screen.getAllByRole('button').find((element) => element.classList.contains('asset') && element.textContent?.includes(timestamp));
  if (!button) throw new Error(`No asset card shows ${timestamp}`);
  return button;
}

/** The timestamps of the pressed asset cards. */
function pressedTimestamps(): string[] {
  return screen.queryAllByRole('button', { pressed: true }).map((button) => button.querySelector('.cw-mono')?.textContent ?? '');
}

afterEach(() => {
  window.location.hash = '';
});

describe('LibraryView', () => {
  it('opens a linked asset that is not on the first page, with its provenance parameters', async () => {
    const calls = renderAt(`#/library/${LINKED_SHA}`);

    await waitFor(() => expect(provenance('Parameters')).toBe('1280x704 · 241 frames · 24 fps · 8 steps'));
    expect(provenance('Seed')).toBe('2814033');
    expect(calls.filter((call) => call.path === `/api/v1/assets/${LINKED_SHA}`)).toHaveLength(1);
    // The list's first page came first; the asset was then loaded by itself.
    expect(calls.findIndex((call) => call.path === '/api/v1/assets')).toBeLessThan(calls.findIndex((call) => call.path === `/api/v1/assets/${LINKED_SHA}`));
    const pressed = screen.getAllByRole('button', { pressed: true });
    expect(pressed).toHaveLength(1);
    expect(within(pressed[0] as HTMLElement).getByText('2026-11-01T08:00:00Z')).toBeTruthy();
  });

  it('selects a linked asset from the first page without loading it again', async () => {
    const calls = renderAt(`#/library/${OLD_SHA}`);

    await waitFor(() => expect(screen.getAllByRole('button', { pressed: true })).toHaveLength(1));
    expect(within(screen.getByRole('button', { pressed: true })).getByText('2026-11-24T12:00:00Z')).toBeTruthy();
    // Earlier records have no parameters.
    expect(provenance('Parameters')).toBe('–');
    expect(calls.map((call) => call.path)).toStrictEqual(['/api/v1/assets']);
  });

  it('builds no request from a link that is not an asset id', async () => {
    const calls = renderAt(`#/library/${'A'.repeat(64)}`);

    expect((await screen.findByRole('alert')).textContent).toContain('The link names no asset; an asset id is 64 lowercase hex characters.');
    await screen.findByText('Select an asset to see its provenance.');
    expect(calls.map((call) => call.path)).toStrictEqual(['/api/v1/assets']);
  });

  it('says so when the linked asset does not exist', async () => {
    renderAt(`#/library/${'4'.repeat(64)}`);

    const alert = await screen.findByRole('alert');
    expect(within(alert).getByText('Could not open the linked asset')).toBeTruthy();
    expect(screen.queryAllByRole('button', { pressed: true })).toHaveLength(0);
  });

  it('keeps the hash in step with a card click, which clears the banner of a link that failed', async () => {
    const missing = '4'.repeat(64);
    const calls = renderAt(`#/library/${missing}`);
    await screen.findByRole('alert');

    fireEvent.click(card('2026-11-24T12:00:00Z'));
    await waitFor(() => expect(provenance('Seed')).toBe('2814033'));
    expect(window.location.hash).toBe(`#/library/${OLD_SHA}`);
    expect(screen.queryByRole('alert')).toBeNull();
    expect(pressedTimestamps()).toStrictEqual(['2026-11-24T12:00:00Z']);

    // Following the failed link again loads the asset again and says so again.
    window.location.hash = `#/library/${missing}`;
    expect(within(await screen.findByRole('alert')).getByText('Could not open the linked asset')).toBeTruthy();
    expect(calls.filter((call) => call.path === `/api/v1/assets/${missing}`)).toHaveLength(2);
    expect(screen.queryAllByRole('button', { pressed: true })).toHaveLength(0);
  });

  it('clears the banner of a link that is not an asset id when a card is selected', async () => {
    renderAt('#/library/not-an-asset');
    await screen.findByRole('alert');

    fireEvent.click(card('2026-11-25T12:00:00Z'));
    await waitFor(() => expect(screen.queryByRole('alert')).toBeNull());
    expect(window.location.hash).toBe(`#/library/${NEW_SHA}`);
    expect(pressedTimestamps()).toStrictEqual(['2026-11-25T12:00:00Z']);
  });

  it('selects the linked asset again when its link is followed after another card was selected', async () => {
    const calls = renderAt(`#/library/${LINKED_SHA}`);
    await waitFor(() => expect(provenance('Parameters')).toBe('1280x704 · 241 frames · 24 fps · 8 steps'));
    const historyLength = window.history.length;

    fireEvent.click(card('2026-11-25T12:00:00Z'));
    await waitFor(() => expect(pressedTimestamps()).toStrictEqual(['2026-11-25T12:00:00Z']));
    expect(window.location.hash).toBe(`#/library/${NEW_SHA}`);
    expect(provenance('Parameters')).toBe('–');
    expect(window.history.length).toBe(historyLength); // a selection replaces the entry, it adds none

    // "Open in Library" in the Generate view links to the job's output.
    window.location.hash = `#/library/${LINKED_SHA}`;
    await waitFor(() => expect(pressedTimestamps()).toStrictEqual(['2026-11-01T08:00:00Z']));
    expect(provenance('Parameters')).toBe('1280x704 · 241 frames · 24 fps · 8 steps');
    // It is in the list by now, so it is not loaded again.
    expect(calls.filter((call) => call.path === `/api/v1/assets/${LINKED_SHA}`)).toHaveLength(1);

    // Clicking the selected card selects none.
    fireEvent.click(card('2026-11-01T08:00:00Z'));
    await screen.findByText('Select an asset to see its provenance.');
    expect(window.location.hash).toBe('#/library');
    expect(screen.queryAllByRole('button', { pressed: true })).toHaveLength(0);
  });
});
