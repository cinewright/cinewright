import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi, type MockInstance } from 'vitest';

import type { Asset } from '../api/client';
import { EventsProvider } from '../api/events';
import { shortHash } from '../lib/format';
import { callSummary, ProblemAnswer, StatusAnswer, stubApi, type ApiCall, type Routes } from '../test/api';
import { stubEventSource } from '../test/events';
import { LibraryView } from './LibraryView';

const NEW_SHA = '1'.repeat(64);
const OLD_SHA = '2'.repeat(64);
const LINKED_SHA = '3'.repeat(64);
const DERIVED_SHA = '5'.repeat(64);
const GONE_SHA = '6'.repeat(64);

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

/** An image made from `parents`. */
function derived(parents: string[]): Asset {
  const base = asset(DERIVED_SHA, '2026-11-26T09:00:00Z');
  const image = { sha256: DERIVED_SHA, media_type: 'image/png', size_bytes: 2048 };
  return { ...base, ...image, provenance: { ...base.provenance, asset: image, parents } };
}

const deletes = (calls: ApiCall[]) => calls.filter((call) => call.method === 'DELETE').map(callSummary);

let load: MockInstance<() => void>;

beforeAll(() => {
  // jsdom has no modal dialogs.
  HTMLDialogElement.prototype.showModal = function showModal(this: HTMLDialogElement) {
    this.setAttribute('open', '');
  };
  HTMLDialogElement.prototype.close = function close(this: HTMLDialogElement) {
    this.removeAttribute('open');
    this.dispatchEvent(new Event('close'));
  };
  // jsdom loads no media; a player that is removed calls load() to stop its download, also at each test's cleanup.
  load = vi.spyOn(HTMLMediaElement.prototype, 'load').mockImplementation(() => undefined);
});

beforeEach(() => {
  load.mockClear();
});

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

  it('drops the banner of a link that failed once its asset arrives with the next page', async () => {
    let pages = 0;
    const calls = renderAt(`#/library/${LINKED_SHA}`, {
      'GET /api/v1/assets': () => (pages++ === 0 ? { items: FIRST_PAGE, next_cursor: 'page-2' } : { items: [LINKED], next_cursor: null }),
      [`GET /api/v1/assets/${LINKED_SHA}`]: () => new StatusAnswer(503),
    });
    expect(within(await screen.findByRole('alert')).getByText('Could not open the linked asset')).toBeTruthy();

    fireEvent.click(screen.getByRole('button', { name: 'Load more assets' }));
    await waitFor(() => expect(provenance('Parameters')).toBe('1280x704 · 241 frames · 24 fps · 8 steps'));
    expect(screen.queryByRole('alert')).toBeNull();
    expect(pressedTimestamps()).toStrictEqual(['2026-11-01T08:00:00Z']);

    // Selecting it anew neither loads it again nor brings the banner back.
    fireEvent.click(card('2026-11-01T08:00:00Z'));
    await screen.findByText('Select an asset to see its provenance.');
    fireEvent.click(card('2026-11-01T08:00:00Z'));
    await waitFor(() => expect(provenance('Parameters')).toBe('1280x704 · 241 frames · 24 fps · 8 steps'));
    expect(screen.queryByRole('alert')).toBeNull();
    expect(calls.filter((call) => call.path === `/api/v1/assets/${LINKED_SHA}`)).toHaveLength(1);
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

  it('deletes the selected asset only after confirmation, then clears the selection and the address', async () => {
    const calls = renderAt(`#/library/${OLD_SHA}`, { [`DELETE /api/v1/assets/${OLD_SHA}`]: () => new StatusAnswer(204) });
    await waitFor(() => expect(pressedTimestamps()).toStrictEqual(['2026-11-24T12:00:00Z']));

    fireEvent.click(screen.getByRole('button', { name: 'Delete' }));
    let dialog = screen.getByRole('dialog', { name: 'Delete this clip?' });
    expect(dialog.textContent).toContain('Its file (4.0 MB) is deleted from the disk and cannot be restored.');
    expect(dialog.textContent).toContain('Jobs that made or used it keep their history but no longer list it as an output.');
    expect(dialog.textContent).toContain('Assets made from it keep their provenance; the Library marks it there as deleted.');
    fireEvent.click(within(dialog).getByRole('button', { name: 'Cancel' }));
    expect(screen.queryByRole('dialog')).toBeNull();
    expect(deletes(calls)).toHaveLength(0);
    expect(pressedTimestamps()).toStrictEqual(['2026-11-24T12:00:00Z']);
    expect(window.location.hash).toBe(`#/library/${OLD_SHA}`);
    expect(load).not.toHaveBeenCalled();

    // The browser returns focus to the button that opened the dialog; jsdom's stand-in dialog does not, so it starts there.
    screen.getByRole('button', { name: 'Delete' }).focus();
    fireEvent.click(screen.getByRole('button', { name: 'Delete' }));
    dialog = screen.getByRole('dialog', { name: 'Delete this clip?' });
    fireEvent.click(within(dialog).getByRole('button', { name: 'Delete clip' }));

    await screen.findByText('Select an asset to see its provenance.');
    // Focus moves from the Delete button, which is gone, to the card next to the deleted one.
    await waitFor(() => expect(document.activeElement).toBe(card('2026-11-25T12:00:00Z')));
    expect(deletes(calls)).toStrictEqual([{ method: 'DELETE', path: `/api/v1/assets/${OLD_SHA}`, query: {}, body: undefined }]);
    expect(() => card('2026-11-24T12:00:00Z')).toThrow();
    expect(() => card('2026-11-25T12:00:00Z')).not.toThrow();
    expect(screen.queryAllByRole('button', { pressed: true })).toHaveLength(0);
    expect(window.location.hash).toBe('#/library');
    expect(screen.getByRole('status').textContent).toBe('The clip was deleted.');
    expect(screen.queryByRole('alert')).toBeNull();
    // Both players of the clip, the card's and the panel's, dropped their source, which stops their downloads.
    expect(load).toHaveBeenCalledTimes(2);
    expect(load.mock.contexts.every((element) => element instanceof HTMLVideoElement && !element.hasAttribute('src'))).toBe(true);
  });

  it('shows why an asset in use was not deleted next to Delete, keeps it, and returns focus to Delete', async () => {
    const detail = `This asset is an input of job ${'a'.repeat(32)}, which is still queued, so it was not deleted. Delete it once that job has ended.`;
    let answer: ((value: unknown) => void) | undefined;
    const calls = renderAt(`#/library/${NEW_SHA}`, { [`DELETE /api/v1/assets/${NEW_SHA}`]: () => new Promise((resolve) => (answer = resolve)) });
    await waitFor(() => expect(pressedTimestamps()).toStrictEqual(['2026-11-25T12:00:00Z']));

    fireEvent.click(screen.getByRole('button', { name: 'Delete' }));
    fireEvent.click(within(screen.getByRole('dialog', { name: 'Delete this clip?' })).getByRole('button', { name: 'Delete clip' }));
    expect((screen.getByRole('button', { name: 'Deleting' }) as HTMLButtonElement).disabled).toBe(true);
    // Focus is on the page itself, as a browser takes it off a button that becomes disabled (jsdom keeps it there).
    expect(document.activeElement).toBe(document.body);
    act(() => answer?.(new ProblemAnswer(409, 'Conflict', detail)));

    const alert = await screen.findByRole('alert');
    expect(within(alert).getByText('Could not delete the clip')).toBeTruthy();
    expect(within(alert).getByText(detail)).toBeTruthy();
    const detailPane = screen.getByRole('complementary', { name: 'Selected asset' });
    expect(detailPane.contains(alert)).toBe(true);
    expect(deletes(calls)).toHaveLength(1);
    expect(pressedTimestamps()).toStrictEqual(['2026-11-25T12:00:00Z']);
    expect(window.location.hash).toBe(`#/library/${NEW_SHA}`);
    // The players are back and Delete can be tried again; it has focus again.
    await waitFor(() => expect(detailPane.querySelector('video')?.getAttribute('src')).toBe(`/api/v1/assets/${NEW_SHA}/content`));
    expect((screen.getByRole('button', { name: 'Delete' }) as HTMLButtonElement).disabled).toBe(false);
    await waitFor(() => expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Delete' })));

    // The message concerns that asset: another selection drops it, and it does not come back with the asset.
    fireEvent.click(card('2026-11-24T12:00:00Z'));
    await waitFor(() => expect(pressedTimestamps()).toStrictEqual(['2026-11-24T12:00:00Z']));
    expect(screen.queryByRole('alert')).toBeNull();
    fireEvent.click(card('2026-11-25T12:00:00Z'));
    await waitFor(() => expect(pressedTimestamps()).toStrictEqual(['2026-11-25T12:00:00Z']));
    expect(screen.queryByRole('alert')).toBeNull();
  });

  it('takes a delete that finds the asset already gone as done', async () => {
    const calls = renderAt(`#/library/${OLD_SHA}`, {
      [`DELETE /api/v1/assets/${OLD_SHA}`]: () => new ProblemAnswer(404, 'Not Found', `Asset ${OLD_SHA} does not exist.`),
    });
    await waitFor(() => expect(pressedTimestamps()).toStrictEqual(['2026-11-24T12:00:00Z']));

    fireEvent.click(screen.getByRole('button', { name: 'Delete' }));
    fireEvent.click(within(screen.getByRole('dialog', { name: 'Delete this clip?' })).getByRole('button', { name: 'Delete clip' }));

    await screen.findByText('Select an asset to see its provenance.');
    expect(deletes(calls)).toHaveLength(1);
    expect(() => card('2026-11-24T12:00:00Z')).toThrow();
    expect(window.location.hash).toBe('#/library');
    expect(screen.getByRole('status').textContent).toBe('The clip was deleted.');
    expect(screen.queryByRole('alert')).toBeNull();
  });

  it('deletes an image', async () => {
    const image = derived([]);
    renderAt(`#/library/${DERIVED_SHA}`, {
      'GET /api/v1/assets': () => ({ items: [image, ...FIRST_PAGE], next_cursor: null }),
      [`DELETE /api/v1/assets/${DERIVED_SHA}`]: () => new StatusAnswer(204),
    });
    await waitFor(() => expect(pressedTimestamps()).toStrictEqual(['2026-11-26T09:00:00Z']));

    fireEvent.click(screen.getByRole('button', { name: 'Delete' }));
    const dialog = screen.getByRole('dialog', { name: 'Delete this image?' });
    expect(dialog.textContent).toContain('Its file (2.0 KB) is deleted from the disk and cannot be restored.');
    fireEvent.click(within(dialog).getByRole('button', { name: 'Delete image' }));

    await screen.findByText('Select an asset to see its provenance.');
    expect(() => card('2026-11-26T09:00:00Z')).toThrow();
    expect(screen.getByRole('status').textContent).toBe('The image was deleted.');
  });

  it('announces each deletion, and keeps Load more once every loaded asset is deleted', async () => {
    renderAt(`#/library/${OLD_SHA}`, {
      [`DELETE /api/v1/assets/${OLD_SHA}`]: () => new StatusAnswer(204),
      [`DELETE /api/v1/assets/${NEW_SHA}`]: () => new StatusAnswer(204),
    });
    await waitFor(() => expect(pressedTimestamps()).toStrictEqual(['2026-11-24T12:00:00Z']));
    const status = screen.getByRole('status');

    fireEvent.click(screen.getByRole('button', { name: 'Delete' }));
    fireEvent.click(within(screen.getByRole('dialog', { name: 'Delete this clip?' })).getByRole('button', { name: 'Delete clip' }));
    await waitFor(() => expect(status.textContent).toBe('The clip was deleted.'));

    // The second clip's deletion says the same words, so the live region is emptied as it starts.
    fireEvent.click(card('2026-11-25T12:00:00Z'));
    await waitFor(() => expect(pressedTimestamps()).toStrictEqual(['2026-11-25T12:00:00Z']));
    screen.getByRole('button', { name: 'Delete' }).focus();
    fireEvent.click(screen.getByRole('button', { name: 'Delete' }));
    fireEvent.click(within(screen.getByRole('dialog', { name: 'Delete this clip?' })).getByRole('button', { name: 'Delete clip' }));
    expect(status.textContent).toBe('');
    await waitFor(() => expect(status.textContent).toBe('The clip was deleted.'));

    // No card is left, but the core has more: the way to them stays, and the Library does not claim to be empty.
    expect(screen.getByRole('button', { name: 'Load more assets' })).toBeTruthy();
    expect(screen.queryByText('No assets yet')).toBeNull();
    // With no card left, focus goes to the view's heading.
    await waitFor(() => expect(document.activeElement).toBe(screen.getByRole('heading', { name: 'Library', level: 1 })));
  });

  it('keeps a deleted asset out of a page that was requested before the deletion', async () => {
    let answer: ((value: unknown) => void) | undefined;
    let pages = 0;
    renderAt(`#/library/${OLD_SHA}`, {
      'GET /api/v1/assets': () => (pages++ === 0 ? { items: FIRST_PAGE, next_cursor: 'page-2' } : new Promise((resolve) => (answer = resolve))),
      [`DELETE /api/v1/assets/${OLD_SHA}`]: () => new StatusAnswer(204),
    });
    await waitFor(() => expect(pressedTimestamps()).toStrictEqual(['2026-11-24T12:00:00Z']));

    fireEvent.click(screen.getByRole('button', { name: 'Load more assets' }));
    await waitFor(() => expect(pages).toBe(2));
    fireEvent.click(screen.getByRole('button', { name: 'Delete' }));
    fireEvent.click(within(screen.getByRole('dialog', { name: 'Delete this clip?' })).getByRole('button', { name: 'Delete clip' }));
    await screen.findByText('Select an asset to see its provenance.');

    // The core answered the page before the deletion went through.
    act(() => answer?.({ items: [asset(OLD_SHA, '2026-11-24T12:00:00Z'), LINKED], next_cursor: null }));
    await waitFor(() => expect(() => card('2026-11-01T08:00:00Z')).not.toThrow());
    expect(() => card('2026-11-24T12:00:00Z')).toThrow();
  });

  it('shows a deleted asset again when a later page lists it, no longer marked as deleted', async () => {
    let pages = 0;
    renderAt(`#/library/${OLD_SHA}`, {
      'GET /api/v1/assets': () =>
        pages++ === 0 ? { items: [derived([OLD_SHA]), ...FIRST_PAGE], next_cursor: 'page-2' } : { items: [asset(OLD_SHA, '2026-11-24T12:00:00Z')], next_cursor: null },
      [`DELETE /api/v1/assets/${OLD_SHA}`]: () => new StatusAnswer(204),
    });
    await waitFor(() => expect(pressedTimestamps()).toStrictEqual(['2026-11-24T12:00:00Z']));
    fireEvent.click(screen.getByRole('button', { name: 'Delete' }));
    fireEvent.click(within(screen.getByRole('dialog', { name: 'Delete this clip?' })).getByRole('button', { name: 'Delete clip' }));
    await screen.findByText('Select an asset to see its provenance.');
    fireEvent.click(card('2026-11-26T09:00:00Z'));
    await waitFor(() => expect(provenance('Parents')).toBe(`${shortHash(OLD_SHA)} (deleted)`));

    // Its bytes were uploaded again, or a job that had uploaded them succeeded.
    fireEvent.click(screen.getByRole('button', { name: 'Load more assets' }));
    await waitFor(() => expect(() => card('2026-11-24T12:00:00Z')).not.toThrow());
    expect(provenance('Parents')).toBe(shortHash(OLD_SHA));
  });

  it('says why Delete is unavailable while another asset is being deleted', async () => {
    let answer: ((value: unknown) => void) | undefined;
    renderAt(`#/library/${OLD_SHA}`, { [`DELETE /api/v1/assets/${OLD_SHA}`]: () => new Promise((resolve) => (answer = resolve)) });
    await waitFor(() => expect(pressedTimestamps()).toStrictEqual(['2026-11-24T12:00:00Z']));
    fireEvent.click(screen.getByRole('button', { name: 'Delete' }));
    fireEvent.click(within(screen.getByRole('dialog', { name: 'Delete this clip?' })).getByRole('button', { name: 'Delete clip' }));

    fireEvent.click(card('2026-11-25T12:00:00Z'));
    await waitFor(() => expect(pressedTimestamps()).toStrictEqual(['2026-11-25T12:00:00Z']));
    expect((screen.getByRole('button', { name: 'Another deletion running' }) as HTMLButtonElement).disabled).toBe(true);

    act(() => answer?.(new StatusAnswer(204)));
    await waitFor(() => expect((screen.getByRole('button', { name: 'Delete' }) as HTMLButtonElement).disabled).toBe(false));
    expect(() => card('2026-11-24T12:00:00Z')).toThrow();
    expect(pressedTimestamps()).toStrictEqual(['2026-11-25T12:00:00Z']); // the selection made meanwhile stays
  });

  it('marks a deleted parent in the provenance and keeps its hash', async () => {
    const image = derived([GONE_SHA, OLD_SHA]);
    const calls = renderAt(`#/library/${DERIVED_SHA}`, {
      'GET /api/v1/assets': () => ({ items: [image, ...FIRST_PAGE], next_cursor: null }),
      [`DELETE /api/v1/assets/${OLD_SHA}`]: () => new StatusAnswer(204),
    });

    // The parent the core no longer has (404) is marked; the one in the list is not looked up.
    await waitFor(() => expect(provenance('Parents')).toBe(`${shortHash(GONE_SHA)} (deleted) · ${shortHash(OLD_SHA)}`));
    expect(calls.filter((call) => call.method === 'GET').map((call) => call.path)).toStrictEqual(['/api/v1/assets', `/api/v1/assets/${GONE_SHA}`]);

    // A parent deleted here is marked without asking the core.
    fireEvent.click(card('2026-11-24T12:00:00Z'));
    await waitFor(() => expect(pressedTimestamps()).toStrictEqual(['2026-11-24T12:00:00Z']));
    fireEvent.click(screen.getByRole('button', { name: 'Delete' }));
    fireEvent.click(within(screen.getByRole('dialog', { name: 'Delete this clip?' })).getByRole('button', { name: 'Delete clip' }));
    await screen.findByText('Select an asset to see its provenance.');

    fireEvent.click(card('2026-11-26T09:00:00Z'));
    await waitFor(() => expect(provenance('Parents')).toBe(`${shortHash(GONE_SHA)} (deleted) · ${shortHash(OLD_SHA)} (deleted)`));
    expect(calls.filter((call) => call.path === `/api/v1/assets/${OLD_SHA}` && call.method === 'GET')).toHaveLength(0);
    expect(calls.filter((call) => call.path === `/api/v1/assets/${GONE_SHA}`)).toHaveLength(1);
  });

  it('does not mark a parent as deleted when looking it up fails', async () => {
    const calls = renderAt(`#/library/${DERIVED_SHA}`, {
      'GET /api/v1/assets': () => ({ items: [derived([GONE_SHA])], next_cursor: null }),
      [`GET /api/v1/assets/${GONE_SHA}`]: () => new StatusAnswer(503),
    });

    await waitFor(() => expect(calls.map((call) => call.path)).toContain(`/api/v1/assets/${GONE_SHA}`));
    await new Promise((resolve) => window.setTimeout(resolve, 0)); // the failed lookup has been handled
    expect(provenance('Parents')).toBe(shortHash(GONE_SHA));
    expect(screen.queryByRole('alert')).toBeNull();
  });
});
