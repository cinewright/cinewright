import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { ApiError, type Asset, type CapabilityEntry, type Estimate, type Job, type JobRequest, type Limits, type ModelSummary } from '../api/client';
import { EventsProvider } from '../api/events';
import { formatDuration, formatEur } from '../lib/format';
import { ProblemAnswer, StatusAnswer, stubApi, type ApiCall, type Routes } from '../test/api';
import { stubEventSource, type FakeEventSource } from '../test/events';
import { EstimatePanel, GenerateView } from './GenerateView';

const COST_EUR = 3.47;
const ETA_SECONDS = 754;
const COST = formatEur(COST_EUR);
const ETA = formatDuration(ETA_SECONDS);

const ROUTE = { model_id: 'z-image-turbo', quant: 'bf16', host_id: 'local', engine: 'wan2gp' };
const FEASIBLE: Estimate = { feasible: true, requires_approval: true, cost_eur: COST_EUR, eta_seconds: ETA_SECONDS, route: ROUTE, reasons: [] };
const INFEASIBLE: Estimate = {
  feasible: false,
  requires_approval: false,
  cost_eur: COST_EUR,
  eta_seconds: ETA_SECONDS,
  reasons: ['No installed model offers t2v'],
};

function panel(): HTMLElement {
  return screen.getByRole('region', { name: 'Estimate' });
}

/** The screen-reader live region: the sr-only paragraph with role="status". */
function liveRegion(): HTMLElement {
  const region = panel().querySelector<HTMLElement>('p.sr-only[role="status"]');
  if (!region) throw new Error('EstimatePanel has no sr-only status paragraph');
  return region;
}

/** Every element in the panel, the panel included, that a screen reader announces on change. */
function announcedElements(): HTMLElement[] {
  const root = panel();
  const live = [root, ...root.querySelectorAll<HTMLElement>('*')].filter((element) => element.matches('[aria-live]:not([aria-live="off"])'));
  return [...within(root).getAllByRole('status'), ...live];
}

describe('EstimatePanel', () => {
  it('announces only "can run"; cost and ETA stay out of every status element', () => {
    render(<EstimatePanel state={{ status: 'done', estimate: FEASIBLE }} hasRequest />);

    expect(liveRegion().textContent).toBe('Estimate ready: can run');
    // The values are on screen, so their absence from the status elements means something.
    expect(within(panel()).getByText(COST)).toBeTruthy();
    expect(within(panel()).getByText(ETA)).toBeTruthy();
    const announced = announcedElements();
    // The live region and the approval banner.
    expect(announced.length).toBeGreaterThanOrEqual(2);
    expect(announced.some((element) => element.textContent?.includes('Approval needed'))).toBe(true);
    for (const element of announced) {
      expect(element.textContent).not.toContain(COST);
      expect(element.textContent).not.toContain(ETA);
    }
  });

  it('announces only "no model can run this request" for an infeasible estimate', () => {
    render(<EstimatePanel state={{ status: 'done', estimate: INFEASIBLE }} hasRequest />);

    expect(liveRegion().textContent).toBe('Estimate ready: no model can run this request');
    expect(within(panel()).getByText(COST)).toBeTruthy();
    for (const element of announcedElements()) {
      expect(element.textContent).not.toContain(COST);
      expect(element.textContent).not.toContain(ETA);
    }
  });

  it('announces only "Estimate unavailable" when the estimate failed', () => {
    const error = new ApiError(500, 'Internal Server Error', 'The router failed');
    render(<EstimatePanel state={{ status: 'error', error }} hasRequest />);

    expect(liveRegion().textContent).toBe('Estimate unavailable');
  });
});

// ------------------------------------------------------------------------------------------- the whole view

const LTX = 'ltx-2.3-distilled-22b';
const LTX_NAME = 'LTX-2 2.3 Distilled 22B';
const WAN = 'wan-2.2-t2v-14b';
const WAN_NAME = 'Wan 2.2 Text-to-Video 14B';
// Shortened resolution lists; the grids and frame rates are the registry's.
const LTX_LIMITS: Limits = { resolutions: ['1280x704', '832x448'], max_frames: 737, min_frames: 17, frames_step: 8, fps: [24], max_duration_s: 30 };
const WAN_LIMITS: Limits = { resolutions: ['1280x720', '832x480'], max_frames: 737, min_frames: 5, frames_step: 4, fps: [16], max_duration_s: 46 };
const LICENSE = { name: 'Apache-2.0', commercial_use: true, geo_exclusions: [], url: null };
const CREATED = '2026-11-25T12:00:00Z';
const INPUT_SHA = 'a'.repeat(64);
const OUTPUT_SHA = 'b'.repeat(64);
const JOB_ID = '0193f2a4-7c1e-7b1a-9d3e-5f6a7b8c9d0e';

function entry(capability: CapabilityEntry['capability'], modelId: string, modelName: string, limits: Limits): CapabilityEntry {
  return { capability, model_id: modelId, model_name: modelName, quant: 'int8', limits, status: null, hosts: ['local'] };
}

function model(id: string, name: string, capabilities: ModelSummary['capabilities']): ModelSummary {
  return {
    id,
    name,
    vendor: 'Vendor',
    family: id,
    capabilities,
    license: LICENSE,
    installed_bytes: 0,
    quantizations: [{ quant: 'int8', install_state: 'installed', size_bytes: 1 }],
  };
}

function feasible(modelId: string): Estimate {
  return {
    feasible: true,
    requires_approval: false,
    cost_eur: 0,
    eta_seconds: null,
    route: { model_id: modelId, quant: 'int8', host_id: 'local', engine: 'wan2gp', reasons: [] },
    reasons: [],
  };
}

const INPUT: Asset = {
  sha256: INPUT_SHA,
  media_type: 'image/png',
  size_bytes: 2048,
  data_class: 'private',
  job_id: null,
  created_at: CREATED,
  provenance: { schema: 'cinewright.provenance/1', asset: { sha256: INPUT_SHA, media_type: 'image/png', size_bytes: 2048 }, created_at: CREATED },
};

function job(request: JobRequest): Job {
  return { id: JOB_ID, state: 'queued', request, client: 'ui', created_at: CREATED, outputs: [] };
}

function renderView(routes: Routes): { calls: ApiCall[]; source: () => FakeEventSource } {
  const source = stubEventSource();
  const calls = stubApi(routes);
  render(
    <EventsProvider>
      <GenerateView />
    </EventsProvider>,
  );
  act(() => source().open());
  return { calls, source };
}

function event(kind: string, seq: number, data: Record<string, unknown>, jobId = JOB_ID) {
  return { job_id: jobId, seq, kind, data, created_at: CREATED };
}

function estimates(calls: ApiCall[]): JobRequest[] {
  return calls.filter((call) => call.method === 'POST' && call.path === '/api/v1/estimate').map((call) => call.body as JobRequest);
}

function framesInput(): HTMLInputElement {
  return screen.getByRole('spinbutton', { name: 'Frames' }) as HTMLInputElement;
}

function select(name: string): HTMLSelectElement {
  return screen.getByRole('combobox', { name }) as HTMLSelectElement;
}

function optionTexts(name: string): string[] {
  return [...select(name).options].map((option) => option.text);
}

function setFrames(value: string): void {
  fireEvent.change(framesInput(), { target: { value } });
  fireEvent.blur(framesInput());
}

function enterPrompt(): void {
  fireEvent.change(screen.getByRole('textbox', { name: /^Prompt/ }), { target: { value: 'Harbour at night' } });
}

function pinModel(id: string): void {
  fireEvent.change(screen.getByRole('combobox', { name: /^Model/ }), { target: { value: id } });
}

function chooseFile(file: File): void {
  fireEvent.change(screen.getByLabelText('Start image file'), { target: { files: [file] } });
}

function uploads(calls: ApiCall[]): ApiCall[] {
  return calls.filter((call) => call.method === 'POST' && call.path === '/api/v1/assets');
}

/** The texts of every live status region in the view. */
function statusTexts(): string[] {
  return screen.getAllByRole('status').map((element) => element.textContent ?? '');
}

// Tests that wait for the estimates to settle run on real timers, through several 300 ms debounces.
const SLOW_TEST_MS = 15_000;

/** Wait until no new estimate has been requested for `quietMs` (the debounce is 300 ms); returns the count. */
async function settled(calls: ApiCall[], quietMs = 600): Promise<number> {
  let count = -1;
  while (count !== estimates(calls).length) {
    count = estimates(calls).length;
    await act(() => new Promise((resolve) => window.setTimeout(resolve, quietMs)));
  }
  return count;
}

describe('GenerateView', () => {
  it('uploads a start image, submits the job, shows its progress and preview frames, and links the output into the Library', async () => {
    const holder: { source?: () => FakeEventSource } = {};
    const { calls, source } = renderView({
      'GET /api/v1/capabilities': () => ({ items: [entry('i2v', LTX, LTX_NAME, LTX_LIMITS)] }),
      'GET /api/v1/models': () => ({ items: [model(LTX, LTX_NAME, ['i2v'])] }),
      'POST /api/v1/estimate': () => feasible(LTX),
      'POST /api/v1/assets': () => INPUT,
      'POST /api/v1/jobs': (body) => {
        // The worker reports before the submit answer arrives; the view must not lose this event.
        holder.source?.().emit('state', event('state', 2, { state: 'leased', worker_id: 'worker-1' }));
        return job(body as JobRequest);
      },
    });
    holder.source = source;

    // Only i2v is offered, so it is selected; pin the model and enter a prompt.
    await screen.findByRole('option', { name: LTX_NAME });
    expect(select('Capability').value).toBe('i2v');
    fireEvent.change(screen.getByRole('combobox', { name: /^Model/ }), { target: { value: LTX } });
    enterPrompt();
    await screen.findByRole('button', { name: 'Add a start image' });
    expect(screen.getByText(`Choices of ${LTX_NAME}.`)).toBeTruthy();

    // A file without a media type is refused in the view, before any request.
    const untyped = new File([new Uint8Array([1, 2, 3])], 'scan.heic', { type: '' });
    fireEvent.change(screen.getByLabelText('Start image file'), { target: { files: [untyped] } });
    expect(screen.getByRole('alert').textContent).toBe('scan.heic has no known image type, so it is not uploaded; choose a PNG, JPEG or WebP image.');
    expect(calls.some((call) => call.path === '/api/v1/assets')).toBe(false);

    const file = new File([new Uint8Array([137, 80, 78, 71])], 'start.png', { type: 'image/png' });
    // The focused button gives way to the thumbnail; keyboard focus moves on to "Replace image".
    screen.getByRole('button', { name: 'Choose image' }).focus();
    chooseFile(file);
    const preview = await screen.findByRole('img', { name: 'Start image preview' });
    expect(preview.getAttribute('src')).toBe(`/api/v1/assets/${INPUT_SHA}/content`);
    expect(screen.queryByRole('alert')).toBeNull();
    const upload = uploads(calls)[0];
    expect(upload?.query).toStrictEqual({ data_class: 'private' });
    expect(upload?.headers).toStrictEqual({ Accept: 'application/json', 'Content-Type': 'image/png' });
    expect(upload?.rawBody).toBe(file);
    expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Replace image' }));
    expect(statusTexts()).toContain('Start image uploaded');

    fireEvent.click(await screen.findByRole('button', { name: 'Generate' }));
    const panel = await screen.findByRole('region', { name: 'Job' });
    const submitted = calls.find((call) => call.method === 'POST' && call.path === '/api/v1/jobs')?.body as JobRequest;
    expect(submitted).toMatchObject({ capability: 'i2v', model: LTX, prompt: 'Harbour at night', inputs: [INPUT_SHA] });
    // Resolution, frames and frame rate stay the model's defaults unless set.
    expect(submitted.params).toStrictEqual({ seed: null });
    expect(within(panel).getByRole('progressbar').getAttribute('aria-label')).toBe('Starting');
    // The submit is announced: the status region existed before the job did.
    expect(statusTexts()).toContain('Job starting');

    act(() => source().emit('state', event('state', 3, { state: 'running' })));
    act(() => source().emit('progress', event('progress', 4, { phase: 'denoise', step: 3, total_steps: 8, percent: 37.5 })));
    // Another job's events do not move this one.
    act(() => source().emit('progress', event('progress', 9, { phase: 'denoise', step: 7, total_steps: 8, percent: 90 }, 'other-job')));
    expect(within(panel).getByRole('progressbar').getAttribute('aria-label')).toBe('Step 3/8');
    expect(within(panel).getByRole('progressbar').getAttribute('aria-valuenow')).toBe('38');

    // Previews repeat the seq of the latest stored event: each one shows, none is dropped as a repeat.
    act(() => source().emit('preview', event('preview', 4, { media_type: 'image/webp', data_base64: 'UklGRgAA' })));
    const frame = () => within(panel).getByRole('img', { name: 'Latest preview frame' }).getAttribute('src');
    expect(frame()).toBe('data:image/webp;base64,UklGRgAA');
    act(() => source().emit('preview', event('preview', 4, { media_type: 'image/jpeg', data_base64: '/9j/4AAQ' })));
    expect(frame()).toBe('data:image/jpeg;base64,/9j/4AAQ');
    // A type outside the allow-list is not rendered.
    act(() => source().emit('preview', event('preview', 4, { media_type: 'image/svg+xml', data_base64: 'PHN2Zz4=' })));
    expect(frame()).toBe('data:image/jpeg;base64,/9j/4AAQ');
    // A replayed stored event is applied once.
    act(() => source().emit('progress', event('progress', 4, { phase: 'denoise', step: 1, total_steps: 8, percent: 12.5 })));
    expect(within(panel).getByRole('progressbar').getAttribute('aria-valuenow')).toBe('38');
    expect(within(panel).queryByRole('link', { name: 'Open in Library' })).toBeNull();

    act(() => source().emit('state', event('state', 5, { state: 'succeeded', outputs: [OUTPUT_SHA] })));
    expect(within(panel).getByRole('link', { name: 'Open in Library' }).getAttribute('href')).toBe(`#/library/${OUTPUT_SHA}`);
    expect(within(panel).getByRole('progressbar').getAttribute('aria-label')).toBe('Done');
    expect(statusTexts()).toContain('Job done');
  });

  it('uploads the input image again when a stricter data class is chosen, also while its first upload runs', async () => {
    // The core answers with the weakest class: the view must go by the class it sent, or it would upload forever.
    const PUBLIC_INPUT: Asset = { ...INPUT, data_class: 'public' };
    let release: ((asset: Asset) => void) | undefined;
    const { calls } = renderView({
      'GET /api/v1/capabilities': () => ({ items: [entry('i2v', LTX, LTX_NAME, LTX_LIMITS), entry('t2v', LTX, LTX_NAME, LTX_LIMITS)] }),
      'GET /api/v1/models': () => ({ items: [model(LTX, LTX_NAME, ['i2v', 't2v'])] }),
      'POST /api/v1/estimate': () => feasible(LTX),
      'POST /api/v1/assets': () => {
        if (uploads(calls).length === 1) return new Promise<Asset>((resolve) => (release = resolve));
        // Ends an upload loop with a refusal, so the counts below fail instead of the test hanging.
        return uploads(calls).length > 8 ? new ProblemAnswer(422, 'Invalid request', 'Too many uploads.') : PUBLIC_INPUT;
      },
    });
    await screen.findByRole('option', { name: LTX_NAME });
    fireEvent.change(select('Capability'), { target: { value: 'i2v' } });
    enterPrompt();
    const file = new File([new Uint8Array([137, 80, 78, 71])], 'face.png', { type: 'image/png' });
    chooseFile(file);
    await screen.findByRole('button', { name: 'Uploading the start image' });

    // Biometric while the first upload (as private) still runs: once it is stored, the same file goes again.
    fireEvent.click(screen.getByRole('radio', { name: 'Biometric' }));
    expect(uploads(calls)).toHaveLength(1);
    act(() => release?.(PUBLIC_INPUT));
    await waitFor(() => expect(uploads(calls)).toHaveLength(2));
    expect(uploads(calls).map((call) => call.query)).toStrictEqual([{ data_class: 'private' }, { data_class: 'biometric' }]);
    expect(uploads(calls)[1]?.rawBody).toBe(file);
    expect(await screen.findByRole('button', { name: 'Generate' })).toBeTruthy();
    await waitFor(() => expect(estimates(calls).at(-1)).toMatchObject({ data_class: 'biometric', inputs: [INPUT_SHA] }));

    // A looser class, or the same one again, needs no upload: the core never loosens a stored class.
    fireEvent.click(screen.getByRole('radio', { name: 'Public' }));
    fireEvent.click(screen.getByRole('radio', { name: 'Biometric' }));
    await settled(calls);
    expect(uploads(calls)).toHaveLength(2);

    // Removed, then chosen again as public; focus stays in the field.
    fireEvent.click(screen.getByRole('radio', { name: 'Public' }));
    screen.getByRole('button', { name: 'Remove image' }).focus();
    fireEvent.click(screen.getByRole('button', { name: 'Remove image' }));
    expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Choose image' }));
    chooseFile(file);
    await waitFor(() => expect(uploads(calls)).toHaveLength(3));
    expect(uploads(calls)[2]?.query).toStrictEqual({ data_class: 'public' });
    await screen.findByRole('img', { name: 'Start image preview' });

    // A text-to-video request takes no image, so the stricter class waits until i2v is selected again.
    fireEvent.change(select('Capability'), { target: { value: 't2v' } });
    fireEvent.click(screen.getByRole('radio', { name: 'Private' }));
    await settled(calls);
    expect(uploads(calls)).toHaveLength(3);
    fireEvent.change(select('Capability'), { target: { value: 'i2v' } });
    await waitFor(() => expect(uploads(calls)).toHaveLength(4));
    expect(uploads(calls)[3]?.query).toStrictEqual({ data_class: 'private' });
    expect(await screen.findByRole('button', { name: 'Generate' })).toBeTruthy();
  }, SLOW_TEST_MS);

  it('drops the input image when its upload for a stricter data class fails', async () => {
    const { calls } = renderView({
      'GET /api/v1/capabilities': () => ({ items: [entry('i2v', LTX, LTX_NAME, LTX_LIMITS)] }),
      'GET /api/v1/models': () => ({ items: [model(LTX, LTX_NAME, ['i2v'])] }),
      'POST /api/v1/estimate': () => feasible(LTX),
      'POST /api/v1/assets': () => (uploads(calls).length === 1 ? INPUT : new ProblemAnswer(422, 'Invalid request', 'The upload stalled.')),
    });
    await screen.findByRole('option', { name: LTX_NAME });
    enterPrompt();
    chooseFile(new File([new Uint8Array([137, 80, 78, 71])], 'face.png', { type: 'image/png' }));
    await screen.findByRole('img', { name: 'Start image preview' });

    fireEvent.click(screen.getByRole('radio', { name: 'Biometric' }));
    expect((await screen.findByRole('alert')).textContent).toBe('The upload stalled.');
    // Not tried again, and the image stored as private is no longer used.
    expect(uploads(calls)).toHaveLength(2);
    expect(screen.queryByRole('img', { name: 'Start image preview' })).toBeNull();
    expect(screen.getByRole('button', { name: 'Add a start image' })).toBeTruthy();
  });

  it('shows the full cause of a failed job and offers no Library link', async () => {
    const { source } = renderView({
      'GET /api/v1/capabilities': () => ({ items: [entry('t2v', LTX, LTX_NAME, LTX_LIMITS)] }),
      'GET /api/v1/models': () => ({ items: [model(LTX, LTX_NAME, ['t2v'])] }),
      'POST /api/v1/estimate': () => feasible(LTX),
      'POST /api/v1/jobs': (body) => job(body as JobRequest),
    });
    await screen.findByRole('option', { name: LTX_NAME });
    enterPrompt();
    fireEvent.click(await screen.findByRole('button', { name: 'Generate' }));
    const panel = await screen.findByRole('region', { name: 'Job' });
    expect(statusTexts()).toContain('Job queued');

    const message =
      'Out of memory at step 3 of 8 (denoise). Wan2GP: torch.OutOfMemoryError: CUDA out of memory, unsufficient VRAM for 1280x704 at 241 frames.';
    act(() => source().emit('state', event('state', 2, { state: 'failed', error: { code: 'out_of_memory', message, retryable: false } })));
    const alert = within(panel).getByRole('alert');
    expect(within(alert).getByText('The job failed')).toBeTruthy();
    expect(within(alert).getByText(message)).toBeTruthy();
    const bar = within(panel).getByRole('progressbar');
    expect(bar.getAttribute('aria-label')).toBe('Failed');
    expect(bar.getAttribute('data-state')).toBe('fail');
    expect(within(panel).queryByRole('link', { name: /Library/ })).toBeNull();
  });

  it('loads the job again when the stream reopens, as a new stream replays nothing', async () => {
    let submitted: Job | undefined;
    const { calls, source } = renderView({
      'GET /api/v1/capabilities': () => ({ items: [entry('t2v', LTX, LTX_NAME, LTX_LIMITS)] }),
      'GET /api/v1/models': () => ({ items: [model(LTX, LTX_NAME, ['t2v'])] }),
      'POST /api/v1/estimate': () => feasible(LTX),
      'POST /api/v1/jobs': (body) => (submitted = job(body as JobRequest)),
      [`GET /api/v1/jobs/${JOB_ID}`]: () => ({ ...submitted, state: 'succeeded', outputs: [OUTPUT_SHA] }),
    });
    await screen.findByRole('option', { name: LTX_NAME });
    enterPrompt();
    fireEvent.click(await screen.findByRole('button', { name: 'Generate' }));
    const panel = await screen.findByRole('region', { name: 'Job' });
    expect(calls.some((call) => call.path === `/api/v1/jobs/${JOB_ID}`)).toBe(false);

    // The stream drops, the job finishes meanwhile, and the stream opens again.
    act(() => source().onerror?.(new Event('error')));
    act(() => source().open());
    const link = await within(panel).findByRole('link', { name: 'Open in Library' });
    expect(link.getAttribute('href')).toBe(`#/library/${OUTPUT_SHA}`);
    expect(within(panel).getByRole('progressbar').getAttribute('aria-label')).toBe('Done');
    expect(calls.filter((call) => call.method === 'GET' && call.path === `/api/v1/jobs/${JOB_ID}`)).toHaveLength(1);
  });

  it('shows readable upload refusals', async () => {
    let answer: unknown = new StatusAnswer(413);
    renderView({
      'GET /api/v1/capabilities': () => ({ items: [entry('i2v', LTX, LTX_NAME, LTX_LIMITS)] }),
      'GET /api/v1/models': () => ({ items: [model(LTX, LTX_NAME, ['i2v'])] }),
      'POST /api/v1/estimate': () => feasible(LTX),
      'POST /api/v1/assets': () => answer,
    });
    await screen.findByRole('option', { name: LTX_NAME });
    const choose = (name: string, type: string) =>
      fireEvent.change(screen.getByLabelText('Start image file'), { target: { files: [new File([new Uint8Array([1])], name, { type })] } });

    choose('big.png', 'image/png');
    expect((await screen.findByRole('alert')).textContent).toBe('The image is too large for the core, so it was not uploaded.');

    answer = new StatusAnswer(415);
    choose('odd.png', 'image/png');
    await waitFor(() =>
      expect(screen.getByRole('alert').textContent).toBe('The core refused the file type, so the image was not uploaded; choose a PNG, JPEG or WebP image.'),
    );

    // The core's own refusals (422) carry a readable detail.
    answer = new ProblemAnswer(422, 'Invalid request', 'Unsupported media type image/png.');
    choose('odd.png', 'image/png');
    await waitFor(() => expect(screen.getByRole('alert').textContent).toBe('Unsupported media type image/png.'));

    choose('anim.gif', 'image/gif');
    expect(screen.getByRole('alert').textContent).toBe('anim.gif is image/gif, so it is not uploaded; choose a PNG, JPEG or WebP image.');
    expect(screen.queryByRole('img', { name: 'Start image preview' })).toBeNull();
  });

  it('follows a change of the routed model once, fitting the fields to its limits without a loop', async () => {
    // The router picks Wan for private requests and LTX for public ones.
    const { calls } = renderView({
      'GET /api/v1/capabilities': () => ({ items: [entry('t2v', LTX, LTX_NAME, LTX_LIMITS), entry('t2v', WAN, WAN_NAME, WAN_LIMITS)] }),
      'GET /api/v1/models': () => ({ items: [model(LTX, LTX_NAME, ['t2v']), model(WAN, WAN_NAME, ['t2v'])] }),
      'POST /api/v1/estimate': (body) => feasible((body as JobRequest).data_class === 'public' ? LTX : WAN),
    });
    await screen.findByRole('option', { name: WAN_NAME });
    // No model known yet: the fields wait for one.
    expect(framesInput().disabled).toBe(true);
    enterPrompt();

    await screen.findByText(`Choices of ${WAN_NAME}, the model the router picked.`);
    expect(optionTexts('Frame rate')).toStrictEqual(['Model default', '16 fps']);
    expect(optionTexts('Resolution')).toStrictEqual(['Model default', '1280x720', '832x480']);
    fireEvent.change(select('Resolution'), { target: { value: '1280x720' } });
    fireEvent.change(select('Frame rate'), { target: { value: '16' } });
    // 0 is below the grid: the button and the estimate say what the field needs, and leaving it snaps to the minimum.
    fireEvent.change(framesInput(), { target: { value: '0' } });
    expect(screen.getByRole('button', { name: 'Frames must be 5 + a multiple of 4, up to 737' })).toBeTruthy();
    expect(within(screen.getByRole('region', { name: 'Estimate' })).getByText('Frames must be 5 + a multiple of 4, up to 737.')).toBeTruthy();
    fireEvent.blur(framesInput());
    expect(framesInput().value).toBe('5');
    setFrames('30');
    // Off Wan's grid (5 + 4n): snapped when the field is left.
    expect(framesInput().value).toBe('29');
    await settled(calls);
    expect(estimates(calls).at(-1)?.params).toStrictEqual({ seed: null, resolution: '1280x720', frames: 29, fps: 16 });

    // A public request routes to LTX: its limits replace Wan's; what LTX lacks falls back to its default or snaps.
    fireEvent.click(screen.getByRole('radio', { name: 'Public' }));
    await screen.findByText(`Choices of ${LTX_NAME}, the model the router picked.`);
    await waitFor(() => expect(framesInput().value).toBe('33'));
    expect(optionTexts('Frame rate')).toStrictEqual(['Model default', '24 fps']);
    expect(select('Resolution').value).toBe('');
    expect(select('Frame rate').value).toBe('');
    expect(screen.getByText('17 + a multiple of 8, up to 737; empty = model default')).toBeTruthy();

    const before = estimates(calls).length;
    const after = await settled(calls);
    // At most the public request and the fitted one, then quiet.
    expect(after - before).toBeLessThanOrEqual(2);
    expect(estimates(calls).at(-1)).toMatchObject({ data_class: 'public' });
    expect(estimates(calls).at(-1)?.params).toStrictEqual({ seed: null, frames: 33 });
    expect(await screen.findByRole('button', { name: 'Generate' })).toBeTruthy();
  }, SLOW_TEST_MS);

  it('snaps at most once per edit when every snap moves the route to a model with another grid', async () => {
    // Two models whose grids share no count, and a router that answers with the other one on every call:
    // snapping on every route change would never stop.
    const P: Limits = { resolutions: [], max_frames: 400, min_frames: 8, frames_step: 8, fps: [] };
    const Q: Limits = { resolutions: [], max_frames: 404, min_frames: 4, frames_step: 8, fps: [] };
    let call = 0;
    const { calls } = renderView({
      'GET /api/v1/capabilities': () => ({ items: [entry('t2v', 'model-p', 'Model P', P), entry('t2v', 'model-q', 'Model Q', Q)] }),
      'GET /api/v1/models': () => ({ items: [model('model-p', 'Model P', ['t2v']), model('model-q', 'Model Q', ['t2v'])] }),
      'POST /api/v1/estimate': () => feasible(call++ % 2 === 0 ? 'model-p' : 'model-q'),
    });
    await screen.findByRole('option', { name: 'Model Q' });
    enterPrompt();
    await screen.findByText('Choices of Model P, the model the router picked.');

    setFrames('16'); // on P's grid
    await screen.findByText('Choices of Model Q, the model the router picked.');
    await waitFor(() => expect(framesInput().value).toBe('20')); // snapped once, to Q's grid
    await screen.findByText('Choices of Model P, the model the router picked.');

    // Prompt, edit, one snap; then quiet (a slow runner may merge requests, never add any).
    expect(await settled(calls)).toBeLessThanOrEqual(3);
    // Left off P's grid: the request waits for the user instead of snapping again.
    expect(framesInput().value).toBe('20');
    expect(framesInput().getAttribute('aria-invalid')).toBe('true');
    expect(screen.getByRole('button', { name: 'Frames must be 8 + a multiple of 8, up to 400' })).toBeTruthy();
  }, SLOW_TEST_MS);

  it('clears resolution, frames and frame rate when the capability changes, so no kept value blocks the new route', async () => {
    const Z_LIMITS: Limits = { resolutions: ['1440x1440', '1024x1024'], max_frames: null, min_frames: null, frames_step: null, fps: [] };
    // LTX refuses a resolution it does not list, as the router does.
    const { calls } = renderView({
      'GET /api/v1/capabilities': () => ({ items: [entry('t2i', 'z-image-turbo', 'Z-Image Turbo 6B', Z_LIMITS), entry('t2v', LTX, LTX_NAME, LTX_LIMITS)] }),
      'GET /api/v1/models': () => ({ items: [model('z-image-turbo', 'Z-Image Turbo 6B', ['t2i']), model(LTX, LTX_NAME, ['t2v'])] }),
      'POST /api/v1/estimate': (body) => {
        const request = body as JobRequest;
        if (request.capability === 't2i') return feasible('z-image-turbo');
        const resolution = request.params?.resolution;
        if (resolution && !(LTX_LIMITS.resolutions ?? []).includes(resolution))
          return new ProblemAnswer(409, 'No worker can run this request', `ltx-2.3-distilled-22b does not support ${resolution}`);
        return feasible(LTX);
      },
    });
    await screen.findByRole('option', { name: 'Z-Image Turbo 6B' });
    expect(select('Capability').value).toBe('t2i');
    enterPrompt();
    await screen.findByText('Choices of Z-Image Turbo 6B, the model the router picked.');
    fireEvent.change(select('Resolution'), { target: { value: '1440x1440' } });
    await waitFor(() => expect(estimates(calls).at(-1)?.params).toStrictEqual({ seed: null, resolution: '1440x1440' }));

    fireEvent.change(select('Capability'), { target: { value: 't2v' } });
    expect(select('Resolution').value).toBe('');
    await screen.findByText(`Choices of ${LTX_NAME}, the model the router picked.`);
    expect(select('Resolution').disabled).toBe(false);
    expect(estimates(calls).at(-1)).toMatchObject({ capability: 't2v', params: { seed: null } });
    expect(estimates(calls).at(-1)?.params).toStrictEqual({ seed: null });
    expect(await screen.findByRole('button', { name: 'Generate' })).toBeTruthy();
  });

  it('clears resolution, frames and frame rate when unpinning leaves no model to offer them', async () => {
    // Wan is not installed, so no estimate pinned to it routes; LTX refuses a resolution it does not list.
    const { calls } = renderView({
      'GET /api/v1/capabilities': () => ({ items: [entry('t2v', LTX, LTX_NAME, LTX_LIMITS), entry('t2v', WAN, WAN_NAME, WAN_LIMITS)] }),
      'GET /api/v1/models': () => ({ items: [model(LTX, LTX_NAME, ['t2v']), model(WAN, WAN_NAME, ['t2v'])] }),
      'POST /api/v1/estimate': (body) => {
        const request = body as JobRequest;
        if (request.model === WAN) return new ProblemAnswer(409, 'No worker can run this request', 'wan-2.2-t2v-14b has no installed quantization');
        const resolution = request.params?.resolution;
        if (resolution && !(LTX_LIMITS.resolutions ?? []).includes(resolution))
          return new ProblemAnswer(409, 'No worker can run this request', `ltx-2.3-distilled-22b does not support ${resolution}`);
        return feasible(LTX);
      },
    });
    await screen.findByRole('option', { name: WAN_NAME });
    pinModel(WAN);
    enterPrompt();
    await screen.findByText(`Choices of ${WAN_NAME}.`);
    fireEvent.change(select('Resolution'), { target: { value: '1280x720' } });
    fireEvent.change(select('Frame rate'), { target: { value: '16' } });
    setFrames('29');
    await waitFor(() => expect(estimates(calls).at(-1)).toMatchObject({ model: WAN, params: { resolution: '1280x720', frames: 29, fps: 16 } }));
    // No estimate has routed yet, so the hint asks for a pin rather than a prompt.
    pinModel('');
    expect(screen.getByText('Pin a model to set these; no estimate has named a model yet.')).toBeTruthy();

    await screen.findByText(`Choices of ${LTX_NAME}, the model the router picked.`);
    expect(select('Resolution').value).toBe('');
    expect(framesInput().value).toBe('');
    expect(select('Frame rate').value).toBe('');
    expect(estimates(calls).at(-1)).toMatchObject({ model: null });
    expect(estimates(calls).at(-1)?.params).toStrictEqual({ seed: null });
    expect(await screen.findByRole('button', { name: 'Generate' })).toBeTruthy();
  });

  it('hides frames and frame rate for still images and sends neither', async () => {
    const { calls } = renderView({
      'GET /api/v1/capabilities': () => ({ items: [entry('t2i', LTX, LTX_NAME, LTX_LIMITS)] }),
      'GET /api/v1/models': () => ({ items: [model(LTX, LTX_NAME, ['t2i'])] }),
      'POST /api/v1/estimate': () => feasible(LTX),
    });
    await screen.findByRole('option', { name: LTX_NAME });
    enterPrompt();
    await screen.findByText(`Choices of ${LTX_NAME}, the model the router picked.`);
    expect(screen.queryByRole('spinbutton', { name: 'Frames' })).toBeNull();
    expect(screen.queryByRole('combobox', { name: 'Frame rate' })).toBeNull();
    expect(screen.queryByLabelText('Start image file')).toBeNull();
    fireEvent.change(select('Resolution'), { target: { value: '832x448' } });
    await waitFor(() => expect(estimates(calls).at(-1)?.params).toStrictEqual({ seed: null, resolution: '832x448' }));
    expect(await screen.findByRole('button', { name: 'Generate' })).toBeTruthy();
  });
});
