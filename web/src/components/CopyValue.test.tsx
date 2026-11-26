import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it, onTestFinished, vi } from 'vitest';

import { stubApi } from '../test/api';
import { stubClipboard } from '../test/clipboard';
import { AboutDialog } from './AboutDialog';
import { COPIED_MS, COPY_FAILED, CopyValue } from './CopyValue';
import { Hash } from './Hash';
import { JobPanel, type TrackedJob } from './JobPanel';

const JOB_ID = '0193f2a4-7c1e-7b1a-9d3e-5f6a7b8c9d0e';
const SHA = `9f2c41d0${'a'.repeat(48)}77b2e41a`;
const COMMIT = '38758d3c0a5be2f1d4c6e8a9b7f3d2c1e0a9b8c7';

/** Click the button and let the clipboard's answer arrive. */
async function click(button: HTMLElement): Promise<void> {
  await act(async () => {
    fireEvent.click(button);
  });
}

function status(): string {
  return screen.getByRole('status').textContent ?? '';
}

describe('CopyValue', () => {
  it('copies the full value, names the button for what it copies, and confirms with a check icon and "Copied"', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    onTestFinished(() => {
      vi.useRealTimers();
    });
    const writes = stubClipboard();
    render(<CopyValue text="0193f2a4…7b8c9d0e" value={JOB_ID} label="Copy job id" />);

    const button = screen.getByRole('button', { name: 'Copy job id' });
    expect(button.getAttribute('title')).toBe('Copy job id');
    expect(button.classList.contains('cw-btn--ghost')).toBe(true);
    expect(button.classList.contains('cw-btn--sm')).toBe(true);
    expect(button.textContent).toBe(''); // icon only
    expect(screen.getByTitle(JOB_ID).textContent).toBe('0193f2a4…7b8c9d0e');
    expect(button.querySelector('.lucide-copy')).not.toBeNull();
    expect(status()).toBe('');

    await click(button);
    expect(writes).toEqual([JOB_ID]);
    expect(button.querySelector('.lucide-check')).not.toBeNull();
    expect(status()).toBe('Copied');
    expect(screen.queryByRole('alert')).toBeNull();

    act(() => {
      vi.advanceTimersByTime(COPIED_MS);
    });
    expect(button.querySelector('.lucide-copy')).not.toBeNull();
    expect(status()).toBe('');

    await click(button);
    expect(writes).toEqual([JOB_ID, JOB_ID]);
    expect(status()).toBe('Copied');
  });

  it('uses the tooltip when it says more than the name', () => {
    stubClipboard();
    render(<CopyValue text="a +1" value={'a\nb'} label="Copy all 2 model hashes" tooltip="Copy all 2 model hashes, one per line" />);

    const button = screen.getByRole('button', { name: 'Copy all 2 model hashes' });
    expect(button.getAttribute('title')).toBe('Copy all 2 model hashes, one per line');
  });

  it('says so when the browser refuses the clipboard, and not "Copied"', async () => {
    const writes = stubClipboard('refuse');
    render(<CopyValue text="0193f2a4…7b8c9d0e" value={JOB_ID} label="Copy job id" />);
    const button = screen.getByRole('button', { name: 'Copy job id' });

    await click(button);
    expect(writes).toEqual([]);
    expect(screen.getByRole('alert').textContent).toBe(COPY_FAILED);
    expect(status()).toBe('');
    expect(button.querySelector('.lucide-check')).toBeNull();

    // A second refusal mounts the message anew, so it is announced again.
    const first = screen.getByRole('alert');
    await click(button);
    expect(screen.getByRole('alert')).not.toBe(first);
  });

  it('says so when there is no clipboard at all', async () => {
    stubClipboard('missing');
    render(<CopyValue text="0193f2a4…7b8c9d0e" value={JOB_ID} label="Copy job id" />);

    await click(screen.getByRole('button', { name: 'Copy job id' }));
    expect(screen.getByRole('alert').textContent).toBe(COPY_FAILED);
    expect(status()).toBe('');
  });
});

describe('Hash', () => {
  it('shows the hash shortened and copies it as sha256:<hex>, the form of its tooltip', async () => {
    const writes = stubClipboard();
    render(<Hash value={SHA} label="Copy asset hash" />);

    expect(screen.getByTitle(`sha256:${SHA}`).textContent).toBe('sha256:9f2c41d0…77b2e41a');
    await click(screen.getByRole('button', { name: 'Copy asset hash' }));
    expect(writes).toEqual([`sha256:${SHA}`]);
  });
});

describe('copy buttons beside shortened values', () => {
  it('copies the job id in the Job panel', async () => {
    const writes = stubClipboard();
    const job: TrackedJob = {
      id: JOB_ID,
      state: 'running',
      progress: null,
      error: null,
      outputs: [],
      preview: null,
      lastSeq: 0,
      lastStateSeq: 0,
      lastProgressSeq: 0,
    };
    render(<JobPanel job={job} />);

    const panel = screen.getByRole('region', { name: 'Job' });
    expect(within(panel).getByTitle(JOB_ID).textContent).toBe('0193f2a4…7b8c9d0e');
    await click(within(panel).getByRole('button', { name: 'Copy job id' }));
    expect(writes).toEqual([JOB_ID]);
  });

  it('copies the engine version in About', async () => {
    // jsdom has no modal dialogs; open and close only flip the attribute.
    const prototype = HTMLDialogElement.prototype as Partial<HTMLDialogElement>;
    Object.defineProperty(prototype, 'showModal', { value: function (this: HTMLDialogElement) { this.open = true; }, configurable: true });
    Object.defineProperty(prototype, 'close', { value: function (this: HTMLDialogElement) { this.open = false; }, configurable: true });
    onTestFinished(() => {
      Reflect.deleteProperty(prototype, 'showModal');
      Reflect.deleteProperty(prototype, 'close');
    });
    stubApi({
      'GET /api/v1/system': () => ({
        name: 'Cinewright',
        version: '0.1.0',
        api_version: '1',
        data_dir: 'C:/data',
        bind: '127.0.0.1:8737',
        engines: [{ id: 'wan2gp', name: 'WanGP (Wan2GP)', version: COMMIT, state: 'ready', notice: null }],
      }),
    });
    const writes = stubClipboard();
    render(<AboutDialog open onClose={() => undefined} />);

    const button = await screen.findByRole('button', { name: 'Copy WanGP (Wan2GP) version' });
    expect(screen.getByTitle(COMMIT).textContent).toBe(`${COMMIT.slice(0, 8)}…${COMMIT.slice(-8)}`);
    expect(screen.getByText('WanGP (Wan2GP)', { selector: 'dt' }).nextElementSibling?.textContent).toBe(
      `${COMMIT.slice(0, 8)}…${COMMIT.slice(-8)} · Ready`,
    );
    await click(button);
    expect(writes).toEqual([COMMIT]);
  });
});
