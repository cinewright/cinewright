import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import type { Provenance } from '../api/client';
import { stubClipboard } from '../test/clipboard';
import { ProvenancePanel } from './ProvenancePanel';

const ASSET = 'a'.repeat(64);
const FILE_1 = `11111111${'0'.repeat(48)}1111aaaa`;
const FILE_2 = `22222222${'0'.repeat(48)}2222bbbb`;
const FILE_3 = `33333333${'0'.repeat(48)}3333cccc`;
const PARENT_1 = `44444444${'0'.repeat(48)}4444dddd`;
const PARENT_2 = `55555555${'0'.repeat(48)}5555eeee`;
const JOB_ID = '0193f2a4-7c1e-7b1a-9d3e-5f6a7b8c9d0e';

const MODEL: NonNullable<Provenance['model']> = {
  id: 'ltx-2.3-distilled-22b',
  name: 'LTX-2 2.3 Distilled 22B',
  quant: 'int8',
  engine: 'wan2gp',
  engine_version: '38758d3',
  file_sha256: [FILE_1, FILE_2, FILE_3],
};

function provenance(fields: Partial<Provenance> = {}): Provenance {
  return {
    schema: 'cinewright.provenance/1',
    asset: { sha256: ASSET, media_type: 'video/mp4', size_bytes: 1024 },
    model: MODEL,
    parents: [PARENT_1, PARENT_2],
    created_at: '2026-11-25T12:00:00Z',
    job_id: JOB_ID,
    ...fields,
  };
}

/** The value cell of `term`. */
function cell(term: string): HTMLElement {
  const dd = screen.getByText(term, { selector: 'dt' }).nextElementSibling;
  if (!(dd instanceof HTMLElement)) throw new Error(`No value for ${term}`);
  return dd;
}

async function click(button: HTMLElement): Promise<void> {
  await act(async () => {
    fireEvent.click(button);
  });
}

describe('ProvenancePanel', () => {
  it('copies every model hash, one per line, from the row that shows only the first', async () => {
    const writes = stubClipboard();
    render(<ProvenancePanel provenance={provenance()} />);

    const row = cell('Model hash');
    expect(row.textContent).toBe('sha256:11111111…1111aaaa +2');
    const all = `sha256:${FILE_1}\nsha256:${FILE_2}\nsha256:${FILE_3}`;
    // The tooltip lists them one per line too (getByTitle would fold the line breaks).
    expect(row.querySelector('.copy-value__text')?.getAttribute('title')).toBe(all);
    const button = within(row).getByRole('button', { name: 'Copy all 3 model hashes' });
    expect(button.getAttribute('title')).toBe('Copy all 3 model hashes, one per line');

    await click(button);
    expect(writes).toEqual([all]);
    expect(within(row).getByRole('status').textContent).toBe('Copied');
  });

  it('names a single model hash plainly', async () => {
    const writes = stubClipboard();
    render(<ProvenancePanel provenance={provenance({ model: { ...MODEL, file_sha256: [FILE_1] } })} />);

    const row = cell('Model hash');
    expect(row.textContent).toBe('sha256:11111111…1111aaaa');
    await click(within(row).getByRole('button', { name: 'Copy model hash' }));
    expect(writes).toEqual([`sha256:${FILE_1}`]);
  });

  it('copies each parent hash and the job id', async () => {
    const writes = stubClipboard();
    render(<ProvenancePanel provenance={provenance()} />);

    const parents = cell('Parents');
    expect(within(parents).getByTitle(`sha256:${PARENT_1}`).textContent).toBe('sha256:44444444…4444dddd');
    await click(within(parents).getByRole('button', { name: 'Copy parent hash 2 of 2' }));
    await click(within(parents).getByRole('button', { name: 'Copy parent hash 1 of 2' }));
    const job = cell('Job');
    expect(within(job).getByTitle(JOB_ID).textContent).toBe('0193f2a4…7b8c9d0e');
    await click(within(job).getByRole('button', { name: 'Copy job id' }));
    expect(writes).toEqual([`sha256:${PARENT_2}`, `sha256:${PARENT_1}`, JOB_ID]);
  });

  it('names a single parent plainly and keeps "–" without a button for missing values', () => {
    stubClipboard();
    render(<ProvenancePanel provenance={provenance({ model: null, parents: [PARENT_1], job_id: null })} />);

    expect(within(cell('Parents')).getByRole('button', { name: 'Copy parent hash' })).toBeTruthy();
    for (const term of ['Model hash', 'Job']) {
      expect(cell(term).textContent).toBe('–');
      expect(within(cell(term)).queryByRole('button')).toBeNull();
    }
  });

  it('says so in the row when the clipboard is unavailable', async () => {
    stubClipboard('missing');
    render(<ProvenancePanel provenance={provenance()} />);

    const row = cell('Model hash');
    await click(within(row).getByRole('button', { name: 'Copy all 3 model hashes' }));
    expect(within(row).getByRole('alert').textContent).toBe('Clipboard unavailable, nothing was copied.');
  });
});
