import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import type { Settings } from '../api/client';
import { stubApi } from '../test/api';
import { SettingsView } from './SettingsView';

const SAVED: Settings = {
  update_channel: 'stable',
  country: 'DE',
  approval_threshold_eur: 5,
  budget_eur_per_month: 40,
  model_store_quota_gb: 200,
  kwh_price_eur: 0.32,
};

describe('SettingsView', () => {
  it('sends an emptied optional field as null and leaves untouched fields out of the PATCH body', async () => {
    const calls = stubApi({
      'GET /api/v1/settings': () => SAVED,
      'GET /api/v1/licenses': () => ({ items: [] }),
      'PATCH /api/v1/settings': (patch) => ({ ...SAVED, ...(patch as Partial<Settings>) }),
    });
    render(<SettingsView />);

    // The wrapping label also holds the hint, so match its start.
    const budget = await screen.findByLabelText(/^Monthly budget/);
    expect(budget).toHaveProperty('value', '40');
    fireEvent.change(budget, { target: { value: '' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save settings' }));
    await screen.findByText('Saved.');

    const patches = calls.filter((call) => call.method === 'PATCH');
    expect(patches).toHaveLength(1);
    expect(patches[0]?.path).toBe('/api/v1/settings');
    expect(patches[0]?.body).toStrictEqual({ budget_eur_per_month: null });
  });
});
