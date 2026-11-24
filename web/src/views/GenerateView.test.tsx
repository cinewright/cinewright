import { render, screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { ApiError, type Estimate } from '../api/client';
import { formatDuration, formatEur } from '../lib/format';
import { EstimatePanel } from './GenerateView';

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
