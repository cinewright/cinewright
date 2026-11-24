import { act, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { EventsProvider } from '../api/events';
import { stubApi } from '../test/api';
import { QueueView } from './QueueView';

/** An EventSource the test opens and drops by hand. */
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

  constructor(url: string) {
    this.url = url;
    FakeEventSource.instances.push(this);
  }

  addEventListener(): void {}

  close(): void {
    this.readyState = FakeEventSource.CLOSED;
  }

  open(): void {
    this.readyState = FakeEventSource.OPEN;
    this.onopen?.(new Event('open'));
  }

  /** A dropped connection the browser retries on its own (readyState CONNECTING, not CLOSED). */
  drop(): void {
    this.readyState = FakeEventSource.CONNECTING;
    this.onerror?.(new Event('error'));
  }
}

describe('QueueView', () => {
  it('reads "Connecting" before the event stream first opens and "Reconnecting" after a drop', async () => {
    FakeEventSource.instances = [];
    vi.stubGlobal('EventSource', FakeEventSource);
    stubApi({ 'GET /api/v1/jobs': () => ({ items: [], next_cursor: null }) });

    render(
      <EventsProvider>
        <QueueView />
      </EventsProvider>,
    );
    expect(FakeEventSource.instances).toHaveLength(1);
    const source = FakeEventSource.instances[0] as FakeEventSource;
    expect(source.url).toBe('/api/v1/events');

    expect(screen.getByText('Connecting · updates paused')).toBeTruthy();
    expect(screen.queryByText(/Reconnecting/)).toBeNull();
    await screen.findByText('No jobs yet');

    act(() => source.open());
    expect(screen.getByText('Live')).toBeTruthy();
    expect(screen.queryByText(/updates paused/)).toBeNull();

    act(() => source.drop());
    expect(screen.getByText('Reconnecting · updates paused')).toBeTruthy();
    expect(screen.queryByText('Connecting · updates paused')).toBeNull();
  });
});
