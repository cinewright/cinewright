// An EventSource stand-in for component tests: the test opens the stream and emits named events by hand.

import { vi } from 'vitest';

export class FakeEventSource {
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

  /** The stream opens, as after a (re)connect. */
  open(): void {
    this.readyState = FakeEventSource.OPEN;
    this.onopen?.(new Event('open'));
  }

  emit(kind: string, data: unknown): void {
    const message = new MessageEvent(kind, { data: JSON.stringify(data) });
    for (const listener of this.listeners.get(kind) ?? []) listener(message);
  }
}

/** Stub the global EventSource; returns a getter for the stream the app opened (Vitest restores the global). */
export function stubEventSource(): () => FakeEventSource {
  FakeEventSource.instances = [];
  vi.stubGlobal('EventSource', FakeEventSource);
  return () => {
    const source = FakeEventSource.instances.at(-1);
    if (!source) throw new Error('No EventSource was opened');
    return source;
  };
}
