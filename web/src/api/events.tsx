// One EventSource on /api/v1/events for the whole app. The core sends named events
// (`event: state|progress|preview|log`) whose `data:` line is a JobEvent; the browser resends
// Last-Event-ID on reconnect, so events can repeat and listeners must be idempotent.
// System events (`event: engine|model`) carry a SystemEvent with the full new state; they have no id
// and are never replayed, so views load the current state over the API, follow them, and load the
// state again whenever the stream (re)opens (`useEventsConnected` turning true).

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';

import { EVENTS_URL, type EventKind, type JobEvent, type SystemEvent, type SystemEventKind } from './client';

type Listener = (event: JobEvent) => void;
type SystemListener = (event: SystemEvent) => void;

const KINDS = { state: true, progress: true, preview: true, log: true } satisfies Record<EventKind, true>;
const SYSTEM_KINDS = { engine: true, model: true } satisfies Record<SystemEventKind, true>;
const RECONNECT_MS = 5000;

interface EventsContextValue {
  subscribe: (listener: Listener) => () => void;
  subscribeSystem: (listener: SystemListener) => () => void;
  connected: boolean;
  /** True once the stream has been open at least once, so a drop reads as "Reconnecting". */
  everConnected: boolean;
}

const EventsContext = createContext<EventsContextValue | null>(null);

function parse(raw: unknown): JobEvent | null {
  if (typeof raw !== 'string') return null;
  try {
    const event = JSON.parse(raw) as Partial<JobEvent>;
    return typeof event.job_id === 'string' && typeof event.seq === 'number' && typeof event.kind === 'string' ? (event as JobEvent) : null;
  } catch {
    return null;
  }
}

function parseSystem(raw: unknown): SystemEvent | null {
  if (typeof raw !== 'string') return null;
  try {
    const event = JSON.parse(raw) as Partial<SystemEvent>;
    return typeof event.kind === 'string' && event.kind in SYSTEM_KINDS ? (event as SystemEvent) : null;
  } catch {
    return null;
  }
}

export function EventsProvider({ children }: { children: ReactNode }) {
  const listeners = useRef(new Set<Listener>());
  const systemListeners = useRef(new Set<SystemListener>());
  const [connected, setConnected] = useState(false);
  const [everConnected, setEverConnected] = useState(false);

  useEffect(() => {
    let source: EventSource | null = null;
    let retry: number | undefined;

    const dispatch = (message: MessageEvent) => {
      const event = parse(message.data);
      if (event) for (const listener of listeners.current) listener(event);
    };
    const dispatchSystem = (message: MessageEvent) => {
      const event = parseSystem(message.data);
      if (event) for (const listener of systemListeners.current) listener(event);
    };

    const open = () => {
      source = new EventSource(EVENTS_URL);
      for (const kind of Object.keys(KINDS)) source.addEventListener(kind, dispatch);
      for (const kind of Object.keys(SYSTEM_KINDS)) source.addEventListener(kind, dispatchSystem);
      source.onmessage = dispatch;
      source.onopen = () => {
        setConnected(true);
        setEverConnected(true);
      };
      source.onerror = () => {
        setConnected(false);
        // The browser retries on its own unless the stream was refused; then start over later.
        if (source?.readyState === EventSource.CLOSED) {
          source.close();
          retry = window.setTimeout(open, RECONNECT_MS);
        }
      };
    };

    open();
    return () => {
      window.clearTimeout(retry);
      source?.close();
    };
  }, []);

  const subscribe = useCallback((listener: Listener) => {
    listeners.current.add(listener);
    return () => {
      listeners.current.delete(listener);
    };
  }, []);
  const subscribeSystem = useCallback((listener: SystemListener) => {
    systemListeners.current.add(listener);
    return () => {
      systemListeners.current.delete(listener);
    };
  }, []);
  const value = useMemo(
    () => ({ subscribe, subscribeSystem, connected, everConnected }),
    [subscribe, subscribeSystem, connected, everConnected],
  );

  return <EventsContext.Provider value={value}>{children}</EventsContext.Provider>;
}

function useEventsContext(): EventsContextValue {
  const context = useContext(EventsContext);
  if (!context) throw new Error('useJobEvents needs an EventsProvider');
  return context;
}

/** Call `listener` for every job event; the latest listener is used without resubscribing. */
export function useJobEvents(listener: Listener): void {
  const { subscribe } = useEventsContext();
  const latest = useRef(listener);
  latest.current = listener;
  useEffect(() => subscribe((event) => latest.current(event)), [subscribe]);
}

/** Call `listener` for every system event (engine provisioning); the latest listener is used without resubscribing. */
export function useSystemEvents(listener: SystemListener): void {
  const { subscribeSystem } = useEventsContext();
  const latest = useRef(listener);
  latest.current = listener;
  useEffect(() => subscribeSystem((event) => latest.current(event)), [subscribeSystem]);
}

/** Whether the event stream is currently open. */
export function useEventsConnected(): boolean {
  return useEventsContext().connected;
}

/** Whether the stream has ever been open since the app loaded. */
export function useEventsEverConnected(): boolean {
  return useEventsContext().everConnected;
}
