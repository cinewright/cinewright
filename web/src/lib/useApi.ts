import { useCallback, useEffect, useState, type DependencyList } from 'react';

import { isAbort } from '../api/client';

export interface ApiState<T> {
  data: T | undefined;
  error: unknown;
  loading: boolean;
  reload: () => void;
  setData: (data: T) => void;
}

/** Load data on mount and when `deps` change; stale requests are aborted. */
export function useApi<T>(load: (signal: AbortSignal) => Promise<T>, deps: DependencyList = []): ApiState<T> {
  const [data, setData] = useState<T>();
  const [error, setError] = useState<unknown>();
  const [loading, setLoading] = useState(true);
  const [generation, setGeneration] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    load(controller.signal)
      .then((result) => {
        setData(result);
        setError(undefined);
      })
      .catch((err: unknown) => {
        if (!isAbort(err)) setError(err);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
    // `load` is a new closure on every render; the caller's deps decide when to load again.
  }, [...deps, generation]);

  const reload = useCallback(() => setGeneration((g) => g + 1), []);
  return { data, error, loading, reload, setData };
}
