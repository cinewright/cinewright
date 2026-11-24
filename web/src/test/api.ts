// A stand-in for the core in component tests: `fetch` answers from a table of "METHOD /path"
// routes (query strings ignored) and records every request with its parsed JSON body.

import { vi } from 'vitest';

export interface ApiCall {
  method: string;
  path: string;
  body: unknown;
}

/** Each route returns the JSON body of a 200 answer, given the parsed request body. */
export type Routes = Record<string, (body: unknown) => unknown>;

function json(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

/** Stub the global `fetch`; unknown routes answer 404. Vitest restores the global after each test. */
export function stubApi(routes: Routes): ApiCall[] {
  const calls: ApiCall[] = [];
  vi.stubGlobal('fetch', async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
    const method = init.method ?? 'GET';
    const path = url.split('?')[0] ?? url;
    const body: unknown = typeof init.body === 'string' ? JSON.parse(init.body) : undefined;
    calls.push({ method, path, body });
    const route = routes[`${method} ${path}`];
    if (!route) return json(404, { type: 'about:blank', title: 'Not Found', status: 404 });
    return json(200, route(body));
  });
  return calls;
}
