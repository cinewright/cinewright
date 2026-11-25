// A stand-in for the core in component tests: `fetch` answers from a table of "METHOD /path"
// routes (query strings ignored for matching) and records every request with its query and parsed JSON body.

import { vi } from 'vitest';

export interface ApiCall {
  method: string;
  path: string;
  /** Query parameters of the request, e.g. `{ quant: 'int8' }`. */
  query: Record<string, string>;
  body: unknown;
}

/** Return this from a route to answer with an RFC 9457 problem instead of 200. */
export class ProblemAnswer {
  constructor(
    readonly status: number,
    readonly title: string,
    readonly detail: string,
  ) {}
}

/** Each route returns the JSON body of a 200 answer (or a ProblemAnswer), given the parsed request body. */
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
    const [path = url, search = ''] = url.split('?');
    const query = Object.fromEntries(new URLSearchParams(search));
    const body: unknown = typeof init.body === 'string' ? JSON.parse(init.body) : undefined;
    calls.push({ method, path, query, body });
    const route = routes[`${method} ${path}`];
    if (!route) return json(404, { type: 'about:blank', title: 'Not Found', status: 404 });
    const answer = route(body);
    if (answer instanceof ProblemAnswer) {
      return json(answer.status, { type: 'about:blank', title: answer.title, status: answer.status, detail: answer.detail });
    }
    return json(200, answer);
  });
  return calls;
}
