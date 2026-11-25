// A stand-in for the core in component tests: `fetch` answers from a table of "METHOD /path"
// routes (query strings ignored for matching) and records every request with its query, headers and body.

import { vi } from 'vitest';

export interface ApiCall {
  method: string;
  path: string;
  /** Query parameters of the request, e.g. `{ quant: 'int8' }`. */
  query: Record<string, string>;
  /** Request headers as sent, e.g. `{ Accept: 'application/json', 'Content-Type': 'image/png' }`. */
  headers: Record<string, string>;
  /** The parsed JSON body, or undefined when the body is not a string. */
  body: unknown;
  /** The body exactly as passed to `fetch` (a File or Blob for uploads). */
  rawBody: unknown;
}

/** Method, path, query and parsed body of a call, for exact comparisons that leave headers out. */
export function callSummary({ method, path, query, body }: ApiCall): Pick<ApiCall, 'method' | 'path' | 'query' | 'body'> {
  return { method, path, query, body };
}

/** Return this from a route to answer with an RFC 9457 problem instead of 200. */
export class ProblemAnswer {
  constructor(
    readonly status: number,
    readonly title: string,
    readonly detail: string,
  ) {}
}

/** Return this from a route to answer with a status and no JSON body (as a proxy might). */
export class StatusAnswer {
  constructor(readonly status: number) {}
}

/** Each route returns the JSON body of a 200 answer (or a ProblemAnswer or StatusAnswer), given the parsed request body. */
export type Routes = Record<string, (body: unknown) => unknown>;

/** Headers with the names as the client spelled them (a plain object), or lower-cased (a Headers instance or pairs). */
function headersOf(init: HeadersInit | undefined): Record<string, string> {
  if (init === undefined) return {};
  if (init instanceof Headers || Array.isArray(init)) return Object.fromEntries(new Headers(init).entries());
  return { ...init };
}

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
    const headers = headersOf(init.headers);
    const body: unknown = typeof init.body === 'string' ? JSON.parse(init.body) : undefined;
    calls.push({ method, path, query, headers, body, rawBody: init.body });
    const route = routes[`${method} ${path}`];
    if (!route) return json(404, { type: 'about:blank', title: 'Not Found', status: 404 });
    const answer = route(body);
    if (answer instanceof ProblemAnswer) {
      return json(answer.status, { type: 'about:blank', title: answer.title, status: answer.status, detail: answer.detail });
    }
    if (answer instanceof StatusAnswer) return new Response(null, { status: answer.status });
    return json(200, answer);
  });
  return calls;
}
