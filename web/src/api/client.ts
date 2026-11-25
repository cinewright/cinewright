// Typed client for the internal API. Paths are checked against the generated contract
// (schema.d.ts from openapi.json); errors arrive as RFC 9457 problems and surface as ApiError.

import type { components, paths } from './schema';

export type Schemas = components['schemas'];
export type ApiPath = keyof paths;

export type Asset = Schemas['Asset'];
export type CalibrationResult = Schemas['CalibrationResult'];
export type CalibrationStatus = Schemas['CalibrationStatus'];
export type Capability = Schemas['Capability'];
export type DataClass = Schemas['DataClass'];
export type EngineInfo = Schemas['EngineInfo'];
export type Estimate = Schemas['Estimate'];
export type HardwareProfile = Schemas['HardwareProfile'];
export type Host = Schemas['Host'];
export type Job = Schemas['Job'];
export type JobRequest = Schemas['JobRequest'];
export type JobState = Schemas['JobState'];
export type ModelSummary = Schemas['ModelSummary'];
export type Priority = Schemas['Priority'];
export type Provenance = Schemas['Provenance'];
export type Settings = Schemas['Settings'];
export type SettingsPatch = Schemas['SettingsPatch'];
export type WorkerHealth = Schemas['WorkerHealth'];

// The SSE stream is not modelled in openapi.json; these mirror EventKind, JobEvent, SystemEventKind,
// ModelInstall and SystemEvent in api/schemas.py.
export type EventKind = 'state' | 'progress' | 'preview' | 'log';
export interface JobEvent {
  job_id: string;
  seq: number;
  kind: EventKind;
  data: Record<string, unknown>;
  created_at: string;
}
/** Non-job events: sent without an id and never replayed. */
export type SystemEventKind = 'engine' | 'model';
/** A model quantization's new install state: download progress, the end of an install, or a removal. */
export interface ModelInstall {
  model_id: string;
  quant: string;
  install_state: Schemas['InstallState'];
  /** While `downloading`: share of the bytes this install downloads. */
  progress?: Schemas['Progress'] | null;
  downloaded_bytes?: number | null;
  total_bytes?: number | null;
  /** Why the install failed (`failed`). */
  error?: Schemas['SetupError'] | null;
}
export interface SystemEvent {
  kind: SystemEventKind;
  /** For `engine`: the engine as GET /system reports it. */
  engine?: EngineInfo | null;
  /** For `model`: the quantization's new install state. */
  model?: ModelInstall | null;
  created_at: string;
}

type Problem = Schemas['Problem'] & Record<string, unknown>;

export class ApiError extends Error {
  readonly status: number;
  readonly type: string;
  readonly title: string;
  readonly detail: string | null;
  /** The full problem body, including extensions such as `reasons` or `errors`. */
  readonly problem: Problem | null;

  constructor(status: number, title: string, detail: string | null, type = 'about:blank', problem: Problem | null = null) {
    super(detail ?? title);
    this.name = 'ApiError';
    this.status = status;
    this.title = title;
    this.detail = detail;
    this.type = type;
    this.problem = problem;
  }

  /** Router findings attached to an infeasible problem (409). */
  get reasons(): string[] {
    const reasons = this.problem?.reasons;
    return Array.isArray(reasons) ? reasons.map(String) : [];
  }

  get infeasible(): boolean {
    return this.type === 'urn:cinewright:problem:infeasible';
  }
}

export const NOT_RUNNING = 'Cinewright is not responding. Requests fail until the core runs again.';

/** Human-readable message for any thrown value. */
export function errorMessage(error: unknown): string {
  if (error instanceof ApiError) return error.detail ?? error.title;
  if (error instanceof Error) return error.message;
  return String(error);
}

export function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError';
}

/** Substitute `{name}` segments of a contract path. */
export function apiUrl(path: ApiPath, params: Record<string, string> = {}, query: Record<string, string | undefined> = {}): string {
  const filled = path.replace(/\{(\w+)\}/g, (_, name: string) => {
    const value = params[name];
    if (value === undefined) throw new Error(`Missing path parameter ${name} for ${path}`);
    return encodeURIComponent(value);
  });
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) if (value !== undefined) search.set(key, value);
  const qs = search.toString();
  return qs ? `${filled}?${qs}` : filled;
}

interface RequestOptions {
  params?: Record<string, string>;
  query?: Record<string, string | undefined>;
  body?: unknown;
  signal?: AbortSignal;
}

async function toApiError(response: Response): Promise<ApiError> {
  const type = response.headers.get('content-type') ?? '';
  if (type.includes('json')) {
    try {
      const body = (await response.json()) as Problem;
      return new ApiError(response.status, body.title ?? response.statusText, body.detail ?? null, body.type ?? 'about:blank', body);
    } catch {
      // fall through to the status line
    }
  }
  return new ApiError(response.status, response.statusText || `HTTP ${response.status}`, null);
}

async function request<T>(method: string, path: ApiPath, options: RequestOptions = {}): Promise<T> {
  const headers: Record<string, string> = { Accept: 'application/json' };
  let body: string | undefined;
  if (options.body !== undefined) {
    headers['Content-Type'] = 'application/json';
    body = JSON.stringify(options.body);
  }
  let response: Response;
  try {
    response = await fetch(apiUrl(path, options.params, options.query), { method, headers, body, signal: options.signal });
  } catch (error) {
    if (isAbort(error)) throw error;
    throw new ApiError(0, 'Not responding', NOT_RUNNING);
  }
  if (!response.ok) throw await toApiError(response);
  if (response.status === 204 || response.headers.get('content-length') === '0') return undefined as T;
  return (await response.json()) as T;
}

const PAGE = '50';

export const api = {
  system: (signal?: AbortSignal) => request<Schemas['SystemInfo']>('GET', '/api/v1/system', { signal }),
  provisionEngine: (engineId: string) =>
    request<EngineInfo>('POST', '/api/v1/engines/{engine_id}/provision', { params: { engine_id: engineId } }),
  licenses: (signal?: AbortSignal) => request<Schemas['LicenseNotices']>('GET', '/api/v1/licenses', { signal }),
  settings: (signal?: AbortSignal) => request<Settings>('GET', '/api/v1/settings', { signal }),
  updateSettings: (patch: SettingsPatch) => request<Settings>('PATCH', '/api/v1/settings', { body: patch }),

  hardware: (signal?: AbortSignal) => request<HardwareProfile>('GET', '/api/v1/hardware', { signal }),
  detectHardware: () => request<HardwareProfile>('POST', '/api/v1/hardware/detect'),

  capabilities: (signal?: AbortSignal) => request<Schemas['CapabilityList']>('GET', '/api/v1/capabilities', { signal }),
  calibration: (signal?: AbortSignal) => request<Schemas['CalibrationReport']>('GET', '/api/v1/calibration', { signal }),
  models: (signal?: AbortSignal) => request<Schemas['ModelList']>('GET', '/api/v1/models', { signal }),

  estimate: (job: JobRequest, signal?: AbortSignal) => request<Estimate>('POST', '/api/v1/estimate', { body: job, signal }),
  submitJob: (job: JobRequest) => request<Job>('POST', '/api/v1/jobs', { body: job }),
  jobs: (cursor?: string, signal?: AbortSignal) =>
    request<Schemas['JobList']>('GET', '/api/v1/jobs', { query: { cursor, limit: PAGE }, signal }),
  job: (jobId: string, signal?: AbortSignal) => request<Job>('GET', '/api/v1/jobs/{job_id}', { params: { job_id: jobId }, signal }),
  cancelJob: (jobId: string) => request<Job>('POST', '/api/v1/jobs/{job_id}/cancel', { params: { job_id: jobId } }),

  assets: (cursor?: string, signal?: AbortSignal) =>
    request<Schemas['AssetList']>('GET', '/api/v1/assets', { query: { cursor, limit: PAGE }, signal }),

  hosts: (signal?: AbortSignal) => request<Schemas['HostList']>('GET', '/api/v1/hosts', { signal }),
};

export const assetContentUrl = (sha256: string) => apiUrl('/api/v1/assets/{sha256}/content', { sha256 });
export const assetProvenanceUrl = (sha256: string) => apiUrl('/api/v1/assets/{sha256}/provenance', { sha256 });
export const EVENTS_URL = apiUrl('/api/v1/events');
