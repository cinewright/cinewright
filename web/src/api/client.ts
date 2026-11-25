// Typed client for the internal API. Paths are checked against the generated contract
// (schema.d.ts from openapi.json); errors arrive as RFC 9457 problems and surface as ApiError.

import type { components, paths } from './schema';

export type Schemas = components['schemas'];
export type ApiPath = keyof paths;

export type Asset = Schemas['Asset'];
export type CalibrationResult = Schemas['CalibrationResult'];
export type CalibrationStatus = Schemas['CalibrationStatus'];
export type Capability = Schemas['Capability'];
export type CapabilityEntry = Schemas['CapabilityEntry'];
export type DataClass = Schemas['DataClass'];
export type EngineInfo = Schemas['EngineInfo'];
export type Estimate = Schemas['Estimate'];
export type HardwareProfile = Schemas['HardwareProfile'];
export type Host = Schemas['Host'];
export type HostPatch = Schemas['HostPatch'];
export type InstallRefusal = Schemas['InstallRefusal'];
export type Job = Schemas['Job'];
export type JobError = Schemas['JobError'];
export type JobRequest = Schemas['JobRequest'];
export type JobState = Schemas['JobState'];
export type Limits = Schemas['Limits'];
export type ModelStoreUsage = Schemas['ModelStoreUsage'];
export type ModelSummary = Schemas['ModelSummary'];
export type Priority = Schemas['Priority'];
export type Progress = Schemas['Progress'];
export type Provenance = Schemas['Provenance'];
export type QuotaEviction = Schemas['QuotaEviction'];
export type Settings = Schemas['Settings'];
export type SettingsPatch = Schemas['SettingsPatch'];
export type WorkerHealth = Schemas['WorkerHealth'];

// The SSE payloads of GET /events, as openapi.json describes the text/event-stream response: a job event
// (`state|progress|preview|log`, with an id unless it is a preview) or a system event (`engine|model`, no id).
export type EventKind = Schemas['EventKind'];
export type JobEvent = Schemas['JobEvent'];
export type SystemEventKind = Schemas['SystemEventKind'];
export type ModelInstall = Schemas['ModelInstall'];
export type SystemEvent = Schemas['SystemEvent'];

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

  /** The stable `code` of a refusal, e.g. `eviction_needed` or `not_enough_space`, or null. */
  get code(): string | null {
    const code = this.problem?.code;
    return typeof code === 'string' ? code : null;
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

function isQuotaEviction(value: unknown): value is QuotaEviction {
  if (typeof value !== 'object' || value === null) return false;
  const { model_id, quant, name, size_bytes } = value as Record<string, unknown>;
  return typeof model_id === 'string' && typeof quant === 'string' && typeof name === 'string' && typeof size_bytes === 'number';
}

/**
 * The removals an install refused with `eviction_needed` (409) asks to confirm, or null for any other error. Send the
 * install again with these as `evict` to remove them first.
 */
export function evictionNeeded(error: unknown): { evictions: QuotaEviction[]; freesBytes: number } | null {
  if (!(error instanceof ApiError) || error.status !== 409 || error.code !== 'eviction_needed' || !error.problem) return null;
  // The contract's field names; the body is still checked, since it arrives unchecked.
  const { evictions, frees_bytes: freesBytes } = error.problem as InstallRefusal;
  if (!Array.isArray(evictions) || evictions.length === 0 || !evictions.every(isQuotaEviction)) return null;
  if (typeof freesBytes !== 'number') return null;
  return { evictions, freesBytes };
}

/** A query parameter: one value, a value per repetition (`?evict=a&evict=b`), or left out when undefined. */
export type QueryValue = string | readonly string[] | undefined;

/** Substitute `{name}` segments of a contract path. */
export function apiUrl(path: ApiPath, params: Record<string, string> = {}, query: Record<string, QueryValue> = {}): string {
  const filled = path.replace(/\{(\w+)\}/g, (_, name: string) => {
    const value = params[name];
    if (value === undefined) throw new Error(`Missing path parameter ${name} for ${path}`);
    return encodeURIComponent(value);
  });
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (typeof value === 'string') search.append(key, value);
    else for (const item of value ?? []) search.append(key, item);
  }
  const qs = search.toString();
  return qs ? `${filled}?${qs}` : filled;
}

interface RequestOptions {
  params?: Record<string, string>;
  query?: Record<string, QueryValue>;
  /** Sent as JSON. */
  body?: unknown;
  /** Sent as the raw body, with `mediaType` as its Content-Type (uploads; the core refuses multipart). */
  file?: { data: Blob; mediaType: string };
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
  let body: string | Blob | undefined;
  if (options.file !== undefined) {
    headers['Content-Type'] = options.file.mediaType;
    body = options.file.data;
  } else if (options.body !== undefined) {
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
  // A 202 without a body (installModel, updateHost) may come without a content-length.
  const text = await response.text();
  return (text ? JSON.parse(text) : undefined) as T;
}

/** Image types an input upload may have: the image types of the core's allow-list (UPLOAD_MEDIA_TYPES in api/deps.py). */
export const UPLOAD_IMAGE_TYPES: readonly string[] = ['image/png', 'image/jpeg', 'image/webp'];
const IMAGE_FORMATS = 'PNG, JPEG or WebP';

/** Why `file` is not uploaded as an image, or null. Browsers leave `File.type` empty for a type they do not know. */
export function imageTypeProblem(file: File): string | null {
  if (!file.type) return `${file.name} has no known image type, so it is not uploaded; choose a ${IMAGE_FORMATS} image.`;
  if (!UPLOAD_IMAGE_TYPES.includes(file.type)) return `${file.name} is ${file.type}, so it is not uploaded; choose a ${IMAGE_FORMATS} image.`;
  return null;
}

/** Readable refusals of an upload. The core itself answers 422 with a detail of its own (type, size, empty body). */
function uploadError(error: unknown): unknown {
  if (!(error instanceof ApiError)) return error;
  if (error.status === 413) return new ApiError(413, 'Too large', 'The image is too large for the core, so it was not uploaded.', error.type, error.problem);
  if (error.status === 415)
    return new ApiError(415, 'Unsupported media type', `The core refused the file type, so the image was not uploaded; choose a ${IMAGE_FORMATS} image.`, error.type, error.problem);
  return error;
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
  /**
   * Starts the install in the background (202); progress follows as `model` events. `evict` confirms the removals an
   * `eviction_needed` refusal listed (see `evictionNeeded`).
   */
  installModel: (modelId: string, quant: string, evict: readonly Pick<QuotaEviction, 'model_id' | 'quant'>[] = []) =>
    request<void>('POST', '/api/v1/models/{model_id}/install', {
      params: { model_id: modelId },
      query: { quant, evict: evict.map((item) => `${item.model_id}:${item.quant}`) },
    }),
  /** Removes one quantization, or every one of the model when `quant` is left out. */
  removeModel: (modelId: string, quant?: string) =>
    request<void>('DELETE', '/api/v1/models/{model_id}', { params: { model_id: modelId }, query: { quant } }),

  estimate: (job: JobRequest, signal?: AbortSignal) => request<Estimate>('POST', '/api/v1/estimate', { body: job, signal }),
  submitJob: (job: JobRequest) => request<Job>('POST', '/api/v1/jobs', { body: job }),
  jobs: (cursor?: string, signal?: AbortSignal) =>
    request<Schemas['JobList']>('GET', '/api/v1/jobs', { query: { cursor, limit: PAGE }, signal }),
  job: (jobId: string, signal?: AbortSignal) => request<Job>('GET', '/api/v1/jobs/{job_id}', { params: { job_id: jobId }, signal }),
  cancelJob: (jobId: string) => request<Job>('POST', '/api/v1/jobs/{job_id}/cancel', { params: { job_id: jobId } }),

  assets: (cursor?: string, signal?: AbortSignal) =>
    request<Schemas['AssetList']>('GET', '/api/v1/assets', { query: { cursor, limit: PAGE }, signal }),
  /** One asset; `sha256` must already be checked as 64 lowercase hex characters. */
  asset: (sha256: string, signal?: AbortSignal) => request<Asset>('GET', '/api/v1/assets/{sha256}', { params: { sha256 }, signal }),
  /** Uploads an input image as the raw body with its media type; a type outside UPLOAD_IMAGE_TYPES is refused before sending. */
  uploadImage: async (file: File, dataClass: DataClass): Promise<Asset> => {
    const problem = imageTypeProblem(file);
    if (problem) throw new Error(problem);
    try {
      return await request<Asset>('POST', '/api/v1/assets', { query: { data_class: dataClass }, file: { data: file, mediaType: file.type } });
    } catch (error) {
      throw uploadError(error);
    }
  },

  hosts: (signal?: AbortSignal) => request<Schemas['HostList']>('GET', '/api/v1/hosts', { signal }),
  /** A new text encoder variant is switched in the background (202); GET /hosts shows it once the switch is done. */
  updateHost: (hostId: string, patch: HostPatch) => request<void>('PATCH', '/api/v1/hosts/{host_id}', { params: { host_id: hostId }, body: patch }),
};

/** An asset id as the API writes it: 64 lowercase hex characters. Check data from links and events before building a path. */
export function isSha256(value: unknown): value is string {
  return typeof value === 'string' && /^[0-9a-f]{64}$/.test(value);
}

export const assetContentUrl = (sha256: string) => apiUrl('/api/v1/assets/{sha256}/content', { sha256 });
export const assetProvenanceUrl = (sha256: string) => apiUrl('/api/v1/assets/{sha256}/provenance', { sha256 });
export const EVENTS_URL = apiUrl('/api/v1/events');
