import type { Connect } from 'vite';

export function originAction(origin: string | undefined, host: string | undefined): 'none' | 'rewrite' | 'deny';
export function originGuard(prefixes: string[], coreOrigin: string): Connect.NextHandleFunction;
