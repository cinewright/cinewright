// Status light: every state has a word and a shape (design/components/StatusLight).

import type { ReactNode } from 'react';

import type { CalibrationStatus } from '../api/client';

export type LightState = 'ok' | 'slow' | 'fail';

const SHAPES: Record<LightState, ReactNode> = {
  ok: (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <circle cx="8" cy="8" r="5.5" fill="currentColor" />
    </svg>
  ),
  slow: (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <circle cx="8" cy="8" r="5" fill="none" stroke="currentColor" strokeWidth="1.5" />
      <path d="M8 3a5 5 0 0 1 0 10z" fill="currentColor" />
    </svg>
  ),
  fail: (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <circle cx="8" cy="8" r="5.5" fill="none" stroke="currentColor" strokeWidth="1.5" />
      <path d="M6 6l4 4M10 6l-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  ),
};

export function FailShape() {
  return SHAPES.fail;
}

export function Light({ state, word, detail }: { state: LightState; word: string; detail?: string | null }) {
  return (
    <span className="cw-status" data-state={state}>
      {SHAPES[state]}
      {word}
      {detail ? <span className="cw-status__detail">· {detail}</span> : null}
    </span>
  );
}

/** A state in progress that needs nobody's attention: half disc in ink-muted, no signal colour. */
export function PendingLight({ word }: { word: string }) {
  return (
    <span className="cw-status cw-muted">
      {SHAPES.slow}
      {word}
    </span>
  );
}

const CALIBRATION: Record<CalibrationStatus, { state: LightState; word: string }> = {
  runs: { state: 'ok', word: 'Runs' },
  slow: { state: 'slow', word: 'Runs slowly' },
  wont_run: { state: 'fail', word: "Won't run" },
};

/** Calibration result; without a measurement only "Not calibrated" in ink-muted. */
export function StatusLight({ status, detail }: { status: CalibrationStatus | null | undefined; detail?: string | null }) {
  if (!status) return <span className="cw-status cw-muted">Not calibrated</span>;
  const { state, word } = CALIBRATION[status];
  return <Light state={state} word={word} detail={detail} />;
}
