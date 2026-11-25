// A model's parameter limits as the Generate view uses them: the frame grid (`min_frames + n * frames_step`,
// at most `max_frames`), as the router checks it (core/router.py, _limit_problem).

import type { Capability, Limits } from '../api/client';

/** Capabilities that make one still image: no frames, no fps (IMAGE_CAPABILITIES in core/router.py). */
export const IMAGE_CAPABILITIES: ReadonlySet<Capability> = new Set<Capability>(['t2i', 'i2i']);

/** Capabilities whose request takes one input image: the start frame (i2v) or the image to edit (i2i). */
export const IMAGE_INPUT_CAPABILITIES: ReadonlySet<Capability> = new Set<Capability>(['i2v', 'i2i']);

export interface FrameGrid {
  /** Fewest frames: `min_frames`, or 1 for a model without a grid. */
  min: number;
  /** Distance between valid counts: `frames_step`, or 1. */
  step: number;
  /** Most frames on the grid, or null when the model sets no maximum. */
  max: number | null;
}

export function frameGrid(limits: Limits): FrameGrid {
  const min = limits.min_frames ?? 1;
  const step = limits.frames_step ?? 1;
  let max: number | null = null;
  // The largest count on the grid that does not exceed max_frames (the registry keeps max_frames on the grid).
  if (limits.max_frames != null) max = limits.max_frames >= min ? limits.max_frames - ((limits.max_frames - min) % step) : min;
  return { min, step, max };
}

export function onGrid(frames: number, grid: FrameGrid): boolean {
  return Number.isSafeInteger(frames) && frames >= grid.min && (grid.max === null || frames <= grid.max) && (frames - grid.min) % grid.step === 0;
}

/** The valid count nearest to `frames` (half way rounds up), within the grid's range. */
export function snapFrames(frames: number, grid: FrameGrid): number {
  const steps = Math.max(0, Math.round((frames - grid.min) / grid.step));
  const snapped = grid.min + steps * grid.step;
  return grid.max !== null && snapped > grid.max ? grid.max : snapped;
}

/** The grid in words that read after "Frames must be": "17 + a multiple of 8, up to 737", "1 to 81", "1 or more". */
export function describeGrid(grid: FrameGrid): string {
  if (grid.step === 1) return grid.max === null ? `${grid.min} or more` : `${grid.min} to ${grid.max}`;
  const base = `${grid.min} + a multiple of ${grid.step}`;
  return grid.max === null ? base : `${base}, up to ${grid.max}`;
}

/**
 * A whole number of frames from the field's text: null when empty (model default), undefined when not a whole number.
 * 0 is a whole number too; the grid refuses it, and leaving the field snaps it to the grid's minimum.
 */
export function parseFrames(text: string): number | null | undefined {
  const trimmed = text.trim();
  if (trimmed === '') return null;
  if (!/^\d+$/.test(trimmed)) return undefined;
  const value = Number(trimmed);
  return Number.isSafeInteger(value) ? value : undefined;
}
