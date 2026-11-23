// Cast, Recipes and Presets arrive with phase 2; until then each view is an empty state.

import { EmptyState } from '../components/Feedback';

const COPY = {
  cast: {
    title: 'Cast',
    display: 'Cast arrives in phase 2',
    body: 'Cast entries keep a character consistent across jobs. Entries with real faces are Biometric and run only on hosts you approve.',
  },
  recipes: {
    title: 'Recipes',
    display: 'Recipes arrive in phase 2',
    body: 'A recipe is a versioned chain of generation steps that runs as one job and is recorded in each asset’s provenance.',
  },
  presets: {
    title: 'Presets',
    display: 'Presets arrive in phase 2',
    body: 'Presets are camera moves and styles built on recipes. They will appear as tiles in Generate.',
  },
} as const;

export function PhaseTwoView({ kind }: { kind: keyof typeof COPY }) {
  const copy = COPY[kind];
  return (
    <div className="view">
      <header className="view__head">
        <h1 className="view__title">{copy.title}</h1>
      </header>
      <EmptyState title={copy.display} display>
        <p>{copy.body}</p>
      </EmptyState>
    </div>
  );
}
