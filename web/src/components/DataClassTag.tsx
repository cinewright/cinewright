// Data class tag with icon and word (design/components/DataClassTag).

import type { ReactNode } from 'react';

import type { DataClass } from '../api/client';

export const DATA_CLASSES: Record<DataClass, { label: string; hint: string; icon: ReactNode }> = {
  public: {
    label: 'Public',
    hint: 'All workers allowed.',
    icon: (
      <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
        <circle cx="8" cy="8" r="6" />
        <path d="M2 8h12M8 2c2 2 2 10 0 12M8 2c-2 2-2 10 0 12" />
      </svg>
    ),
  },
  private: {
    label: 'Private',
    hint: 'Local and own remote hosts.',
    icon: (
      <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" aria-hidden="true">
        <rect x="3" y="7" width="10" height="7" rx="1.5" />
        <path d="M5.5 7V5a2.5 2.5 0 0 1 5 0v2" />
      </svg>
    ),
  },
  biometric: {
    label: 'Biometric',
    hint: 'Local only, approval per host.',
    icon: (
      <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" aria-hidden="true">
        <path d="M2 5V3.5A1.5 1.5 0 0 1 3.5 2H5M11 2h1.5A1.5 1.5 0 0 1 14 3.5V5M14 11v1.5a1.5 1.5 0 0 1-1.5 1.5H11M5 14H3.5A1.5 1.5 0 0 1 2 12.5V11" />
        <circle cx="8" cy="7" r="2" />
        <path d="M5 12c.6-1.4 1.7-2 3-2s2.4.6 3 2" />
      </svg>
    ),
  },
};

export function DataClassTag({ value }: { value: DataClass }) {
  const { label, icon } = DATA_CLASSES[value];
  return (
    <span className="cw-tag" data-class={value}>
      {icon}
      {label}
    </span>
  );
}

export function CapabilityChip({ value }: { value: string }) {
  return <span className="cw-cap">{value}</span>;
}
