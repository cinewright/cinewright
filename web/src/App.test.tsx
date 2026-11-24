import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { ThemeSelect } from './App';

describe('ThemeSelect', () => {
  it('has the accessible name "Theme"', () => {
    render(<ThemeSelect />);

    // Exact match: a select without a name, or with any other name, fails here.
    const select = screen.getByRole('combobox', { name: 'Theme' });
    expect(select.tagName).toBe('SELECT');
    // The select sits inside its own label, where browsers may append the selected option to the
    // name ("Theme System"). The explicit aria-label pins the name; jsdom's name computation already
    // returns "Theme" without it, so check the attribute too.
    expect(select.getAttribute('aria-label')).toBe('Theme');
  });
});
