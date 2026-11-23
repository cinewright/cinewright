// Theme choice: "system" follows prefers-color-scheme (tokens.css handles it), "dark" and "light"
// set data-theme on <html>. The choice is a per-browser convenience, so storage failures are ignored.

export type ThemeChoice = 'system' | 'dark' | 'light';

export const THEME_CHOICES: { value: ThemeChoice; label: string }[] = [
  { value: 'system', label: 'System' },
  { value: 'dark', label: 'Dark' },
  { value: 'light', label: 'Light' },
];

const KEY = 'cinewright.theme';

export function readTheme(): ThemeChoice {
  try {
    const value = window.localStorage.getItem(KEY);
    if (value === 'system' || value === 'dark' || value === 'light') return value;
  } catch {
    // storage blocked or unavailable
  }
  return 'system';
}

export function storeTheme(choice: ThemeChoice): void {
  try {
    window.localStorage.setItem(KEY, choice);
  } catch {
    // storage blocked or unavailable
  }
}

export function applyTheme(choice: ThemeChoice): void {
  const root = document.documentElement;
  if (choice === 'system') root.removeAttribute('data-theme');
  else root.setAttribute('data-theme', choice);
}
