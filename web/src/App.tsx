// App shell: header with logo, theme and About; left navigation; hash routing (#/queue), so the
// core needs no history fallback to serve the UI.

import {
  BookOpen,
  Boxes,
  Clapperboard,
  Gauge,
  Images,
  Info,
  LayoutGrid,
  ListOrdered,
  Server,
  Settings as SettingsIcon,
  Users,
  type LucideIcon,
} from 'lucide-react';
import { useEffect, useState, type ReactNode } from 'react';

import { EventsProvider } from './api/events';
import { AboutDialog } from './components/AboutDialog';
import { hashSegments } from './lib/route';
import { THEME_CHOICES, applyTheme, readTheme, storeTheme, type ThemeChoice } from './lib/theme';
import { GenerateView } from './views/GenerateView';
import { HostsView } from './views/HostsView';
import { LibraryView } from './views/LibraryView';
import { ModelsView } from './views/ModelsView';
import { PhaseTwoView } from './views/PhaseTwoView';
import { QueueView } from './views/QueueView';
import { SettingsView } from './views/SettingsView';
import { SetupView } from './views/SetupView';

interface Route {
  id: string;
  label: string;
  icon: LucideIcon;
  render: () => ReactNode;
}

const ROUTES: Route[] = [
  { id: 'setup', label: 'Setup', icon: Gauge, render: () => <SetupView /> },
  { id: 'generate', label: 'Generate', icon: Clapperboard, render: () => <GenerateView /> },
  { id: 'queue', label: 'Queue', icon: ListOrdered, render: () => <QueueView /> },
  { id: 'library', label: 'Library', icon: Images, render: () => <LibraryView /> },
  { id: 'cast', label: 'Cast', icon: Users, render: () => <PhaseTwoView kind="cast" /> },
  { id: 'recipes', label: 'Recipes', icon: BookOpen, render: () => <PhaseTwoView kind="recipes" /> },
  { id: 'presets', label: 'Presets', icon: LayoutGrid, render: () => <PhaseTwoView kind="presets" /> },
  { id: 'hosts', label: 'Hosts', icon: Server, render: () => <HostsView /> },
  { id: 'models', label: 'Models', icon: Boxes, render: () => <ModelsView /> },
  { id: 'settings', label: 'Settings', icon: SettingsIcon, render: () => <SettingsView /> },
];
const DEFAULT_ROUTE = ROUTES[0] as Route;

function routeFromHash(): Route {
  const id = hashSegments()[0];
  return ROUTES.find((route) => route.id === id) ?? DEFAULT_ROUTE;
}

function useRoute(): Route {
  const [route, setRoute] = useState(routeFromHash);
  useEffect(() => {
    const onChange = () => setRoute(routeFromHash());
    window.addEventListener('hashchange', onChange);
    return () => window.removeEventListener('hashchange', onChange);
  }, []);
  useEffect(() => {
    document.title = `${route.label} · Cinewright`;
  }, [route]);
  return route;
}

/** Exported for the component tests. */
export function ThemeSelect() {
  const [theme, setTheme] = useState<ThemeChoice>(readTheme);
  const change = (value: ThemeChoice) => {
    setTheme(value);
    storeTheme(value);
    applyTheme(value);
  };
  return (
    <label className="theme-select">
      <span className="text-caption cw-muted">Theme</span>
      <select className="cw-select" aria-label="Theme" value={theme} onChange={(e) => change(e.target.value as ThemeChoice)}>
        {THEME_CHOICES.map((choice) => (
          <option key={choice.value} value={choice.value}>
            {choice.label}
          </option>
        ))}
      </select>
    </label>
  );
}

export function App() {
  const route = useRoute();
  const [aboutOpen, setAboutOpen] = useState(false);

  return (
    <EventsProvider>
      <div className="shell">
        <header className="shell__header">
          <a className="shell__brand" href="#/setup">
            <img className="logo logo--dark" src="/logos/cinewright-logo-on-dark.svg" alt="Cinewright" height={24} />
            <img className="logo logo--light" src="/logos/cinewright-logo-on-light.svg" alt="Cinewright" height={24} />
          </a>
          <div className="shell__tools">
            <ThemeSelect />
            <button type="button" className="cw-btn cw-btn--ghost" onClick={() => setAboutOpen(true)}>
              <Info strokeWidth={1.5} aria-hidden="true" />
              About
            </button>
          </div>
        </header>
        <nav className="shell__nav" aria-label="Main">
          <ul className="nav">
            {ROUTES.map(({ id, label, icon: Icon }) => (
              <li key={id}>
                <a href={`#/${id}`} aria-current={route.id === id ? 'page' : undefined}>
                  <Icon strokeWidth={1.5} aria-hidden="true" />
                  {label}
                </a>
              </li>
            ))}
          </ul>
        </nav>
        <main className="shell__main" key={route.id}>
          {route.render()}
        </main>
      </div>
      <AboutDialog open={aboutOpen} onClose={() => setAboutOpen(false)} />
    </EventsProvider>
  );
}
