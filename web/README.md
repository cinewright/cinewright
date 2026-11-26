# Cinewright web UI

React 18 + TypeScript + Vite. `npm run build` writes the UI to `../src/cinewright/web_dist/`, which the core serves at `/`. Routing uses the URL hash (`#/queue`), so the server needs no history fallback. `#/library/<sha256>` opens the Library with that asset selected.

| Script | Does |
| --- | --- |
| `npm run dev` | Dev server; `/api` and `/mcp` are proxied to the core on `127.0.0.1:8737` |
| `npm run build` | `prebuild` (tokens, components), type check, production build |
| `npm run typecheck` | `tsc --noEmit` |
| `npm test` | Tests of the build scripts (`node --test`), then the component tests (`vitest run`) |
| `npm run gen:tokens` | `../design/tokens.json` to `src/styles/tokens.css` |
| `npm run copy:components` | `../design/components.css` (without the Google Fonts import) to `src/styles/components.css`, logos to `public/logos/` |
| `npm run gen:api` | `../openapi.json` to `src/api/schema.d.ts` |

Generated files are committed: `src/styles/tokens.css`, `src/styles/components.css`, `src/api/schema.d.ts` and `public/logos/`. Change the design system or the API contract, then run the matching script.

Component tests sit next to the code they test (`src/**/*.test.tsx`). They run in jsdom with Vitest and Testing Library and import the test APIs from `vitest`. The type check covers them like the rest of `src`. `src/test/` holds the setup, `stubApi`, which stubs `fetch` with fixed answers per route and records each request with its headers and body, `stubEventSource`, an event stream the test opens and feeds by hand, and `stubClipboard`, a `navigator.clipboard` (which jsdom lacks) that records each copied text or fails the way browsers do: it refuses the write, throws, has no `writeText`, or is missing.

Fonts (Geist, Geist Mono) come from the `@fontsource-variable` packages and are registered in `src/styles/fonts.css` under the names the tokens use; the UI makes no third-party requests.
