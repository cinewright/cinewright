import react from '@vitejs/plugin-react';
import { defineConfig, type ProxyOptions } from 'vite';

import { originGuard } from './scripts/dev-origin.mjs';

const CORE = 'http://127.0.0.1:8737';

// Origin policy lives in scripts/dev-origin.mjs: own origin -> rewritten for the core, foreign -> 403.
const toCore: ProxyOptions = { target: CORE, changeOrigin: true };

export default defineConfig({
  plugins: [
    react(),
    {
      name: 'cinewright-origin-guard',
      configureServer(server) {
        server.middlewares.use(originGuard(['/api', '/mcp'], CORE));
      },
    },
  ],
  base: '/',
  build: {
    outDir: '../src/cinewright/web_dist',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/api': toCore,
      '/mcp': toCore,
    },
  },
});
