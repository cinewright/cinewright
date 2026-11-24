/// <reference types="vitest/config" />

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
  // Component tests only; scripts/*.test.mjs are node:test files that `npm test` runs separately.
  test: {
    environment: 'jsdom',
    include: ['src/**/*.test.{ts,tsx}'],
    setupFiles: ['src/test/setup.ts'],
    unstubGlobals: true,
  },
});
