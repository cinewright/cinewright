import assert from 'node:assert/strict';
import { test } from 'node:test';

import { originAction, originGuard } from './dev-origin.mjs';

const CORE = 'http://127.0.0.1:8737';

function run(url, headers) {
  const req = { url, headers };
  const res = { statusCode: 200, setHeader() {}, end() { this.ended = true; } };
  let nexted = false;
  originGuard(['/api', '/mcp'], CORE)(req, res, () => { nexted = true; });
  return { req, res, nexted };
}

test('originAction', () => {
  assert.equal(originAction(undefined, 'localhost:5173'), 'none');
  assert.equal(originAction('http://localhost:5173', 'localhost:5173'), 'rewrite');
  assert.equal(originAction('http://evil.example', 'localhost:5173'), 'deny');
  assert.equal(originAction('http://localhost:5173', undefined), 'deny');
});

test('own origin is rewritten to the core origin', () => {
  const { req, nexted } = run('/api/v1/jobs', { host: 'localhost:5173', origin: 'http://localhost:5173' });
  assert.ok(nexted);
  assert.equal(req.headers.origin, CORE);
});

test('foreign origin gets 403 and is not proxied', () => {
  const { res, nexted } = run('/api/v1/jobs', { host: 'localhost:5173', origin: 'http://evil.example' });
  assert.equal(res.statusCode, 403);
  assert.equal(nexted, false);
});

test('other paths and requests without Origin pass untouched', () => {
  assert.ok(run('/assets/x.js', { host: 'localhost:5173', origin: 'http://evil.example' }).nexted);
  assert.ok(run('/mcp', { host: 'localhost:5173' }).nexted);
});
