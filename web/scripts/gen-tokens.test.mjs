import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { TOKENS_CSS, generateTokensCss, loadTokens } from './gen-tokens.mjs';

const tokens = loadTokens();
const css = generateTokensCss(tokens);

/** Body of the first rule whose selector line is exactly `selector {`. */
function block(text, selector) {
  const start = text.indexOf(`${selector} {\n`);
  assert.notEqual(start, -1, `missing block ${selector}`);
  const open = start + selector.length + 2;
  let depth = 1;
  for (let i = open; i < text.length; i += 1) {
    if (text[i] === '{') depth += 1;
    if (text[i] === '}') depth -= 1;
    if (depth === 0) return text.slice(open, i);
  }
  throw new Error(`unterminated block ${selector}`);
}

function declared(body) {
  const map = new Map();
  for (const match of body.matchAll(/--([a-z0-9-]+):\s*([^;]+);/g)) map.set(match[1], match[2].trim());
  return map;
}

const root = declared(block(css, ':root'));
const light = declared(block(css, ':root[data-theme="light"]'));
const lightMedia = declared(block(block(css, '@media (prefers-color-scheme: light)'), '  :root:not([data-theme="dark"])'));

const alias = /^\{(.+)\}$/;
function resolve(themedTokens, name, theme, seen = []) {
  assert.ok(!seen.includes(name), `alias cycle at ${name}`);
  const value = themedTokens.get(name);
  const raw = typeof value === 'object' ? value[theme] : value;
  const match = alias.exec(raw);
  return match ? resolve(themedTokens, match[1], theme, [...seen, name]) : raw;
}

const themedGroups = [...tokens.color.tokens, ...tokens.shadow.tokens];
const all = new Map([...themedGroups, ...tokens.spacing.tokens, ...tokens.radius.tokens].map((t) => [t.name, t.value]));

test('dark is the default theme on :root, light applies by attribute and by media query', () => {
  assert.match(block(css, ':root'), /color-scheme: dark;/);
  assert.match(block(css, ':root[data-theme="light"]'), /color-scheme: light;/);
  assert.match(css, /@media \(prefers-color-scheme: light\) \{\n {2}:root:not\(\[data-theme="dark"\]\) \{/);
});

test('every colour and shadow token is emitted for both themes with its resolved value', () => {
  for (const token of themedGroups) {
    assert.equal(root.get(token.name), resolve(all, token.name, 'dark'), `dark ${token.name}`);
    assert.equal(light.get(token.name), resolve(all, token.name, 'light'), `light ${token.name}`);
    assert.equal(lightMedia.get(token.name), resolve(all, token.name, 'light'), `media light ${token.name}`);
  }
  assert.equal(light.size, lightMedia.size);
});

test('aliases are resolved per theme and none survive', () => {
  assert.ok(!css.includes('{accent}') && !/:\s*\{[a-z0-9-]+\}/.test(css), 'unresolved alias in output');
  const accent = new Map(tokens.color.tokens.map((t) => [t.name, t.value]));
  assert.equal(light.get('accent-mark'), accent.get('accent').light);
  assert.equal(root.get('status-slow'), accent.get('safelight').dark);
  assert.equal(light.get('status-slow'), accent.get('safelight').light);
});

test('spacing, radius and font families are emitted unchanged', () => {
  for (const token of [...tokens.spacing.tokens, ...tokens.radius.tokens]) {
    assert.equal(root.get(token.name), token.value, token.name);
  }
  for (const [family, stack] of Object.entries(tokens.type.families)) {
    assert.equal(root.get(`font-${family}`), stack, `font-${family}`);
  }
});

test('every text style becomes a variable and a class', () => {
  for (const group of tokens.type.groups) {
    for (const style of group.styles) {
      assert.equal(root.get(`text-${style.name}`), `${style.fontWeight} ${style.fontSize}/${style.lineHeight} var(--font-${group.family})`);
      const cls = block(css, `.text-${style.name}`);
      assert.match(cls, new RegExp(`font: var\\(--text-${style.name}\\);`));
      if (style.letterSpacing) assert.equal(root.get(`text-${style.name}-letter-spacing`), style.letterSpacing);
      if (group.family === 'mono') assert.match(cls, /tabular-nums/);
    }
  }
});

test('the committed tokens.css matches the generator output', () => {
  assert.equal(readFileSync(TOKENS_CSS, 'utf8'), css);
});
