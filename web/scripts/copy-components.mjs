// Copies the design system files the UI consumes from ../design:
// - components.css -> src/styles/components.css, without the Google Fonts @import
//   (Geist and Geist Mono ship with the app, see src/styles/fonts.css);
// - the logo SVGs -> public/logos/, unchanged.

import { copyFileSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const design = resolve(here, '../../design');

const GOOGLE_FONTS_IMPORT = /^@import\s+url\(['"]?https:\/\/fonts\.googleapis\.com\/[^)]*\)\s*;?[ \t]*\r?\n?/gm;

export function stripGoogleFonts(css) {
  return css.replace(GOOGLE_FONTS_IMPORT, '').replace(/^\s+/, '');
}

function main() {
  const source = readFileSync(resolve(design, 'components.css'), 'utf8');
  const css = stripGoogleFonts(source);
  if (/fonts\.(googleapis|gstatic)\.com/.test(css)) throw new Error('components.css still references Google Fonts');
  const header = '/* Copied by web/scripts/copy-components.mjs from design/components.css without the Google Fonts import. Do not edit. */\n\n';
  const target = resolve(here, '../src/styles/components.css');
  writeFileSync(target, header + css, 'utf8');
  console.log(`Wrote ${target}`);

  const logos = resolve(here, '../public/logos');
  mkdirSync(logos, { recursive: true });
  for (const name of readdirSync(resolve(design, 'logos')).filter((f) => f.endsWith('.svg'))) {
    copyFileSync(resolve(design, 'logos', name), resolve(logos, name));
  }
  console.log(`Copied logos to ${logos}`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) main();
