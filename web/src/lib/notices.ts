// The WanGP Community License 2.0 requires integrations to disclose the use of WanGP in UI and docs.

export const WANGP_NOTICE =
  'Cinewright uses WanGP (Wan2GP) by deepbeepmeep as its first generation engine. WanGP is source-available under the WanGP Community License 2.0 and is downloaded from its official repository during setup; it is not part of Cinewright.';

export const WANGP_LICENSE_URL = 'https://github.com/deepbeepmeep/Wan2GP/blob/main/LICENSE.txt';

export const CINEWRIGHT_LICENSE = 'Cinewright is free software under the MIT licence.';

/** Third-party parts bundled into the web UI (the core lists its own through /licenses). */
export const WEB_UI_NOTICES: { component: string; license: string; url: string }[] = [
  { component: 'React', license: 'MIT', url: 'https://github.com/facebook/react/blob/main/LICENSE' },
  { component: 'Lucide icons', license: 'ISC', url: 'https://github.com/lucide-icons/lucide/blob/main/LICENSE' },
  { component: 'Geist and Geist Mono fonts', license: 'SIL OFL 1.1', url: 'https://github.com/vercel/geist-font/blob/main/OFL.txt' },
];
