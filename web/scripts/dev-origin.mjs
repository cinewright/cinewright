// Dev-proxy Origin policy: the core accepts only its own Origin, so the dev server presents its
// own pages as same-origin and refuses everything else instead of laundering foreign origins.

/** 'none' (no Origin header), 'rewrite' (the dev server's own origin) or 'deny' (anything else). */
export function originAction(origin, host) {
  if (origin === undefined || origin === '') return 'none';
  return host && origin === `http://${host}` ? 'rewrite' : 'deny';
}

/** Connect middleware for the proxied prefixes: rewrite own Origin to `coreOrigin`, answer 403 otherwise. */
export function originGuard(prefixes, coreOrigin) {
  return (req, res, next) => {
    const url = req.url ?? '';
    if (!prefixes.some((p) => url === p || url.startsWith(`${p}/`) || url.startsWith(`${p}?`))) return next();
    const action = originAction(req.headers.origin, req.headers.host);
    if (action === 'deny') {
      res.statusCode = 403;
      res.setHeader('content-type', 'text/plain; charset=utf-8');
      res.end('Foreign Origin refused by the dev proxy');
      return undefined;
    }
    if (action === 'rewrite') req.headers.origin = coreOrigin;
    return next();
  };
}
