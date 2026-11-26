// An asset's media from the asset store, as the Library and the Job panel show it: an image, a clip (muted, never
// playing by itself), audio, or a file icon for any other type.

import { AudioLines, File } from 'lucide-react';

import { assetContentUrl, type Asset } from '../api/client';

/** `controls` adds the player controls to clips and audio; without them audio shows as an icon. */
export function AssetMedia({ asset, controls }: { asset: Asset; controls: boolean }) {
  const url = assetContentUrl(asset.sha256);
  const label = `${asset.media_type} ${asset.sha256.slice(0, 8)}`;
  if (asset.media_type.startsWith('image/')) return <img src={url} alt={label} loading="lazy" />;
  if (asset.media_type.startsWith('video/'))
    return <video src={url} controls={controls} muted playsInline preload="metadata" aria-label={label} />;
  if (asset.media_type.startsWith('audio/'))
    return controls ? <audio src={url} controls preload="metadata" aria-label={label} /> : <AudioLines strokeWidth={1.5} aria-hidden="true" />;
  return <File strokeWidth={1.5} aria-hidden="true" />;
}
