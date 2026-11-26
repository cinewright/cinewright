// An asset's media from the asset store, as the Library and the Job panel show it: an image, a clip (muted, never
// playing by itself), audio, or a file icon for any other type.

import { AudioLines, File } from 'lucide-react';

import { assetContentUrl, type Asset } from '../api/client';

const CONCEALED = { visibility: 'hidden' } as const;

interface AssetMediaProps {
  asset: Asset;
  /** Adds the player controls to clips and audio; without them audio shows as an icon. */
  controls: boolean;
  /** Loads the media at once and in full, for a place that shows it right away; otherwise images load lazily. */
  eager?: boolean;
  /** Lays the media out but keeps it out of sight, so it can load before it shows. */
  concealed?: boolean;
  /** The image, or the clip's first frame, can be shown. Audio and file icons have nothing to wait for. */
  onReady?: () => void;
  /** The media could not be loaded or decoded. */
  onFail?: () => void;
}

export function AssetMedia({ asset, controls, eager = false, concealed = false, onReady, onFail }: AssetMediaProps) {
  const url = assetContentUrl(asset.sha256);
  const label = `${asset.media_type} ${asset.sha256.slice(0, 8)}`;
  const style = concealed ? CONCEALED : undefined;
  if (asset.media_type.startsWith('image/'))
    return <img src={url} alt={label} loading={eager ? 'eager' : 'lazy'} style={style} onLoad={onReady} onError={onFail} />;
  if (asset.media_type.startsWith('video/'))
    return (
      <video
        src={url}
        controls={controls}
        muted
        playsInline
        preload={eager ? 'auto' : 'metadata'}
        aria-label={label}
        style={style}
        onLoadedData={onReady}
        onError={onFail}
      />
    );
  if (asset.media_type.startsWith('audio/'))
    return controls ? (
      <audio src={url} controls preload="metadata" aria-label={label} style={style} onError={onFail} />
    ) : (
      <AudioLines strokeWidth={1.5} aria-hidden="true" />
    );
  return <File strokeWidth={1.5} aria-hidden="true" />;
}
