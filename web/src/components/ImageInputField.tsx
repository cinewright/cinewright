// An input image of a job (start frame or image to edit): choose, upload, preview, replace, remove.
// The file goes to POST /assets as soon as it is chosen, so the estimate can check it.

import { ImagePlus } from 'lucide-react';
import { useEffect, useId, useRef } from 'react';

import { UPLOAD_IMAGE_TYPES, assetContentUrl, errorMessage, type Asset } from '../api/client';
import { formatBytes } from '../lib/format';

interface Props {
  label: string;
  image: Asset | null;
  /** Name of the file being uploaded, or null. */
  uploading: string | null;
  error: unknown;
  onChoose: (file: File) => void;
  onRemove: () => void;
}

export function ImageInputField({ label, image, uploading, error, onChoose, onRemove }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const chooseButton = useRef<HTMLButtonElement>(null);
  const replaceButton = useRef<HTMLButtonElement>(null);
  const hadImage = useRef(image !== null);
  const labelId = useId();
  const busy = uploading !== null;
  const choose = () => input.current?.click();

  // Adding or removing the image replaces the focused button; keep keyboard focus in the field, unless the user has
  // already moved it elsewhere. A disabled button takes no focus, so this waits until the upload has ended.
  useEffect(() => {
    if (busy) return;
    const hasImage = image !== null;
    if (hasImage === hadImage.current) return;
    hadImage.current = hasImage;
    const focused = document.activeElement;
    if (focused === null || focused === document.body) (hasImage ? replaceButton : chooseButton).current?.focus();
  }, [image, busy]);

  return (
    <div className="cw-field" role="group" aria-labelledby={labelId} data-invalid={error ? 'true' : undefined}>
      <span className="cw-field__label" id={labelId}>
        {label}
      </span>
      {image ? (
        <div className="image-input">
          <span className="image-input__thumb">
            <img src={assetContentUrl(image.sha256)} alt={`${label} preview`} />
          </span>
          <span className="image-input__meta text-value">
            {image.media_type} · {formatBytes(image.size_bytes)}
          </span>
          <span className="image-input__actions">
            <button ref={replaceButton} type="button" className="cw-btn cw-btn--secondary cw-btn--sm" onClick={choose} disabled={busy}>
              {busy ? 'Uploading' : 'Replace image'}
            </button>
            <button type="button" className="cw-btn cw-btn--ghost cw-btn--sm" onClick={onRemove} disabled={busy}>
              Remove image
            </button>
          </span>
        </div>
      ) : (
        <div>
          <button ref={chooseButton} type="button" className="cw-btn cw-btn--secondary" onClick={choose} disabled={busy}>
            <ImagePlus strokeWidth={1.5} aria-hidden="true" />
            {busy ? 'Uploading' : 'Choose image'}
          </button>
        </div>
      )}
      <input
        ref={input}
        type="file"
        accept={UPLOAD_IMAGE_TYPES.join(',')}
        hidden
        aria-label={`${label} file`}
        onChange={(event) => {
          const file = event.target.files?.[0];
          event.target.value = ''; // choosing the same file again is a new choice
          if (file) onChoose(file);
        }}
      />
      {/* The finished upload is announced; a refusal speaks through the alert below. */}
      <span className="sr-only" role="status">
        {image && !busy ? `${label} uploaded` : ''}
      </span>
      {error ? (
        <span className="cw-field__hint" role="alert">
          {errorMessage(error)}
        </span>
      ) : (
        <span className="cw-field__hint">
          {busy ? `Uploading ${uploading}` : 'PNG, JPEG or WebP, stored in the Library with the selected data class, or a stricter one it already has.'}
        </span>
      )}
    </div>
  );
}
