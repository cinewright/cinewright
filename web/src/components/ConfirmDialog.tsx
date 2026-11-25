// Confirmation for a permanent deletion (design/components/Button: danger actions that delete need one).
// Mount it to ask; Cancel, Escape and the confirm button each end it through the callbacks.

import { useEffect, useId, useRef, type ReactNode } from 'react';

export function ConfirmDialog({
  title,
  confirmLabel,
  onConfirm,
  onCancel,
  children,
}: {
  title: string;
  confirmLabel: string;
  onConfirm: () => void;
  onCancel: () => void;
  children: ReactNode;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const titleId = useId();

  useEffect(() => {
    const dialog = ref.current;
    // Removing the element on unmount also ends the modal state.
    if (dialog && !dialog.open) dialog.showModal();
  }, []);

  return (
    <dialog ref={ref} className="dialog" aria-labelledby={titleId} onClose={onCancel}>
      <div className="dialog__head">
        <h2 id={titleId} className="dialog__title">
          {title}
        </h2>
      </div>
      <div className="dialog__body stack">
        {children}
        <div className="dialog__actions">
          <button type="button" className="cw-btn cw-btn--ghost" onClick={onCancel}>
            Cancel
          </button>
          <button type="button" className="cw-btn cw-btn--danger" onClick={onConfirm}>
            {confirmLabel}
          </button>
        </div>
      </div>
    </dialog>
  );
}
