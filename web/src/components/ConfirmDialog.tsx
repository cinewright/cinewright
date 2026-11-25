// Confirmation for a permanent deletion (design/components/Button: danger actions that delete need one).
// Mount it to ask; Cancel, Escape and the confirm button each end it through the callbacks. Every way out closes the
// dialog first, so the browser returns focus to the control that opened it.

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
  const confirmed = useRef(false);
  const titleId = useId();

  useEffect(() => {
    const dialog = ref.current;
    if (dialog && !dialog.open) dialog.showModal();
  }, []);

  const close = (confirm: boolean) => {
    confirmed.current = confirm;
    ref.current?.close();
  };

  return (
    <dialog ref={ref} className="dialog" aria-labelledby={titleId} onClose={() => (confirmed.current ? onConfirm() : onCancel())}>
      <div className="dialog__head">
        <h2 id={titleId} className="dialog__title">
          {title}
        </h2>
      </div>
      <div className="dialog__body stack">
        {children}
        <div className="dialog__actions">
          <button type="button" className="cw-btn cw-btn--ghost" onClick={() => close(false)}>
            Cancel
          </button>
          <button type="button" className="cw-btn cw-btn--danger" onClick={() => close(true)}>
            {confirmLabel}
          </button>
        </div>
      </div>
    </dialog>
  );
}
