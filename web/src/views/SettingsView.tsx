// Settings: GET/PATCH /settings, the read-only update channel and the licence notices.

import { useEffect, useState, type FormEvent } from 'react';

import { api, type Settings, type SettingsPatch } from '../api/client';
import { ErrorBanner, RetryButton } from '../components/Feedback';
import { WANGP_LICENSE_URL, WANGP_NOTICE, WEB_UI_NOTICES } from '../lib/notices';
import { useApi } from '../lib/useApi';

type NumberKey = 'approval_threshold_eur' | 'budget_eur_per_month' | 'model_store_quota_gb' | 'kwh_price_eur';

interface Draft {
  country: string;
  approval_threshold_eur: string;
  budget_eur_per_month: string;
  model_store_quota_gb: string;
  kwh_price_eur: string;
}

const NUMBER_FIELDS: { key: NumberKey; label: string; hint: string; integer: boolean; required: boolean }[] = [
  { key: 'approval_threshold_eur', label: 'Approval threshold (€)', hint: 'Remote and cloud jobs above it need approval.', integer: false, required: true },
  { key: 'budget_eur_per_month', label: 'Monthly budget (€)', hint: 'Empty = no budget set.', integer: false, required: false },
  { key: 'model_store_quota_gb', label: 'Model store quota (GB)', hint: 'Empty = no quota set.', integer: true, required: false },
  { key: 'kwh_price_eur', label: 'Electricity price (€ per kWh)', hint: 'Empty = not set.', integer: false, required: false },
];

function toDraft(settings: Settings): Draft {
  const text = (value: number | null | undefined) => (value == null ? '' : String(value));
  return {
    country: settings.country ?? 'DE',
    approval_threshold_eur: text(settings.approval_threshold_eur),
    budget_eur_per_month: text(settings.budget_eur_per_month),
    model_store_quota_gb: text(settings.model_store_quota_gb),
    kwh_price_eur: text(settings.kwh_price_eur),
  };
}

function fieldError(field: (typeof NUMBER_FIELDS)[number], text: string): string | null {
  const trimmed = text.trim();
  if (trimmed === '') return field.required ? 'Required' : null;
  const value = Number(trimmed);
  if (!Number.isFinite(value) || value < 0) return 'A number, 0 or more';
  if (field.integer && (!Number.isInteger(value) || value < 1)) return 'A whole number, 1 or more';
  return null;
}

/** Only changed fields: a missing field means "unchanged"; an optional field emptied after having a saved value is sent as null (clears it). */
function toPatch(draft: Draft, saved: Draft): SettingsPatch {
  const patch: SettingsPatch = {};
  const country = draft.country.trim().toUpperCase();
  if (country !== saved.country) patch.country = country;
  for (const field of NUMBER_FIELDS) {
    const text = draft[field.key].trim();
    if (text === saved[field.key]) continue;
    if (text !== '') patch[field.key] = Number(text);
    else if (!field.required && saved[field.key] !== '') patch[field.key] = null;
  }
  return patch;
}

export function SettingsView() {
  const settings = useApi((signal) => api.settings(signal));
  const licenses = useApi((signal) => api.licenses(signal));
  const [draft, setDraft] = useState<Draft>();
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<unknown>();
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    if (settings.data) setDraft(toDraft(settings.data));
  }, [settings.data]);

  const baseline = settings.data ? toDraft(settings.data) : undefined;
  const countryInvalid = draft ? !/^[A-Za-z]{2}$/.test(draft.country.trim()) : false;
  const errors = draft ? NUMBER_FIELDS.map((field) => fieldError(field, draft[field.key])) : [];
  const patch = draft && baseline ? toPatch(draft, baseline) : {};
  const changed = Object.keys(patch).length > 0;

  let blocker: string | null = null;
  if (countryInvalid) blocker = 'Country needs two letters';
  else if (errors.some(Boolean)) blocker = 'Fix the marked fields';
  else if (saving) blocker = 'Saving';
  else if (!changed) blocker = 'No changes to save';

  const update = (key: keyof Draft, value: string) => {
    setSaved(false);
    setDraft((current) => (current ? { ...current, [key]: value } : current));
  };

  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (blocker) return;
    setSaving(true);
    setSaveError(undefined);
    api
      .updateSettings(patch)
      .then((result) => {
        settings.setData(result);
        setSaved(true);
      })
      .catch(setSaveError)
      .finally(() => setSaving(false));
  };

  const notices = licenses.data?.items ?? [];
  const hasWanGP = notices.some((n) => /wan ?gp|wan2gp/i.test(n.component));

  return (
    <div className="view">
      <header className="view__head">
        <h1 className="view__title">Settings</h1>
      </header>

      {settings.error ? <ErrorBanner title="Could not load settings" error={settings.error} action={<RetryButton onClick={settings.reload} />} /> : null}

      {draft ? (
        <form className="panel stack settings" onSubmit={submit} noValidate>
          <div className="form-row">
            <div className="cw-field">
              <span className="cw-field__label">Update channel</span>
              <span className="readonly">Stable</span>
              <span className="cw-field__hint">Releases from main. Release candidates are not offered in the app.</span>
            </div>
            <label className="cw-field" data-invalid={countryInvalid ? 'true' : undefined}>
              <span className="cw-field__label">Country</span>
              <input
                className="cw-input cw-input--mono country-input"
                value={draft.country}
                maxLength={2}
                placeholder="DE"
                onChange={(e) => update('country', e.target.value)}
                aria-invalid={countryInvalid}
              />
              <span className="cw-field__hint">{countryInvalid ? 'Two letters, ISO 3166-1' : 'Decides which model licences apply.'}</span>
            </label>
          </div>
          <div className="form-row">
            {NUMBER_FIELDS.map((field, index) => (
              <label key={field.key} className="cw-field" data-invalid={errors[index] ? 'true' : undefined}>
                <span className="cw-field__label">{field.label}</span>
                <input
                  className="cw-input cw-input--mono number-input"
                  inputMode="decimal"
                  value={draft[field.key]}
                  onChange={(e) => update(field.key, e.target.value)}
                  aria-invalid={Boolean(errors[index])}
                />
                <span className="cw-field__hint">{errors[index] ?? field.hint}</span>
              </label>
            ))}
          </div>
          {saveError ? <ErrorBanner title="Settings were not saved" error={saveError} /> : null}
          <div className="form-actions">
            <button type="submit" className="cw-btn cw-btn--primary" disabled={blocker !== null}>
              {blocker ?? 'Save settings'}
            </button>
            {saved ? (
              <span className="cw-muted" role="status">
                Saved.
              </span>
            ) : null}
          </div>
        </form>
      ) : null}

      <section className="panel stack" aria-labelledby="licences-title">
        <h2 id="licences-title" className="panel__title">
          Licence notices
        </h2>
        {licenses.error ? <ErrorBanner title="Could not load licence notices" error={licenses.error} /> : null}
        <ul className="notices">
          {notices.map((notice) => (
            <li key={notice.component} className="notice">
              <div className="notice__head">
                <span className="text-heading">{notice.component}</span>
                <span className="text-caption cw-muted">
                  {notice.url ? (
                    <a href={notice.url} target="_blank" rel="noreferrer">
                      {notice.license}
                    </a>
                  ) : (
                    notice.license
                  )}
                </span>
              </div>
              {notice.notice ? <p className="notice__body">{notice.notice}</p> : null}
            </li>
          ))}
          {!hasWanGP ? (
            <li className="notice">
              <div className="notice__head">
                <span className="text-heading">WanGP (Wan2GP)</span>
                <span className="text-caption cw-muted">
                  <a href={WANGP_LICENSE_URL} target="_blank" rel="noreferrer">
                    WanGP Community License 2.0
                  </a>
                </span>
              </div>
              <p className="notice__body">{WANGP_NOTICE}</p>
            </li>
          ) : null}
          {WEB_UI_NOTICES.map((notice) => (
            <li key={notice.component} className="notice">
              <div className="notice__head">
                <span className="text-heading">{notice.component}</span>
                <span className="text-caption cw-muted">
                  <a href={notice.url} target="_blank" rel="noreferrer">
                    {notice.license}
                  </a>
                  {' · bundled with the web UI'}
                </span>
              </div>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
