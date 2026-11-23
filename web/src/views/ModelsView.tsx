// Models: one card per model and quantization from /models.

import { api } from '../api/client';
import { ErrorBanner, RetryButton } from '../components/Feedback';
import { ModelCard } from '../components/ModelCard';
import { useApi } from '../lib/useApi';

export function ModelsView() {
  const models = useApi((signal) => api.models(signal));
  const items = models.data?.items ?? [];

  return (
    <div className="view">
      <header className="view__head">
        <h1 className="view__title">Models</h1>
      </header>

      {models.error ? <ErrorBanner title="Could not load models" error={models.error} action={<RetryButton onClick={models.reload} />} /> : null}
      {!models.loading && !models.error && items.length === 0 ? <p className="cw-muted">No models in the registry.</p> : null}

      <div className="card-grid">
        {items.flatMap((model) => model.quantizations.map((quant) => <ModelCard key={`${model.id}/${quant.quant}`} model={model} quant={quant} />))}
      </div>
    </div>
  );
}
