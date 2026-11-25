"""Prompt dialects: a request's prompt in the form the routed model reads.

The core translates once, when it hands a job to a worker (`Core._worker_job`); `WorkerJob.prompt` is the result,
and workers have no registry to translate with. Engine adapters pass the prompt on unchanged.
"""

from __future__ import annotations

from cinewright.api.schemas import PromptDialect


def translate_prompt(prompt: str, dialect: PromptDialect) -> str:
    """The prompt in the model's dialect (the descriptor's `prompt_dialect`).

    `natural` models read plain sentences, so the prompt passes unchanged. `tags` returns it unchanged too: no model
    in the registry reads tags yet, and tag translation arrives with the first one.
    """
    return prompt
