from __future__ import annotations

from typing import Literal

PromptStyle = Literal['gemma', 'none']

# EmbeddingGemma is trained with task prefixes and degrades noticeably without
# them. Queries and documents use deliberately different prompts so the model
# can tell a question apart from a passage.
# https://ai.google.dev/gemma/docs/embeddinggemma/model_card
_GEMMA_QUERY_PREFIX = 'task: search result | query: '


def resolve_prompt_style(configured: str | None, model_name: str) -> PromptStyle:
    """Explicit config wins; otherwise infer from the model name."""
    normalized = (configured or '').strip().lower()
    if normalized in ('gemma', 'none'):
        return normalized  # type: ignore[return-value]
    return 'gemma' if 'gemma' in model_name.lower() else 'none'


def format_query(text: str, style: PromptStyle) -> str:
    if style == 'gemma':
        return f'{_GEMMA_QUERY_PREFIX}{text}'
    return text


def format_document(text: str, title: str | None, style: PromptStyle) -> str:
    if style == 'gemma':
        clean_title = (title or '').strip() or 'none'
        return f'title: {clean_title} | text: {text}'
    return text
