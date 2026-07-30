from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from app.core.config import Settings
from app.providers.ollama_provider import OllamaEmbeddingProvider
from app.providers.openai_provider import OpenAIEmbeddingProvider


class EmbeddingProvider(Protocol):
    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == 'ollama':
        return OllamaEmbeddingProvider(settings)
    return OpenAIEmbeddingProvider(settings)
