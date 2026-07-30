from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar

import httpx

from app.core.config import Settings
from app.core.errors import OllamaServiceError
from app.providers._provider_utils import is_timeout_error, is_transient_http_error
from app.utils.retry import retry_async

T = TypeVar('T')


async def _run_with_ollama_retry(
    call: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    retry_base_delay: float,
    error_message: str,
) -> T:
    try:
        return await retry_async(
            call,
            max_attempts=max_retries,
            base_delay=retry_base_delay,
            retryable=is_transient_http_error,
        )
    except Exception as exc:
        raise OllamaServiceError(
            f'{error_message}: {exc}',
            is_timeout=isinstance(exc, httpx.TimeoutException) or is_timeout_error(exc),
        ) from exc


class OllamaEmbeddingProvider:
    def __init__(self, settings: Settings) -> None:
        base_url = (settings.ollama_url or '').strip().rstrip('/')
        if not base_url:
            raise ValueError('OLLAMA_URL fehlt.')
        self._base_url = base_url
        self._model = settings.ollama_embedding_model
        self._timeout = settings.ollama_timeout_seconds
        self._max_retries = settings.provider_max_retries
        self._retry_base_delay = settings.provider_retry_base_delay_seconds
        self._client = httpx.AsyncClient(timeout=self._timeout)

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        semaphore = asyncio.Semaphore(8)

        async def _limited(text: str) -> list[float]:
            async with semaphore:
                return await self._embed_one(text)

        return list(await asyncio.gather(*(_limited(text) for text in texts)))
    async def embed_query(self, text: str) -> list[float]:
        vectors = await self.embed_texts([text])
        return vectors[0]

    async def _embed_one(self, text: str) -> list[float]:
        async def _call() -> list[float]:
            response = await self._client.post(
                f'{self._base_url}/api/embeddings',
                json={
                    'model': self._model,
                    'prompt': text,
                },
            )
            response.raise_for_status()
            data = response.json()
            embedding = data.get('embedding')
            if not isinstance(embedding, list) or not embedding:
                raise ValueError('Ollama embedding response missing embedding vector')
            return [float(value) for value in embedding]

        return await _run_with_ollama_retry(
            _call,
            max_retries=self._max_retries,
            retry_base_delay=self._retry_base_delay,
            error_message='Ollama embedding request failed',
        )

    async def close(self) -> None:
        await self._client.aclose()
