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
        self._batch_size = max(1, settings.embedding_batch_size)
        self._max_parallel_batches = max(1, settings.embedding_max_parallel_batches)
        headers: dict[str, str] = {}
        if settings.ollama_api_key:
            headers['Authorization'] = f'Bearer {settings.ollama_api_key}'
        self._client = httpx.AsyncClient(timeout=self._timeout, headers=headers)

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        items = list(texts)
        if not items:
            return []

        batches = [
            items[start : start + self._batch_size]
            for start in range(0, len(items), self._batch_size)
        ]
        semaphore = asyncio.Semaphore(self._max_parallel_batches)

        async def _limited(batch: list[str]) -> list[list[float]]:
            async with semaphore:
                return await self._embed_batch(batch)

        results = await asyncio.gather(*(_limited(batch) for batch in batches))
        return [vector for batch_result in results for vector in batch_result]

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self.embed_texts([text])
        return vectors[0]

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        async def _call() -> list[list[float]]:
            # /api/embed supersedes /api/embeddings and accepts an array,
            # turning one request per chunk into one request per batch.
            response = await self._client.post(
                f'{self._base_url}/api/embed',
                json={
                    'model': self._model,
                    'input': batch,
                },
            )
            response.raise_for_status()
            data = response.json()
            embeddings = data.get('embeddings')
            if not isinstance(embeddings, list) or len(embeddings) != len(batch):
                raise ValueError(
                    'Ollama embedding response did not return one vector per input'
                )
            return [[float(value) for value in vector] for vector in embeddings]

        return await _run_with_ollama_retry(
            _call,
            max_retries=self._max_retries,
            retry_base_delay=self._retry_base_delay,
            error_message='Ollama embedding request failed',
        )

    async def close(self) -> None:
        await self._client.aclose()
