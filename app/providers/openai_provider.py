from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, TypeVar

from openai import AsyncOpenAI

from app.core.config import Settings
from app.core.errors import OpenAIServiceError
from app.providers._provider_utils import is_timeout_error, is_transient_openai_error
from app.utils.retry import retry_async

T = TypeVar('T')


def _create_async_client(settings: Settings) -> AsyncOpenAI:
    client_kwargs: dict[str, Any] = {
        'api_key': settings.openai_api_key,
        'timeout': settings.openai_timeout_seconds,
    }
    if settings.openai_base_url:
        client_kwargs['base_url'] = settings.openai_base_url
    return AsyncOpenAI(**client_kwargs)


async def _run_with_openai_retry(
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
            retryable=is_transient_openai_error,
        )
    except Exception as exc:
        raise OpenAIServiceError(
            f'{error_message}: {exc}',
            is_timeout=is_timeout_error(exc),
        ) from exc


def _require_api_key(settings: Settings) -> None:
    if not settings.openai_api_key:
        raise ValueError('OPENAI_API_KEY fehlt.')


class OpenAIEmbeddingProvider:
    def __init__(self, settings: Settings) -> None:
        _require_api_key(settings)
        self._client = _create_async_client(settings)
        self._model = settings.openai_embedding_model
        self._max_retries = settings.provider_max_retries
        self._retry_base_delay = settings.provider_retry_base_delay_seconds

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        async def _call() -> list[list[float]]:
            response = await self._client.embeddings.create(
                model=self._model,
                input=list(texts),
            )
            return [item.embedding for item in response.data]

        return await _run_with_openai_retry(
            _call,
            max_retries=self._max_retries,
            retry_base_delay=self._retry_base_delay,
            error_message='OpenAI embedding request failed',
        )

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self.embed_texts([text])
        return vectors[0]
