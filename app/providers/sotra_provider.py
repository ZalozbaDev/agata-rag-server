from __future__ import annotations

import httpx

from app.core.config import Settings
from app.core.errors import SotraServiceError
from app.providers._provider_utils import is_timeout_error, is_transient_http_error
from app.utils.retry import retry_async


class SotraProvider:
    def __init__(self, settings: Settings) -> None:
        self._url = (settings.sotra_url or '').strip() or None
        self._api_key = (settings.sotra_api_key or '').strip() or None
        self._timeout = settings.sotra_timeout_seconds
        self._max_retries = settings.provider_max_retries
        self._retry_base_delay = settings.provider_retry_base_delay_seconds
        self._client = httpx.AsyncClient(timeout=self._timeout)

    async def translate_hsb_to_de(self, text: str) -> str:
        return await self._translate(text, direction='hsb_de')

    async def translate_de_to_hsb(self, text: str) -> str:
        return await self._translate(text, direction='de_hsb')

    async def _translate(self, text: str, *, direction: str) -> str:
        if not text.strip():
            return text
        if not self._url:
            raise SotraServiceError('SOTRA_URL fehlt.')
        if not self._api_key:
            raise SotraServiceError('SOTRA_API_KEY fehlt.')

        async def _call() -> str:
            response = await self._client.post(
                self._url.rstrip('/'),
                params={
                    'uri': '/ws/translate/',
                    'api_key': self._api_key,
                    '_version': '2.2.01',
                },
                json={
                    'direction': direction,
                    'warnings': False,
                    'text': text,
                },
            )
            response.raise_for_status()
            data = response.json()
            return str(data.get('output_html') or '').strip()

        try:
            return await retry_async(
                _call,
                max_attempts=self._max_retries,
                base_delay=self._retry_base_delay,
                retryable=is_transient_http_error,
            )
        except Exception as exc:
            raise SotraServiceError(
                f'Sotra translation failed: {exc}',
                is_timeout=isinstance(exc, httpx.TimeoutException) or is_timeout_error(exc),
            ) from exc

    async def close(self) -> None:
        await self._client.aclose()
