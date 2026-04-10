from __future__ import annotations

import re
from typing import Any

import httpx

from app.core.config import Settings


class SotraProvider:
    def __init__(self, settings: Settings) -> None:
        self._url = (settings.sotra_url or '').strip() or None
        self._api_key = (settings.sotra_api_key or '').strip() or None
        self._timeout = settings.sotra_timeout_seconds

    async def translate_hsb_to_de(self, text: str) -> str:
        if not text.strip():
            return text

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            if not self._url:
                raise ValueError('SOTRA_URL fehlt.')
            if not self._api_key:
                raise ValueError('SOTRA_API_KEY fehlt.')

            response = await client.post(
                self._url.rstrip('/'),
                params={
                    'uri': '/ws/translate/',
                    'api_key': self._api_key,
                    '_version': '2.2.01',
                },
                json={
                    'direction': 'hsb_de',
                    'warnings': False,
                    'text': text,
                },
            )
            response.raise_for_status()
            data = response.json()
            return str(data.get('output_html') or '').strip()

    async def translate_de_to_hsb(self, text: str) -> str:
        if not text.strip():
            return text

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            if not self._url:
                raise ValueError('SOTRA_URL fehlt.')
            if not self._api_key:
                raise ValueError('SOTRA_API_KEY fehlt.')

            response = await client.post(
                self._url.rstrip('/'),
                params={
                    'uri': '/ws/translate/',
                    'api_key': self._api_key,
                    '_version': '2.2.01',
                },
                json={
                    'direction': 'de_hsb',
                    'warnings': False,
                    'text': text,
                },
            )
            response.raise_for_status()
            data = response.json()
            return str(data.get('output_html') or '').strip()


    @staticmethod
    def _join_marked_translation(marked_translation: Any, separator: str) -> str:
        if not isinstance(marked_translation, list):
            return ''

        joined_rows: list[str] = []
        for item in marked_translation:
            if isinstance(item, list):
                joined_rows.append(' '.join(str(part) for part in item).strip())
            else:
                joined_rows.append(str(item).strip())

        return separator.join(row for row in joined_rows if row).strip()