from __future__ import annotations

import asyncio

from app.models.schemas import ParsedSection
from app.parsers.adapters import parse_html_content, parse_url_content


class ParserService:
    async def parse_html(
        self,
        html: str,
        *,
        source_hint: str | None = None,
        min_chars: int = 40,
    ) -> list[ParsedSection]:
        return await asyncio.to_thread(parse_html_content, html, source_hint, min_chars)

    async def parse_url(self, url: str, *, min_chars: int = 40) -> list[ParsedSection]:
        return await parse_url_content(url, min_chars=min_chars)
