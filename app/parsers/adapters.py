from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlparse

import httpx

from app.models.schemas import ParsedSection
from app.parsers.site_parsers import (
    parse_function,
    parse_katolski_posol_html,
    parse_lucija,
    parse_mdr_serbski,
    parse_pfarrei_crostwitz,
    parse_posol,
    parse_serbske_nowiny,
    parse_zalozba,
)


class InvalidUrlError(ValueError):
    pass


class FetchError(RuntimeError):
    def __init__(self, message: str, *, is_timeout: bool = False) -> None:
        super().__init__(message)
        self.is_timeout = is_timeout


@dataclass(frozen=True)
class ParserSpec:
    name: str
    matcher: Callable[[str], bool]
    parser: Callable[..., list[dict[str, str | int]]]


def _normalize_items(items: list[dict] | list[ParsedSection]) -> list[ParsedSection]:
    normalized: list[ParsedSection] = []
    for item in items:
        if isinstance(item, ParsedSection):
            normalized.append(item)
        else:
            normalized.append(
                ParsedSection(
                    title=str(item.get('title', '')).strip(),
                    text=str(item.get('text', '')).strip(),
                )
            )
    return [item for item in normalized if item.text]


def _contains_any(target: str, needles: tuple[str, ...]) -> bool:
    lowered = target.lower()
    return any(needle in lowered for needle in needles)


PARSER_REGISTRY: list[ParserSpec] = [
    ParserSpec(
        name='katolski_posol_html',
        matcher=lambda value: _contains_any(value, ('katolski_posol', 'katolski-posol', 'katolskiposol')),
        parser=parse_katolski_posol_html,
    ),
    ParserSpec(
        name='serbske_nowiny',
        matcher=lambda value: _contains_any(value, ('serbske-nowiny', 'serbskenowiny', 'serbske_nowiny')),
        parser=parse_serbske_nowiny,
    ),
    ParserSpec(
        name='pfarrei_crostwitz',
        matcher=lambda value: _contains_any(value, ('crostwitz', 'pfarrei-crostwitz', 'pfarrei_crostwitz')),
        parser=parse_pfarrei_crostwitz,
    ),
    ParserSpec(
        name='mdr_serbski',
        matcher=lambda value: _contains_any(value, ('mdr', 'serbski')),
        parser=parse_mdr_serbski,
    ),
    ParserSpec(
        name='zalozba',
        matcher=lambda value: _contains_any(value, ('zalozba',)),
        parser=parse_zalozba,
    ),
    ParserSpec(
        name='lucija',
        matcher=lambda value: _contains_any(value, ('lucija',)),
        parser=parse_lucija,
    ),
    ParserSpec(
        name='posol',
        matcher=lambda value: _contains_any(value, ('posol',)),
        parser=parse_posol,
    ),
]


def select_parser(source_hint: str | None = None) -> Callable[..., list[dict[str, str | int]]]:
    hint = (source_hint or '').strip().lower()
    if hint:
        for spec in PARSER_REGISTRY:
            if spec.matcher(hint):
                return spec.parser
    return parse_function


async def fetch_html(url: str, timeout: float = 12.0) -> str:
    raw_url = (url or '').strip()
    parsed = urlparse(raw_url)
    if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
        raise InvalidUrlError('Ungültige URL. Erwarte http:// oder https:// mit Hostnamen.')

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
            response = await client.get(raw_url, headers={'User-Agent': 'rag-server/0.1'})
            response.raise_for_status()
            return response.text
    except httpx.TimeoutException as exc:
        raise FetchError(f'Timeout beim Laden von {raw_url}', is_timeout=True) from exc
    except httpx.HTTPStatusError as exc:
        raise FetchError(
            f'Fehler beim Laden von {raw_url}: HTTP {exc.response.status_code}',
            is_timeout=False,
        ) from exc
    except httpx.HTTPError as exc:
        raise FetchError(f'Fehler beim Laden von {raw_url}: {exc}') from exc


def parse_html_content(
    html: str,
    source_hint: str | None = None,
    min_chars: int = 40,
) -> list[ParsedSection]:
    parser = select_parser(source_hint)
    items = parser(
        html,
        min_text_length=min_chars,
        url=source_hint,
    )
    return _normalize_items(items)


async def parse_url_content(url: str, min_chars: int = 40) -> list[ParsedSection]:
    html = await fetch_html(url)
    return parse_html_content(html, source_hint=url, min_chars=min_chars)
