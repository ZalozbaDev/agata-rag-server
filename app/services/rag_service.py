from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from time import perf_counter

from app.models.schemas import AskResponse, AskSource
from app.services.retrieval_service import RetrievalResult, RetrievalService


logger = logging.getLogger(__name__)

_PAGE_RE = re.compile(
    r'(?:^|[|\-–—\s])(?:page|seite|strona)\s*(\d+)\b',
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r'\b('
    r'\d{4}-\d{2}-\d{2}'
    r'|\d{1,2}\.\d{1,2}\.\d{4}'
    r'|\d{1,2}\.\s*[A-Za-zäöüÄÖÜ]{3,}\s+\d{4}'
    r')\b'
)


@dataclass
class RequestTimings:
    started_at: float = field(default_factory=perf_counter)
    db_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return (perf_counter() - self.started_at) * 1000


def _extract_page(title: str, payload: dict) -> int | None:
    raw_page = payload.get('page')
    if isinstance(raw_page, int) and raw_page > 0:
        return raw_page
    if isinstance(raw_page, str) and raw_page.strip().isdigit():
        return int(raw_page.strip())

    match = _PAGE_RE.search(title)
    if match:
        return int(match.group(1))
    return None


def _extract_published_at(text: str, payload: dict) -> str | None:
    for key in ('published_at', 'publishedAt', 'date', 'indexed_at'):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    match = _DATE_RE.search(text[:180])
    if match:
        return match.group(1)
    return None


class RagService:
    def __init__(
        self,
        retrieval: RetrievalService,
        top_k: int,
        max_context_chunks: int,
        retrieval_min_score: float,
        min_rag_hits: int,
    ) -> None:
        self.retrieval = retrieval
        self.top_k = top_k
        self.max_context_chunks = max_context_chunks
        self.retrieval_min_score = retrieval_min_score
        self.min_rag_hits = min_rag_hits

    async def retrieve(
        self,
        question: str,
        *,
        collection: str | None = None,
        top_k: int | None = None,
    ) -> AskResponse:
        timings = RequestTimings()
        effective_top_k = top_k if top_k and top_k > 0 else self.top_k

        db_started = perf_counter()
        results = await self.retrieval.retrieve(question, effective_top_k, collection)
        timings.db_ms = (perf_counter() - db_started) * 1000

        strong_hits = self._filter_strong_hits(results)
        if len(strong_hits) < self.min_rag_hits:
            self._log_timing(
                timings=timings,
                strong_hits=len(strong_hits),
                total_hits=len(results),
                context_count=0,
            )
            return AskResponse()

        contexts, sources = self._build_contexts_and_sources(
            strong_hits,
            limit=min(self.max_context_chunks, effective_top_k),
        )
        self._log_timing(
            timings=timings,
            strong_hits=len(strong_hits),
            total_hits=len(results),
            context_count=len(contexts),
        )
        return AskResponse(contexts=contexts, sources=sources)

    def _filter_strong_hits(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        return [
            result
            for result in results
            if float(result.get('score', 0.0)) >= self.retrieval_min_score
        ]

    def _build_contexts_and_sources(
        self,
        strong_hits: list[RetrievalResult],
        limit: int,
    ) -> tuple[list[str], list[AskSource]]:
        """Keep contexts and sources 1:1 aligned for downstream citation mapping."""
        contexts: list[str] = []
        sources: list[AskSource] = []

        for result in strong_hits[:limit]:
            payload = result['payload']
            title = str(payload.get('title', '')).strip()
            text = str(payload.get('text', '')).strip()
            source_url = str(payload.get('source_url', '')).strip()
            source_id = str(payload.get('source_id', '')).strip()
            source_type = str(payload.get('source_type', '')).strip() or 'rag'

            if not text:
                continue

            if title:
                contexts.append(f'Titel: {title}\nInhalt: {text}')
            else:
                contexts.append(text)

            sources.append(
                AskSource(
                    source_type=source_type,
                    source_url=source_url,
                    title=title,
                    source_id=source_id,
                    page=_extract_page(title, payload),
                    published_at=_extract_published_at(text, payload),
                    section=title or None,
                )
            )

        return contexts, sources

    @staticmethod
    def _log_timing(
        *,
        timings: RequestTimings,
        strong_hits: int,
        total_hits: int,
        context_count: int,
    ) -> None:
        logger.info(
            'ask timing | total=%.0fms | db=%.0fms | hits=%d/%d | contexts=%d',
            timings.total_ms,
            timings.db_ms,
            strong_hits,
            total_hits,
            context_count,
        )
