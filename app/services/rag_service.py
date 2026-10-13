from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from time import perf_counter

from app.models.schemas import AskResponse, AskSource
from app.services.retrieval_service import RetrievalResult, RetrievalService
from app.utils.dedup import similarity, stitch


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
        min_rag_hits: int,
        dedup_similarity_threshold: float,
        merge_adjacent_chunks: bool,
    ) -> None:
        self.retrieval = retrieval
        self.top_k = top_k
        self.max_context_chunks = max_context_chunks
        self.min_rag_hits = min_rag_hits
        self.dedup_similarity_threshold = dedup_similarity_threshold
        self.merge_adjacent_chunks = merge_adjacent_chunks

    async def retrieve(
        self,
        question: str,
        *,
        collection: str | None = None,
        top_k: int | None = None,
        language: str | None = None,
    ) -> AskResponse:
        timings = RequestTimings()
        effective_top_k = top_k if top_k and top_k > 0 else self.top_k

        db_started = perf_counter()
        results = await self.retrieval.retrieve(
            question,
            effective_top_k,
            collection,
            language,
        )
        timings.db_ms = (perf_counter() - db_started) * 1000

        # Relevance filtering happens in the retrieval stage now (dense score
        # threshold, plus the cross-encoder when RERANK_ENABLED is on).
        strong_hits = self._collapse_overlapping(results)

        if len(strong_hits) < self.min_rag_hits:
            self._log_timing(
                timings=timings,
                strong_hits=len(strong_hits),
                total_hits=len(results),
                context_count=0,
            )
            return AskResponse()

        contexts, sources, scores = self._build_contexts_and_sources(
            strong_hits,
            limit=min(self.max_context_chunks, effective_top_k),
        )
        self._log_timing(
            timings=timings,
            strong_hits=len(strong_hits),
            total_hits=len(results),
            context_count=len(contexts),
        )
        return AskResponse(contexts=contexts, sources=sources, scores=scores)

    def _collapse_overlapping(
        self,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Fold chunk_overlap duplicates into a single context.

        Neighbouring chunks share up to CHUNK_OVERLAP characters, so a top-k of
        6 can easily be three renderings of the same passage. Every duplicate
        costs prompt budget and shows up as its own citation in the answer.
        """
        kept: list[RetrievalResult] = []

        for result in results:
            payload = result['payload']
            text = str(payload.get('text', '')).strip()
            if not text:
                continue

            absorbed = False
            for existing in kept:
                existing_payload = existing['payload']

                if self.merge_adjacent_chunks and self._is_adjacent(
                    existing_payload, payload
                ):
                    first, second = sorted(
                        (existing_payload, payload),
                        key=lambda item: int(item.get('chunk_idx', 0) or 0),
                    )
                    existing_payload['text'] = stitch(
                        str(first.get('text', '')),
                        str(second.get('text', '')),
                    )
                    absorbed = True
                    break

                existing_text = str(existing_payload.get('text', ''))
                if similarity(existing_text, text) >= self.dedup_similarity_threshold:
                    absorbed = True
                    break

            if not absorbed:
                kept.append(result)

        if len(kept) < len(results):
            logger.info(
                'collapsed overlapping chunks | in=%d out=%d',
                len(results),
                len(kept),
            )
        return kept

    @staticmethod
    def _is_adjacent(left: dict[str, object], right: dict[str, object]) -> bool:
        if left.get('source_id') != right.get('source_id'):
            return False
        if left.get('section_idx') != right.get('section_idx'):
            return False
        try:
            distance = abs(
                int(left.get('chunk_idx', 0) or 0)
                - int(right.get('chunk_idx', 0) or 0)
            )
        except (TypeError, ValueError):
            return False
        return distance == 1

    def _build_contexts_and_sources(
        self,
        strong_hits: list[RetrievalResult],
        limit: int,
    ) -> tuple[list[str], list[AskSource], list[float]]:
        """Keep contexts, sources, and scores 1:1 aligned for citation mapping."""
        contexts: list[str] = []
        sources: list[AskSource] = []
        scores: list[float] = []

        for result in strong_hits[:limit]:
            payload = result['payload']
            title = str(payload.get('title', '')).strip()
            text = str(payload.get('text', '')).strip()
            source_url = str(payload.get('source_url', '')).strip()
            source_id = str(payload.get('source_id', '')).strip()
            source_type = str(payload.get('source_type', '')).strip() or 'rag'
            score = float(result.get('score', 0.0))

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
                    score=score,
                )
            )
            scores.append(score)

        return contexts, sources, scores

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
