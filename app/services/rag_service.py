from __future__ import annotations

import logging
from dataclasses import dataclass, field
from time import perf_counter

from app.models.schemas import AskResponse, AskSource
from app.services.retrieval_service import RetrievalResult, RetrievalService


logger = logging.getLogger(__name__)


@dataclass
class RequestTimings:
    started_at: float = field(default_factory=perf_counter)
    db_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return (perf_counter() - self.started_at) * 1000


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

    async def retrieve(self, question: str) -> AskResponse:
        timings = RequestTimings()

        db_started = perf_counter()
        results = await self.retrieval.retrieve(question, self.top_k)
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

        contexts, sources = self._build_contexts_and_sources(strong_hits)
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
    ) -> tuple[list[str], list[AskSource]]:
        contexts: list[str] = []
        sources: list[AskSource] = []
        seen_urls: set[str] = set()

        for result in strong_hits[: self.max_context_chunks]:
            payload = result['payload']
            title = str(payload.get('title', '')).strip()
            text = str(payload.get('text', '')).strip()
            source_url = str(payload.get('source_url', '')).strip()

            if not text:
                continue

            if title:
                contexts.append(f'Titel: {title}\nInhalt: {text}')
            else:
                contexts.append(text)

            if source_url and source_url not in seen_urls:
                seen_urls.add(source_url)
                sources.append(
                    AskSource(
                        source_type='rag',
                        source_url=source_url,
                        title=title,
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
