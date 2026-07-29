from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from time import perf_counter

from app.models.schemas import AskResponse, AskSource
from app.providers.openai_provider import OpenAILLMProvider, WebSource
from app.providers.sotra_provider import SotraProvider
from app.services.retrieval_service import RetrievalResult, RetrievalService
from app.utils.language import detect_query_language


logger = logging.getLogger(__name__)


@dataclass
class RequestTimings:
    started_at: float = field(default_factory=perf_counter)
    db_ms: float = 0.0
    translation_ms: float = 0.0
    openai_ms: float = 0.0
    back_translation_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return (perf_counter() - self.started_at) * 1000


class RagService:
    def __init__(
        self,
        retrieval: RetrievalService,
        llm: OpenAILLMProvider,
        sotra: SotraProvider,
        top_k: int,
        max_context_chunks: int,
        retrieval_min_score: float,
        min_rag_hits: int,
    ) -> None:
        self.retrieval = retrieval
        self.llm = llm
        self.sotra = sotra
        self.top_k = top_k
        self.max_context_chunks = max_context_chunks
        self.retrieval_min_score = retrieval_min_score
        self.min_rag_hits = min_rag_hits

    async def answer(
        self,
        question_hsb: str,
        history: list[str] | None = None,
        is_phone_call: bool = False,
    ) -> AskResponse:
        history_items = history or []
        timings = RequestTimings()
        query_language = detect_query_language(question_hsb)

        db_started = perf_counter()
        results = await self.retrieval.retrieve(question_hsb, self.top_k)
        timings.db_ms = (perf_counter() - db_started) * 1000

        strong_hits = self._filter_strong_hits(results)
        if len(strong_hits) >= self.min_rag_hits:
            rag_response = await self._answer_via_rag(
                question_hsb=question_hsb,
                query_language=query_language,
                strong_hits=strong_hits,
                history_items=history_items,
                is_phone_call=is_phone_call,
                timings=timings,
            )
            if rag_response is not None:
                self._log_timing(
                    strategy='rag',
                    timings=timings,
                    strong_hits=len(strong_hits),
                    total_hits=len(results),
                    history_items=len(history_items),
                    query_language=query_language,
                )
                return rag_response

        response = await self._answer_via_web(
            question_hsb=question_hsb,
            query_language=query_language,
            history_items=history_items,
            is_phone_call=is_phone_call,
            timings=timings,
        )
        self._log_timing(
            strategy='web',
            timings=timings,
            strong_hits=len(strong_hits),
            total_hits=len(results),
            history_items=len(history_items),
            query_language=query_language,
        )
        return response

    def _filter_strong_hits(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        return [
            result
            for result in results
            if float(result.get('score', 0.0)) >= self.retrieval_min_score
        ]

    async def _answer_via_rag(
        self,
        *,
        question_hsb: str,
        query_language: str,
        strong_hits: list[RetrievalResult],
        history_items: list[str],
        is_phone_call: bool,
        timings: RequestTimings,
    ) -> AskResponse | None:
        contexts, sources = self._build_contexts_and_sources(strong_hits)
        if not contexts:
            return None

        translation_started = perf_counter()
        question_de = await self._question_for_llm(question_hsb, query_language)
        context_de = await asyncio.gather(
            *[self.sotra.translate_hsb_to_de(context) for context in contexts]
        )
        timings.translation_ms = (perf_counter() - translation_started) * 1000

        openai_started = perf_counter()
        answer_de = await self.llm.answer_question(
            question=question_de,
            contexts=list(context_de),
            history=history_items,
            is_phone_call=is_phone_call,
        )
        timings.openai_ms = (perf_counter() - openai_started) * 1000

        back_started = perf_counter()
        answer_hsb = await self.sotra.translate_de_to_hsb(answer_de)
        timings.back_translation_ms = (perf_counter() - back_started) * 1000

        return AskResponse(
            answer=answer_hsb,
            sources=sources,
            source_strategy='rag',
        )

    async def _answer_via_web(
        self,
        *,
        question_hsb: str,
        query_language: str,
        history_items: list[str],
        is_phone_call: bool,
        timings: RequestTimings,
    ) -> AskResponse:
        translation_started = perf_counter()
        question_de = await self._question_for_llm(question_hsb, query_language)
        timings.translation_ms = (perf_counter() - translation_started) * 1000

        openai_started = perf_counter()
        web_result = await self.llm.answer_with_web_search(
            question_de,
            history=history_items,
            is_phone_call=is_phone_call,
        )
        timings.openai_ms = (perf_counter() - openai_started) * 1000

        back_started = perf_counter()
        answer_hsb = await self.sotra.translate_de_to_hsb(web_result['answer'])
        timings.back_translation_ms = (perf_counter() - back_started) * 1000

        return AskResponse(
            answer=answer_hsb,
            sources=[self._to_ask_source(src) for src in web_result['sources']],
            source_strategy='web',
        )

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

    async def _question_for_llm(self, question: str, query_language: str) -> str:
        if query_language == 'de':
            return question
        return await self.sotra.translate_hsb_to_de(question)

    @staticmethod
    def _to_ask_source(source: WebSource) -> AskSource:
        return AskSource(
            source_type=source['source_type'],
            source_url=source['source_url'],
            title=source.get('title', ''),
        )

    @staticmethod
    def _log_timing(
        *,
        strategy: str,
        timings: RequestTimings,
        strong_hits: int,
        total_hits: int,
        history_items: int,
        query_language: str,
    ) -> None:
        logger.info(
            'ask timing | strategy=%s | total=%.0fms | db=%.0fms | tr=%.0fms | '
            'openai=%.0fms | back_tr=%.0fms | hits=%d/%d | history_items=%d | query_lang=%s',
            strategy,
            timings.total_ms,
            timings.db_ms,
            timings.translation_ms,
            timings.openai_ms,
            timings.back_translation_ms,
            strong_hits,
            total_hits,
            history_items,
            query_language,
        )
