from __future__ import annotations

import logging
from time import perf_counter

from app.models.schemas import AskResponse, AskSource
from app.providers.openai_provider import OpenAILLMProvider
from app.providers.sotra_provider import SotraProvider
from app.services.retrieval_service import RetrievalService


logger = logging.getLogger(__name__)


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

    async def answer(self, question_hsb: str) -> AskResponse:
        request_started = perf_counter()

        local_db_search_started = perf_counter()
        results = await self.retrieval.retrieve(question_hsb, self.top_k)
        local_db_search_ms = (perf_counter() - local_db_search_started) * 1000

        translation_ms = 0.0
        openai_query_ms = 0.0
        back_translation_ms = 0.0

        strong_hits = [
            result
            for result in results
            if float(result.get('score', 0.0)) >= self.retrieval_min_score
        ]

        if len(strong_hits) >= self.min_rag_hits:
            sorbian_contexts: list[str] = []
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
                    sorbian_contexts.append(f'Titel: {title}\nInhalt: {text}')
                else:
                    sorbian_contexts.append(text)

                if source_url and source_url not in seen_urls:
                    seen_urls.add(source_url)
                    sources.append(
                        AskSource(
                            source_type='rag',
                            source_url=source_url,
                            title=title,
                        )
                    )

            if sorbian_contexts:
                translation_started = perf_counter()
                question_de = await self.sotra.translate_hsb_to_de(question_hsb)
                context_de = [
                    await self.sotra.translate_hsb_to_de(context)
                    for context in sorbian_contexts
                ]
                translation_ms = (perf_counter() - translation_started) * 1000

                openai_started = perf_counter()
                answer_de = await self.llm.answer_question(
                    question=question_de,
                    contexts=context_de,
                )
                openai_query_ms = (perf_counter() - openai_started) * 1000

                back_translation_started = perf_counter()
                answer_hsb = await self.sotra.translate_de_to_hsb(answer_de)
                back_translation_ms = (perf_counter() - back_translation_started) * 1000

                total_ms = (perf_counter() - request_started) * 1000
                logger.info(
                    'ask timing | strategy=rag | total=%.0fms | db=%.0fms | tr=%.0fms | openai=%.0fms | back_tr=%.0fms | hits=%d/%d',
                    total_ms,
                    local_db_search_ms,
                    translation_ms,
                    openai_query_ms,
                    back_translation_ms,
                    len(strong_hits),
                    len(results),
                )

                return AskResponse(
                    answer=answer_hsb,
                    sources=sources,
                    source_strategy='rag',
                )

        translation_started = perf_counter()
        question_de = await self.sotra.translate_hsb_to_de(question_hsb)
        translation_ms = (perf_counter() - translation_started) * 1000

        openai_started = perf_counter()
        web_result = await self.llm.answer_with_web_search(question_de)
        openai_query_ms = (perf_counter() - openai_started) * 1000

        back_translation_started = perf_counter()
        answer_hsb = await self.sotra.translate_de_to_hsb(web_result['answer'])
        back_translation_ms = (perf_counter() - back_translation_started) * 1000

        total_ms = (perf_counter() - request_started) * 1000
        logger.info(
            'ask timing | strategy=web | total=%.0fms | db=%.0fms | tr=%.0fms | openai=%.0fms | back_tr=%.0fms | hits=%d/%d',
            total_ms,
            local_db_search_ms,
            translation_ms,
            openai_query_ms,
            back_translation_ms,
            len(strong_hits),
            len(results),
        )

        return AskResponse(
            answer=answer_hsb,
            sources=[
                AskSource(
                    source_type=src['source_type'],
                    source_url=src['source_url'],
                    title=src.get('title', ''),
                )
                for src in web_result['sources']
            ],
            source_strategy='web',
        )