from __future__ import annotations

from app.models.schemas import AskResponse, AskSource
from app.providers.openai_provider import OpenAILLMProvider
from app.services.retrieval_service import RetrievalService


class RagService:
    def __init__(
        self,
        retrieval: RetrievalService,
        llm: OpenAILLMProvider,
        top_k: int,
        max_context_chunks: int,
    ) -> None:
        self.retrieval = retrieval
        self.llm = llm
        self.top_k = top_k
        self.max_context_chunks = max_context_chunks

    async def answer(self, question: str) -> AskResponse:
        results = await self.retrieval.retrieve(question, self.top_k)

        contexts: list[str] = []
        sources: list[AskSource] = []
        seen_sources: set[tuple[str, str, str, str]] = set()

        for result in results[: self.max_context_chunks]:
            payload = result['payload']
            title = str(payload.get('title', '')).strip()
            text = str(payload.get('text', '')).strip()
            source_id = str(payload.get('source_id', '')).strip()
            source_type = str(payload.get('source_type', '')).strip()
            source_url = str(payload.get('source_url', '')).strip()

            if not text:
                continue

            if title:
                contexts.append(f'Titel: {title}\nInhalt: {text}')
            else:
                contexts.append(text)

            dedupe_key = (source_id, source_type, source_url, title)
            if source_id and dedupe_key not in seen_sources:
                seen_sources.add(dedupe_key)
                sources.append(
                    AskSource(
                        source_id=source_id,
                        source_type=source_type,
                        source_url=source_url,
                        title=title,
                    )
                )

        if not contexts:
            return AskResponse(
                answer='Ich habe in der Wissensdatenbank keinen passenden Kontext gefunden.',
                sources=[],
            )

        answer = await self.llm.answer_question(question, contexts)
        return AskResponse(answer=answer, sources=sources)