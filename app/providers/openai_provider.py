from __future__ import annotations

from typing import Sequence

from openai import AsyncOpenAI

from app.core.config import Settings


class OpenAIEmbeddingProvider:
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError('OPENAI_API_KEY fehlt.')
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self._model = settings.openai_embedding_model

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(
            model=self._model,
            input=list(texts),
        )
        return [item.embedding for item in response.data]

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self.embed_texts([text])
        return vectors[0]


class OpenAILLMProvider:
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError('OPENAI_API_KEY fehlt.')
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self._model = settings.openai_chat_model

    async def answer_question(self, question: str, contexts: list[str]) -> str:
        context_block = '\n\n---\n\n'.join(contexts)
        response = await self._client.responses.create(
            model=self._model,
            instructions=(
                'Du bist ein präziser RAG-Assistent. '
                'Beantworte nur mit Hilfe des gegebenen Kontexts. '
                'Wenn die Information fehlt, sage klar, dass die Datenbasis nicht ausreicht.'
            ),
            input=(
                f'Frage:\n{question}\n\n'
                f'Kontext:\n{context_block}\n\n'
                'Gib eine knappe, sachliche Antwort auf Deutsch.'
            ),
        )
        return response.output_text.strip()
