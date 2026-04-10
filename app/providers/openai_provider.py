from __future__ import annotations

from typing import Any, Mapping, Sequence

from openai import AsyncOpenAI

from app.core.config import Settings


def _to_plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, 'model_dump'):
        dumped = value.model_dump()
        if isinstance(dumped, dict):
            return dumped
    if isinstance(value, Mapping):
        return dict(value)
    return {}


class OpenAIEmbeddingProvider:
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError('OPENAI_API_KEY fehlt.')

        client_kwargs: dict[str, Any] = {
            'api_key': settings.openai_api_key,
        }
        if settings.openai_base_url:
            client_kwargs['base_url'] = settings.openai_base_url

        self._client = AsyncOpenAI(**client_kwargs)
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

        client_kwargs: dict[str, Any] = {
            'api_key': settings.openai_api_key,
        }
        if settings.openai_base_url:
            client_kwargs['base_url'] = settings.openai_base_url

        self._client = AsyncOpenAI(**client_kwargs)
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

    async def answer_with_web_search(self, question: str) -> dict[str, object]:
        response = await self._client.responses.create(
            model=self._model,
            tools=[
                {
                    'type': 'web_search',
                    'search_context_size': 'medium',
                }
            ],
            include=['web_search_call.action.sources'],
            instructions=(
                'Beantworte die Frage mit Websuche. '
                'Antworte knapp, sachlich und auf Deutsch.'
            ),
            input=question,
        )

        sources: list[dict[str, str]] = []
        output_items = getattr(response, 'output', None) or []

        for item in output_items:
            item_dict = _to_plain_dict(item)
            item_type = str(item_dict.get('type') or getattr(item, 'type', '')).strip()

            if item_type != 'web_search_call':
                continue

            action = item_dict.get('action')
            action_dict = _to_plain_dict(action)

            raw_sources = action_dict.get('sources')
            if raw_sources is None and hasattr(item, 'action'):
                raw_sources = getattr(getattr(item, 'action'), 'sources', None)

            for src in raw_sources or []:
                src_dict = _to_plain_dict(src)
                url = str(src_dict.get('url', '')).strip()
                title = str(src_dict.get('title', '')).strip()

                if not url:
                    continue

                sources.append(
                    {
                        'source_type': 'web',
                        'source_url': url,
                        'title': title,
                    }
                )

        deduped_sources: list[dict[str, str]] = []
        seen_urls: set[str] = set()

        for src in sources:
            url = src['source_url']
            if url in seen_urls:
                continue
            seen_urls.add(url)
            deduped_sources.append(src)

        return {
            'answer': response.output_text.strip(),
            'sources': deduped_sources,
        }