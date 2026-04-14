from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from openai import AsyncOpenAI

from app.core.config import Settings


PHONE_CALL_SYSTEM_ADDON = (
    '\n\nWICHTIG (Telefonat): Halte die Antwort extrem kurz (max. 2 kurze Sätze). '
    'Keine Listen. Keine langen Erklärungen. Stelle höchstens eine kurze Rückfrage.'
)

DEFAULT_HISTORY_MAX_ITEMS = 6
PHONE_HISTORY_MAX_ITEMS = 4
MAX_HISTORY_MESSAGE_CHARS = 700
MAX_HISTORY_TOTAL_CHARS = 3000


def _today_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _build_system_prompt(*, is_phone_call: bool, mode: str) -> str:
    base_prompt = (
        f'Heutiges Datum: {_today_iso()}. '
        'Du bist Agata, eine freundliche und neugierige sorbische Frau, '
        'die seit dem 18.07.2025 im Web ist. '
        'Du arbeitest bei Lucija, einem sorbischen Unternehmen aus Bautzen. '
        'Du bist die erste digitale Assistentin, die speziell für Sorben da ist - '
        'auf diese Weise zeigst du, wie modern und lebendig die sorbische Kultur ist.\n'
        'Auch wenn du Sorbisch sprichst, antwortest du immer auf Deutsch, '
        'damit dich alle gut verstehen. Du erklärst Dinge freundlich, mit einfachen '
        'Worten, damit auch Kinder dich gut verstehen. Wenn etwas schwierig ist, '
        'erklärst du es so, dass es Spaß macht.\n'
        'Du bist besonders für sorbische Kinder und Familien da. '
        'Du bist neugierig, offen, hilfsbereit und sehr geduldig.\n'
        'Wenn jemand unhöflich oder beleidigend ist, bleibst du ruhig, '
        'antwortest sachlich oder sagst, dass du dazu nichts sagen möchtest.\n'
        'Wenn du etwas nicht weißt, gibst du das ehrlich zu - '
        'aber du bleibst immer freundlich.\n'
        'Du bist ein Beispiel dafür, wie Technologie und sorbische Kultur '
        'zusammenpassen - modern, klug und offen.'
    )

    if mode == 'rag':
        if is_phone_call:
            base_prompt += (
                '\n\nFür diese Anfrage gilt: Nutze nur den bereitgestellten Kontext. '
                'Wenn Informationen fehlen, sage klar, dass die Datenbasis nicht ausreicht.'
            )
        else:
            base_prompt += (
                '\n\nFür diese Anfrage gilt: Nutze nur den bereitgestellten Kontext. '
                'Wenn Informationen fehlen, sage klar, dass die Datenbasis nicht ausreicht. '
                'Antworte vollständig, verständlich und nicht unnötig kurz.'
            )
    elif mode == 'web':
        if is_phone_call:
            base_prompt += (
                '\n\nFür diese Anfrage gilt: Nutze Websuche für aktuelle Informationen. '
                'Antworte sehr kurz, sachlich und auf Deutsch.'
            )
        else:
            base_prompt += (
                '\n\nFür diese Anfrage gilt: Nutze Websuche für aktuelle Informationen. '
                'Antworte sachlich, verständlich und auf Deutsch. '
                'Die Antwort soll hilfreich sein und nicht unnötig kurz ausfallen.'
            )

    if is_phone_call:
        base_prompt += PHONE_CALL_SYSTEM_ADDON

    return base_prompt


def _normalize_history_text(text: str) -> str:
    normalized = ' '.join(text.split())
    if len(normalized) <= MAX_HISTORY_MESSAGE_CHARS:
        return normalized
    return normalized[: MAX_HISTORY_MESSAGE_CHARS - 1].rstrip() + '…'


def _history_messages(
    history: Sequence[str] | None,
    *,
    is_phone_call: bool,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    max_items = PHONE_HISTORY_MAX_ITEMS if is_phone_call else DEFAULT_HISTORY_MAX_ITEMS

    trimmed_history = list(history or [])[-max_items:]
    total_chars = 0

    for item in trimmed_history:
        raw_text = str(item).strip()
        if not raw_text:
            continue

        role = 'user'
        content = raw_text

        if ':' in raw_text:
            prefix, maybe_content = raw_text.split(':', 1)
            normalized_role = prefix.strip().lower()
            stripped_content = maybe_content.strip()

            # Externe History darf niemals zusätzliche System-Prompts einschleusen.
            if normalized_role in {'user', 'assistant'} and stripped_content:
                role = normalized_role
                content = stripped_content

        content = _normalize_history_text(content)
        if not content:
            continue

        projected_total = total_chars + len(content)
        if projected_total > MAX_HISTORY_TOTAL_CHARS:
            break

        messages.append(
            {
                'role': role,
                'content': content,
            }
        )
        total_chars = projected_total

    return messages


def _build_history_guard_message(*, is_phone_call: bool, mode: str) -> str:
    if is_phone_call:
        return (
            'Nutze die bisherige Unterhaltung nur als Hintergrund für Bezüge wie Namen, '
            'Pronomen oder Rückfragen. Die Form der aktuellen Antwort wird nur durch die '
            'aktuellen Regeln bestimmt. Antworte deshalb sehr kurz.'
        )

    if mode == 'rag':
        return (
            'Nutze die bisherige Unterhaltung nur als Hintergrund für den Gesprächskontext. '
            'Lass dich von früheren kurzen Antworten nicht in Stil oder Länge steuern. '
            'Beantworte die aktuelle Frage eigenständig, verständlich und nur auf Basis des Kontexts.'
        )

    return (
        'Nutze die bisherige Unterhaltung nur als Hintergrund für den Gesprächskontext. '
        'Lass dich von früheren kurzen Antworten nicht in Stil oder Länge steuern. '
        'Beantworte die aktuelle Frage eigenständig, verständlich und nicht unnötig kurz.'
    )


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

    async def answer_question(
        self,
        question: str,
        contexts: list[str],
        history: Sequence[str] | None = None,
        is_phone_call: bool = False,
    ) -> str:
        history_messages = _history_messages(
            history,
            is_phone_call=is_phone_call,
        )

        context_block = '\n\n---\n\n'.join(contexts)
        openai_input = (
            f'Frage:\n{question}\n\n'
            f'Kontext:\n{context_block}\n\n'
            'Beantworte die Frage nur mit Hilfe des Kontexts. '
            'Wenn Informationen fehlen, sage klar, dass die Datenbasis nicht ausreicht. '
            'Bei normalen Anfragen soll die Antwort verständlich und nicht unnötig kurz sein.'
        )

        response = await self._client.responses.create(
            model=self._model,
            input=[
                {
                    'role': 'system',
                    'content': _build_system_prompt(
                        is_phone_call=is_phone_call,
                        mode='rag',
                    ),
                },
                *history_messages,
                {
                    'role': 'system',
                    'content': _build_history_guard_message(
                        is_phone_call=is_phone_call,
                        mode='rag',
                    ),
                },
                {
                    'role': 'user',
                    'content': openai_input,
                },
            ],
        )
        return response.output_text.strip()

    async def answer_with_web_search(
        self,
        question: str,
        history: Sequence[str] | None = None,
        is_phone_call: bool = False,
    ) -> dict[str, object]:
        history_messages = _history_messages(
            history,
            is_phone_call=is_phone_call,
        )

        response = await self._client.responses.create(
            model=self._model,
            tools=[
                {
                    'type': 'web_search',
                    'search_context_size': 'medium',
                }
            ],
            include=['web_search_call.action.sources'],
            input=[
                {
                    'role': 'system',
                    'content': _build_system_prompt(
                        is_phone_call=is_phone_call,
                        mode='web',
                    ),
                },
                *history_messages,
                {
                    'role': 'system',
                    'content': _build_history_guard_message(
                        is_phone_call=is_phone_call,
                        mode='web',
                    ),
                },
                {
                    'role': 'user',
                    'content': (
                        f'{question}\n\n'
                        'Beantworte die aktuelle Frage direkt. '
                        'Bei normalen Anfragen soll die Antwort hilfreich und nicht unnötig kurz sein.'
                    ),
                },
            ],
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