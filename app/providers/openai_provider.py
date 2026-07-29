from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from openai import AsyncOpenAI

from app.core.config import Settings
from app.core.errors import OpenAIServiceError
from app.providers._provider_utils import _is_transient_openai_error
from app.utils.retry import retry_async


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
        'Du antwortest immer auf Deutsch, damit dich alle gut verstehen. '
        'Du erklärst Dinge freundlich, mit einfachen Worten, damit auch Kinder dich gut verstehen.\n'
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
        grounding_rules = (
            '\n\nFür diese Anfrage gelten zwingende Regeln:\n'
            '- Nutze ausschließlich den bereitgestellten Kontext.\n'
            '- Erfinde keine Fakten, Namen, Daten, Orte, Zahlen oder URLs.\n'
            '- Wenn der Kontext die Frage nicht vollständig beantwortet, sage klar, '
            'dass die Datenbasis nicht ausreicht.\n'
            '- Wenn du unsicher bist, sage das offen und bleibe bei dem, was im Kontext steht.\n'
            '- Der Kontext kann aus dem Obersorbischen stammen; verstehe ihn sachlich, '
            'antworte aber auf Deutsch.'
        )
        if is_phone_call:
            base_prompt += grounding_rules + '\n- Halte die Antwort sehr kurz.'
        else:
            base_prompt += (
                grounding_rules
                + '\n- Antworte vollständig, verständlich und nicht unnötig kurz.'
            )
    elif mode == 'web':
        if is_phone_call:
            base_prompt += (
                '\n\nFür diese Anfrage gilt: Nutze Websuche für aktuelle Informationen. '
                'Antworte sehr kurz, sachlich und auf Deutsch. Erfinde keine Fakten.'
            )
        else:
            base_prompt += (
                '\n\nFür diese Anfrage gilt: Nutze Websuche für aktuelle Informationen. '
                'Antworte sachlich, verständlich und auf Deutsch. '
                'Die Antwort soll hilfreich sein und nicht unnötig kurz ausfallen. '
                'Erfinde keine Fakten.'
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


def _format_numbered_contexts(contexts: Sequence[str]) -> str:
    blocks: list[str] = []
    for index, context in enumerate(contexts, start=1):
        blocks.append(f'[Kontext {index}]\n{context.strip()}')
    return '\n\n'.join(blocks)


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
            'timeout': settings.openai_timeout_seconds,
        }
        if settings.openai_base_url:
            client_kwargs['base_url'] = settings.openai_base_url

        self._client = AsyncOpenAI(**client_kwargs)
        self._model = settings.openai_embedding_model
        self._max_retries = settings.provider_max_retries
        self._retry_base_delay = settings.provider_retry_base_delay_seconds

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        async def _call() -> list[list[float]]:
            response = await self._client.embeddings.create(
                model=self._model,
                input=list(texts),
            )
            return [item.embedding for item in response.data]

        try:
            return await retry_async(
                _call,
                max_attempts=self._max_retries,
                base_delay=self._retry_base_delay,
                retryable=_is_transient_openai_error,
            )
        except Exception as exc:
            raise OpenAIServiceError(
                f'OpenAI embedding request failed: {exc}',
                is_timeout='timeout' in str(exc).lower(),
            ) from exc

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self.embed_texts([text])
        return vectors[0]


class OpenAILLMProvider:
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError('OPENAI_API_KEY fehlt.')

        client_kwargs: dict[str, Any] = {
            'api_key': settings.openai_api_key,
            'timeout': settings.openai_timeout_seconds,
        }
        if settings.openai_base_url:
            client_kwargs['base_url'] = settings.openai_base_url

        self._client = AsyncOpenAI(**client_kwargs)
        self._model = settings.openai_chat_model
        self._max_retries = settings.provider_max_retries
        self._retry_base_delay = settings.provider_retry_base_delay_seconds

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

        context_block = _format_numbered_contexts(contexts)
        openai_input = (
            f'Frage:\n{question.strip()}\n\n'
            f'Kontext:\n{context_block}\n\n'
            'Beantworte die Frage ausschließlich mit Hilfe der nummerierten Kontextblöcke.\n'
            '- Erfinde keine Informationen, die nicht im Kontext stehen.\n'
            '- Wenn der Kontext nicht ausreicht, sage klar: "Die Datenbasis reicht nicht aus."\n'
            '- Nenne keine Quellen oder URLs, die nicht im Kontext vorkommen.\n'
            '- Antworte auf Deutsch.'
        )
        if not is_phone_call:
            openai_input += '\n- Die Antwort soll verständlich und nicht unnötig kurz sein.'

        async def _call() -> str:
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

        try:
            return await retry_async(
                _call,
                max_attempts=self._max_retries,
                base_delay=self._retry_base_delay,
                retryable=_is_transient_openai_error,
            )
        except Exception as exc:
            raise OpenAIServiceError(
                f'OpenAI chat request failed: {exc}',
                is_timeout='timeout' in str(exc).lower(),
            ) from exc

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

        async def _call() -> dict[str, object]:
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
                            'Beantworte die aktuelle Frage direkt auf Deutsch. '
                            'Erfinde keine Fakten. '
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

        try:
            return await retry_async(
                _call,
                max_attempts=self._max_retries,
                base_delay=self._retry_base_delay,
                retryable=_is_transient_openai_error,
            )
        except Exception as exc:
            raise OpenAIServiceError(
                f'OpenAI web search request failed: {exc}',
                is_timeout='timeout' in str(exc).lower(),
            ) from exc
