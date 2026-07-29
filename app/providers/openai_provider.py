from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Literal, TypedDict, TypeVar

from openai import AsyncOpenAI

from app.core.config import Settings
from app.core.errors import OpenAIServiceError
from app.providers._provider_utils import is_timeout_error, is_transient_openai_error
from app.utils.retry import retry_async

T = TypeVar('T')
PromptMode = Literal['rag', 'web']

PHONE_CALL_SYSTEM_ADDON = (
    '\n\nWICHTIG (Telefonat): Halte die Antwort extrem kurz (max. 2 kurze Sätze). '
    'Keine Listen. Keine langen Erklärungen. Stelle höchstens eine kurze Rückfrage.'
)

DEFAULT_HISTORY_MAX_ITEMS = 6
PHONE_HISTORY_MAX_ITEMS = 4
MAX_HISTORY_MESSAGE_CHARS = 700
MAX_HISTORY_TOTAL_CHARS = 3000


class WebSource(TypedDict):
    source_type: str
    source_url: str
    title: str


class WebSearchResult(TypedDict):
    answer: str
    sources: list[WebSource]


class ChatMessage(TypedDict):
    role: str
    content: str


def _today_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _create_async_client(settings: Settings) -> AsyncOpenAI:
    client_kwargs: dict[str, Any] = {
        'api_key': settings.openai_api_key,
        'timeout': settings.openai_timeout_seconds,
    }
    if settings.openai_base_url:
        client_kwargs['base_url'] = settings.openai_base_url
    return AsyncOpenAI(**client_kwargs)


async def _run_with_openai_retry(
    call: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    retry_base_delay: float,
    error_message: str,
) -> T:
    try:
        return await retry_async(
            call,
            max_attempts=max_retries,
            base_delay=retry_base_delay,
            retryable=is_transient_openai_error,
        )
    except Exception as exc:
        raise OpenAIServiceError(
            f'{error_message}: {exc}',
            is_timeout=is_timeout_error(exc),
        ) from exc


def _require_api_key(settings: Settings) -> None:
    if not settings.openai_api_key:
        raise ValueError('OPENAI_API_KEY fehlt.')


def _build_system_prompt(*, is_phone_call: bool, mode: PromptMode) -> str:
    """Persona + mode-specific grounding; phone mode always appends brevity rules."""
    prompt = (
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
        prompt += (
            '\n\nFür diese Anfrage gelten zwingende Regeln:\n'
            '- Nutze ausschließlich den bereitgestellten Kontext.\n'
            '- Erfinde keine Fakten, Namen, Daten, Orte, Zahlen oder URLs.\n'
            '- Wenn der Kontext die Frage nicht vollständig beantwortet, sage klar, '
            'dass die Datenbasis nicht ausreicht.\n'
            '- Wenn du unsicher bist, sage das offen und bleibe bei dem, was im Kontext steht.\n'
            '- Der Kontext kann aus dem Obersorbischen stammen; verstehe ihn sachlich, '
            'antworte aber auf Deutsch.\n'
            + (
                '- Halte die Antwort sehr kurz.'
                if is_phone_call
                else '- Antworte vollständig, verständlich und nicht unnötig kurz.'
            )
        )
    elif mode == 'web':
        length_hint = (
            'Antworte sehr kurz, sachlich und auf Deutsch.'
            if is_phone_call
            else (
                'Antworte sachlich, verständlich und auf Deutsch. '
                'Die Antwort soll hilfreich sein und nicht unnötig kurz ausfallen.'
            )
        )
        prompt += (
            f'\n\nFür diese Anfrage gilt: Nutze Websuche für aktuelle Informationen. '
            f'{length_hint} Erfinde keine Fakten.'
        )

    if is_phone_call:
        prompt += PHONE_CALL_SYSTEM_ADDON

    return prompt


def _normalize_history_text(text: str) -> str:
    normalized = ' '.join(text.split())
    if len(normalized) <= MAX_HISTORY_MESSAGE_CHARS:
        return normalized
    return normalized[: MAX_HISTORY_MESSAGE_CHARS - 1].rstrip() + '…'


def _parse_history_item(raw_text: str) -> ChatMessage | None:
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
        return None
    return {'role': role, 'content': content}


def _history_messages(
    history: Sequence[str] | None,
    *,
    is_phone_call: bool,
) -> list[ChatMessage]:
    """Cap history by item count and total chars so phone calls stay within context budget."""
    max_items = PHONE_HISTORY_MAX_ITEMS if is_phone_call else DEFAULT_HISTORY_MAX_ITEMS
    messages: list[ChatMessage] = []
    total_chars = 0

    for item in list(history or [])[-max_items:]:
        raw_text = str(item).strip()
        if not raw_text:
            continue

        message = _parse_history_item(raw_text)
        if message is None:
            continue

        projected_total = total_chars + len(message['content'])
        if projected_total > MAX_HISTORY_TOTAL_CHARS:
            break

        messages.append(message)
        total_chars = projected_total

    return messages


def _build_history_guard_message(*, is_phone_call: bool, mode: PromptMode) -> str:
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
    blocks = [
        f'[Kontext {index}]\n{context.strip()}'
        for index, context in enumerate(contexts, start=1)
    ]
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


def _dedupe_sources_by_url(sources: Sequence[WebSource]) -> list[WebSource]:
    deduped: list[WebSource] = []
    seen_urls: set[str] = set()
    for source in sources:
        url = source['source_url']
        if url in seen_urls:
            continue
        seen_urls.add(url)
        deduped.append(source)
    return deduped


def _extract_web_sources(response: Any) -> list[WebSource]:
    """Pull URLs from OpenAI web_search_call items; SDK shapes vary between versions."""
    sources: list[WebSource] = []
    output_items = getattr(response, 'output', None) or []

    for item in output_items:
        item_dict = _to_plain_dict(item)
        item_type = str(item_dict.get('type') or getattr(item, 'type', '')).strip()
        if item_type != 'web_search_call':
            continue

        action_dict = _to_plain_dict(item_dict.get('action'))
        raw_sources = action_dict.get('sources')
        if raw_sources is None and hasattr(item, 'action'):
            raw_sources = getattr(item.action, 'sources', None)

        for src in raw_sources or []:
            src_dict = _to_plain_dict(src)
            url = str(src_dict.get('url', '')).strip()
            if not url:
                continue
            sources.append(
                {
                    'source_type': 'web',
                    'source_url': url,
                    'title': str(src_dict.get('title', '')).strip(),
                }
            )

    return _dedupe_sources_by_url(sources)


def _build_rag_user_prompt(question: str, contexts: Sequence[str], *, is_phone_call: bool) -> str:
    prompt = (
        f'Frage:\n{question.strip()}\n\n'
        f'Kontext:\n{_format_numbered_contexts(contexts)}\n\n'
        'Beantworte die Frage ausschließlich mit Hilfe der nummerierten Kontextblöcke.\n'
        '- Erfinde keine Informationen, die nicht im Kontext stehen.\n'
        '- Wenn der Kontext nicht ausreicht, sage klar: "Die Datenbasis reicht nicht aus."\n'
        '- Nenne keine Quellen oder URLs, die nicht im Kontext vorkommen.\n'
        '- Antworte auf Deutsch.'
    )
    if not is_phone_call:
        prompt += '\n- Die Antwort soll verständlich und nicht unnötig kurz sein.'
    return prompt


def _chat_input(
    *,
    is_phone_call: bool,
    mode: PromptMode,
    history: Sequence[str] | None,
    user_content: str,
) -> list[dict[str, str]]:
    return [
        {
            'role': 'system',
            'content': _build_system_prompt(is_phone_call=is_phone_call, mode=mode),
        },
        *_history_messages(history, is_phone_call=is_phone_call),
        {
            'role': 'system',
            'content': _build_history_guard_message(is_phone_call=is_phone_call, mode=mode),
        },
        {
            'role': 'user',
            'content': user_content,
        },
    ]


class OpenAIEmbeddingProvider:
    def __init__(self, settings: Settings) -> None:
        _require_api_key(settings)
        self._client = _create_async_client(settings)
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

        return await _run_with_openai_retry(
            _call,
            max_retries=self._max_retries,
            retry_base_delay=self._retry_base_delay,
            error_message='OpenAI embedding request failed',
        )

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self.embed_texts([text])
        return vectors[0]


class OpenAILLMProvider:
    def __init__(self, settings: Settings) -> None:
        _require_api_key(settings)
        self._client = _create_async_client(settings)
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
        async def _call() -> str:
            response = await self._client.responses.create(
                model=self._model,
                input=_chat_input(
                    is_phone_call=is_phone_call,
                    mode='rag',
                    history=history,
                    user_content=_build_rag_user_prompt(
                        question,
                        contexts,
                        is_phone_call=is_phone_call,
                    ),
                ),
            )
            return response.output_text.strip()

        return await _run_with_openai_retry(
            _call,
            max_retries=self._max_retries,
            retry_base_delay=self._retry_base_delay,
            error_message='OpenAI chat request failed',
        )

    async def answer_with_web_search(
        self,
        question: str,
        history: Sequence[str] | None = None,
        is_phone_call: bool = False,
    ) -> WebSearchResult:
        user_content = (
            f'{question}\n\n'
            'Beantworte die aktuelle Frage direkt auf Deutsch. '
            'Erfinde keine Fakten. '
            'Bei normalen Anfragen soll die Antwort hilfreich und nicht unnötig kurz sein.'
        )

        async def _call() -> WebSearchResult:
            response = await self._client.responses.create(
                model=self._model,
                tools=[
                    {
                        'type': 'web_search',
                        'search_context_size': 'medium',
                    }
                ],
                include=['web_search_call.action.sources'],
                input=_chat_input(
                    is_phone_call=is_phone_call,
                    mode='web',
                    history=history,
                    user_content=user_content,
                ),
            )
            return {
                'answer': response.output_text.strip(),
                'sources': _extract_web_sources(response),
            }

        return await _run_with_openai_retry(
            _call,
            max_retries=self._max_retries,
            retry_base_delay=self._retry_base_delay,
            error_message='OpenAI web search request failed',
        )
