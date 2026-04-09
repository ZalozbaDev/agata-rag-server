from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, UploadFile

from app.api.dependencies import get_container
from app.core.container import ServiceContainer
from app.models.schemas import AskRequest, AskResponse, HealthResponse, ParsedSection, ParseUrlRequest
from app.parsers.adapters import FetchError, InvalidUrlError
from app.utils.hashing import stable_sha256

router = APIRouter()


@router.get('/health', response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


def _extract_min_chars(value: Any, default: int = 40) -> int:
    raw = default if value is None or value == '' else value
    try:
        parsed = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail='min_chars must be an integer') from exc
    if parsed < 0:
        raise HTTPException(status_code=400, detail='min_chars must be >= 0')
    return parsed


def _extract_store_in_db(value: Any, default: bool = False) -> bool:
    raw = default if value is None or value == '' else value
    if isinstance(raw, bool):
        return raw

    normalized = str(raw).strip().lower()
    if normalized in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'n', 'off'}:
        return False

    raise HTTPException(
        status_code=400,
        detail='store_in_db must be a boolean (true/false)',
    )


async def _parse_html_request(request: Request) -> tuple[str, str | None, int, str | None, bool]:
    query_min_chars = request.query_params.get('min_chars')
    content_type = request.headers.get('content-type', '')
    html = ''
    source_url: str | None = None
    file_name: str | None = None
    query_store_in_db = request.query_params.get('store_in_db')
    min_chars_candidate: Any = query_min_chars
    store_in_db_candidate: Any = query_store_in_db

    if 'application/json' in content_type:
        payload = await request.json()
        html = str(payload.get('html') or '').strip()
        source_url = (payload.get('url') or payload.get('source_url') or '').strip() or None
        if min_chars_candidate in (None, ''):
            min_chars_candidate = payload.get('min_chars')
        if store_in_db_candidate in (None, ''):
            store_in_db_candidate = payload.get('store_in_db')
    else:
        form = await request.form()
        upload_obj = form.get('file')
        if isinstance(upload_obj, UploadFile):
            file_name = upload_obj.filename or None
            raw = await upload_obj.read()
            html = raw.decode('utf-8', errors='ignore').strip()
            source_url = (form.get('url') or form.get('source_url') or file_name or '').strip() or None
        else:
            html = str(form.get('html') or '').strip()
            source_url = (form.get('url') or form.get('source_url') or '').strip() or None

        if min_chars_candidate in (None, ''):
            min_chars_candidate = form.get('min_chars')
        if store_in_db_candidate in (None, ''):
            store_in_db_candidate = form.get('store_in_db')

    if not html:
        raise HTTPException(status_code=400, detail='Missing HTML content')

    min_chars = _extract_min_chars(min_chars_candidate)
    store_in_db = _extract_store_in_db(store_in_db_candidate)
    return html, source_url, min_chars, file_name, store_in_db


@router.post('/parseHtml', response_model=list[ParsedSection])
@router.post('/parse_html', response_model=list[ParsedSection])
async def parse_html(
    request: Request,
    container: ServiceContainer = Depends(get_container),
) -> list[ParsedSection]:
    html, source_url, min_chars, file_name, store_in_db = await _parse_html_request(request)
    sections = await container.parser_service.parse_html(
        html,
        source_hint=source_url,
        min_chars=min_chars,
    )

    if store_in_db:
        source_seed = source_url or file_name or html[:256]
        source_id = f'html:{stable_sha256(source_seed + html)}'
        await container.indexing_service.store_sections(
            source_id=source_id,
            source_type='html_upload',
            sections=sections,
            source_url=source_url,
        )

    return sections


@router.get('/parse', response_model=list[ParsedSection])
@router.get('/parseUrl', response_model=list[ParsedSection])
async def parse_url_get(
    url: str = Query(...),
    min_chars: int = Query(default=40, ge=0),
    store_in_db: bool = Query(default=False),
    container: ServiceContainer = Depends(get_container),
) -> list[ParsedSection]:
    try:
        sections = await container.parser_service.parse_url(url, min_chars=min_chars)
    except InvalidUrlError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FetchError as exc:
        status = 504 if exc.is_timeout else 502
        raise HTTPException(status_code=status, detail=str(exc)) from exc

    if store_in_db:
        await container.indexing_service.store_sections(
            source_id=f'url:{url}',
            source_type='url',
            sections=sections,
            source_url=url,
        )
    return sections


@router.post('/parseUrl', response_model=list[ParsedSection])
async def parse_url_post(
    request: ParseUrlRequest,
    container: ServiceContainer = Depends(get_container),
) -> list[ParsedSection]:
    url = str(request.url)
    try:
        sections = await container.parser_service.parse_url(url, min_chars=request.min_chars)
    except InvalidUrlError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FetchError as exc:
        status = 504 if exc.is_timeout else 502
        raise HTTPException(status_code=status, detail=str(exc)) from exc

    if request.store_in_db:
        await container.indexing_service.store_sections(
            source_id=f'url:{url}',
            source_type='url',
            sections=sections,
            source_url=url,
        )

    return sections


@router.post('/ask', response_model=AskResponse)
async def ask(
    request: AskRequest,
    container: ServiceContainer = Depends(get_container),
) -> AskResponse:
    return await container.rag_service.answer(request.question)