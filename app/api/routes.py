from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.datastructures import UploadFile

from app.api.dependencies import get_container
from app.core.container import ServiceContainer
from app.core.errors import ProviderError
from app.models.schemas import AskRequest, AskResponse, HealthResponse, ParsedSection, ParseUrlRequest
from app.parsers.adapters import FetchError, InvalidUrlError
from app.parsers.pdf_parser import InvalidPdfError
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


def _is_pdf_upload(upload: UploadFile) -> bool:
    content_type = (upload.content_type or '').lower()
    filename = (upload.filename or '').lower()
    return content_type == 'application/pdf' or filename.endswith('.pdf')


async def _parse_pdf_request(
    request: Request,
) -> tuple[list[tuple[bytes, str | None]], int, bool]:
    content_type = request.headers.get('content-type', '')
    if 'multipart/form-data' not in content_type:
        raise HTTPException(
            status_code=400,
            detail='Expected multipart/form-data with one or more PDF files',
        )

    form = await request.form()
    uploads: list[UploadFile] = []
    seen: set[int] = set()

    for key in ('files', 'file', 'uploads'):
        for item in form.getlist(key):
            if isinstance(item, UploadFile) and id(item) not in seen:
                seen.add(id(item))
                uploads.append(item)

    if not uploads:
        for value in form.values():
            if isinstance(value, UploadFile) and id(value) not in seen:
                seen.add(id(value))
                uploads.append(value)

    if not uploads:
        raise HTTPException(status_code=400, detail='Missing PDF file(s)')

    min_chars_candidate = form.get('min_chars')
    if min_chars_candidate in (None, ''):
        min_chars_candidate = request.query_params.get('min_chars')
    min_chars = _extract_min_chars(min_chars_candidate)

    store_in_db_candidate = form.get('store_in_db')
    if store_in_db_candidate in (None, ''):
        store_in_db_candidate = request.query_params.get('store_in_db')
    store_in_db = _extract_store_in_db(store_in_db_candidate)

    pdf_files: list[tuple[bytes, str | None]] = []
    for upload in uploads:
        if not _is_pdf_upload(upload):
            raise HTTPException(
                status_code=400,
                detail=f'Unsupported file type: {upload.filename or "unknown"}',
            )
        raw = await upload.read()
        pdf_files.append((raw, upload.filename))

    return pdf_files, min_chars, store_in_db


@router.post('/parsePdf', response_model=list[ParsedSection])
async def parse_pdf(
    request: Request,
    container: ServiceContainer = Depends(get_container),
) -> list[ParsedSection]:
    pdf_files, min_chars, store_in_db = await _parse_pdf_request(request)
    sections: list[ParsedSection] = []

    for raw, file_name in pdf_files:
        try:
            file_sections = await container.parser_service.parse_pdf(
                raw,
                source_hint=file_name,
                min_chars=min_chars,
            )
        except InvalidPdfError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if store_in_db:
            source_seed = file_name or raw[:256]
            source_id = f'pdf:{stable_sha256(source_seed + raw.hex())}'
            await container.indexing_service.store_sections(
                source_id=source_id,
                source_type='pdf_upload',
                sections=file_sections,
                source_url=file_name,
            )

        sections.extend(file_sections)

    return sections


@router.post('/parseHtml', response_model=list[ParsedSection])
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
    try:
        return await container.rag_service.answer(
            request.question,
            history=request.history,
            is_phone_call=request.is_phone_call,
        )
    except ProviderError as exc:
        status = 504 if exc.is_timeout else 502
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail='Unexpected server error') from exc