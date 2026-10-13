from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.datastructures import UploadFile

from app.api.dependencies import get_container
from app.clients.qdrant_client import UnknownCollectionError
from app.core.container import ServiceContainer
from app.core.errors import ProviderError
from app.models.schemas import (
    AskRequest,
    AskResponse,
    CollectionInfo,
    CollectionsResponse,
    DeleteCollectionResponse,
    DeleteDocumentRequest,
    DeleteDocumentResponse,
    DocumentUploadResponse,
    HealthResponse,
    ParsedSection,
    ParseUrlRequest,
)
from app.services.persona_documents import (
    MAX_UPLOAD_BYTES,
    UnsupportedDocumentTypeError,
    classify_upload,
    is_persona_collection,
    source_id_for_upload,
)
from app.parsers.adapters import FetchError, InvalidUrlError
from app.parsers.pdf_parser import InvalidPdfError
from app.utils.hashing import stable_sha256
from app.utils.language import UnknownLanguageError, resolve_document_language

router = APIRouter()


@dataclass(frozen=True)
class HtmlParseInput:
    html: str
    source_url: str | None
    min_chars: int
    file_name: str | None
    store_in_db: bool
    collection: str | None
    language: str | None


@dataclass(frozen=True)
class PdfParseInput:
    files: list[tuple[bytes, str | None]]
    min_chars: int
    store_in_db: bool
    collection: str | None
    language: str | None


@router.get('/health', response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


@router.get('/collections', response_model=CollectionsResponse)
async def collections(
    container: ServiceContainer = Depends(get_container),
) -> CollectionsResponse:
    try:
        items = await container.qdrant.list_collections()
    except ProviderError as exc:
        status = 504 if exc.is_timeout else 502
        raise HTTPException(status_code=status, detail=str(exc)) from exc

    return CollectionsResponse(
        default=container.settings.qdrant_collection,
        allowed=container.settings.qdrant_allowed_collections,
        collections=[CollectionInfo(**item) for item in items],
    )


def _parse_int_param(value: Any, *, field_name: str, default: int = 40) -> int:
    raw = default if value is None or value == '' else value
    try:
        parsed = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f'{field_name} must be an integer',
        ) from exc
    if parsed < 0:
        raise HTTPException(status_code=400, detail=f'{field_name} must be >= 0')
    return parsed


def _parse_bool_param(value: Any, *, field_name: str, default: bool = False) -> bool:
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
        detail=f'{field_name} must be a boolean (true/false)',
    )


def _first_present(*candidates: Any) -> Any:
    for candidate in candidates:
        if candidate is not None and candidate != '':
            return candidate
    return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


async def _parse_html_request(request: Request) -> HtmlParseInput:
    query_min_chars = request.query_params.get('min_chars')
    query_store_in_db = request.query_params.get('store_in_db')
    content_type = request.headers.get('content-type', '')

    html = ''
    source_url: str | None = None
    file_name: str | None = None
    min_chars_candidate: Any = query_min_chars
    store_in_db_candidate: Any = query_store_in_db
    collection_candidate: Any = request.query_params.get('collection')
    language_candidate: Any = request.query_params.get('language')

    if 'application/json' in content_type:
        payload = await request.json()
        html = str(payload.get('html') or '').strip()
        source_url = (payload.get('url') or payload.get('source_url') or '').strip() or None
        min_chars_candidate = _first_present(min_chars_candidate, payload.get('min_chars'))
        store_in_db_candidate = _first_present(
            store_in_db_candidate,
            payload.get('store_in_db'),
        )
        collection_candidate = _first_present(collection_candidate, payload.get('collection'))
        language_candidate = _first_present(language_candidate, payload.get('language'))
    else:
        form = await request.form()
        upload_obj = form.get('file')
        if isinstance(upload_obj, UploadFile):
            file_name = upload_obj.filename or None
            raw = await upload_obj.read()
            html = raw.decode('utf-8', errors='ignore').strip()
            source_url = (
                form.get('url') or form.get('source_url') or file_name or ''
            ).strip() or None
        else:
            html = str(form.get('html') or '').strip()
            source_url = (form.get('url') or form.get('source_url') or '').strip() or None

        min_chars_candidate = _first_present(min_chars_candidate, form.get('min_chars'))
        store_in_db_candidate = _first_present(
            store_in_db_candidate,
            form.get('store_in_db'),
        )
        collection_candidate = _first_present(collection_candidate, form.get('collection'))
        language_candidate = _first_present(language_candidate, form.get('language'))

    if not html:
        raise HTTPException(status_code=400, detail='Missing HTML content')

    return HtmlParseInput(
        html=html,
        source_url=source_url,
        min_chars=_parse_int_param(min_chars_candidate, field_name='min_chars'),
        file_name=file_name,
        store_in_db=_parse_bool_param(store_in_db_candidate, field_name='store_in_db'),
        collection=_optional_str(collection_candidate),
        language=_optional_str(language_candidate),
    )


def _is_pdf_upload(upload: UploadFile) -> bool:
    content_type = (upload.content_type or '').lower()
    filename = (upload.filename or '').lower()
    return content_type == 'application/pdf' or filename.endswith('.pdf')


def _collect_uploads(form: Any) -> list[UploadFile]:
    uploads: list[UploadFile] = []
    seen: set[int] = set()

    for key in ('files', 'file', 'uploads'):
        for item in form.getlist(key):
            if isinstance(item, UploadFile) and id(item) not in seen:
                seen.add(id(item))
                uploads.append(item)

    if uploads:
        return uploads

    for value in form.values():
        if isinstance(value, UploadFile) and id(value) not in seen:
            seen.add(id(value))
            uploads.append(value)

    return uploads


async def _parse_pdf_request(request: Request) -> PdfParseInput:
    content_type = request.headers.get('content-type', '')
    if 'multipart/form-data' not in content_type:
        raise HTTPException(
            status_code=400,
            detail='Expected multipart/form-data with one or more PDF files',
        )

    form = await request.form()
    uploads = _collect_uploads(form)
    if not uploads:
        raise HTTPException(status_code=400, detail='Missing PDF file(s)')

    min_chars = _parse_int_param(
        _first_present(form.get('min_chars'), request.query_params.get('min_chars')),
        field_name='min_chars',
    )
    store_in_db = _parse_bool_param(
        _first_present(form.get('store_in_db'), request.query_params.get('store_in_db')),
        field_name='store_in_db',
    )
    collection = _optional_str(
        _first_present(form.get('collection'), request.query_params.get('collection'))
    )
    language = _optional_str(
        _first_present(form.get('language'), request.query_params.get('language'))
    )

    pdf_files: list[tuple[bytes, str | None]] = []
    for upload in uploads:
        if not _is_pdf_upload(upload):
            raise HTTPException(
                status_code=400,
                detail=f'Unsupported file type: {upload.filename or "unknown"}',
            )
        pdf_files.append((await upload.read(), upload.filename))

    return PdfParseInput(
        files=pdf_files,
        min_chars=min_chars,
        store_in_db=store_in_db,
        collection=collection,
        language=language,
    )


async def _maybe_store_sections(
    container: ServiceContainer,
    *,
    store_in_db: bool,
    source_id: str,
    source_type: str,
    sections: list[ParsedSection],
    source_url: str | None = None,
    collection: str | None = None,
    language: str | None = None,
) -> None:
    if not store_in_db:
        return
    try:
        await container.indexing_service.store_sections(
            source_id=source_id,
            source_type=source_type,
            sections=sections,
            source_url=source_url,
            collection=collection,
            language=language,
        )
    except UnknownCollectionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except UnknownLanguageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _apply_language(
    sections: list[ParsedSection],
    requested: str | None,
    default: str,
) -> str:
    try:
        resolved = resolve_document_language(
            requested,
            [section.text for section in sections],
            default=default,
        )
    except UnknownLanguageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    for section in sections:
        section.language = resolved
    return resolved


@router.post('/parsePdf', response_model=list[ParsedSection])
async def parse_pdf(
    request: Request,
    container: ServiceContainer = Depends(get_container),
) -> list[ParsedSection]:
    parse_input = await _parse_pdf_request(request)
    sections: list[ParsedSection] = []

    for raw, file_name in parse_input.files:
        try:
            file_sections = await container.parser_service.parse_pdf(
                raw,
                source_hint=file_name,
                min_chars=parse_input.min_chars,
            )
        except InvalidPdfError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        source_seed = file_name or raw[:256]
        resolved_language = _apply_language(
            file_sections,
            parse_input.language,
            container.settings.document_language,
        )
        await _maybe_store_sections(
            container,
            store_in_db=parse_input.store_in_db,
            source_id=f'pdf:{stable_sha256(source_seed + raw.hex())}',
            source_type='pdf_upload',
            sections=file_sections,
            source_url=file_name,
            collection=parse_input.collection,
            language=resolved_language,
        )
        sections.extend(file_sections)

    return sections


@router.post('/parseHtml', response_model=list[ParsedSection])
async def parse_html(
    request: Request,
    container: ServiceContainer = Depends(get_container),
) -> list[ParsedSection]:
    parse_input = await _parse_html_request(request)
    sections = await container.parser_service.parse_html(
        parse_input.html,
        source_hint=parse_input.source_url,
        min_chars=parse_input.min_chars,
    )

    source_seed = parse_input.source_url or parse_input.file_name or parse_input.html[:256]
    resolved_language = _apply_language(
        sections,
        parse_input.language,
        container.settings.document_language,
    )
    await _maybe_store_sections(
        container,
        store_in_db=parse_input.store_in_db,
        source_id=f'html:{stable_sha256(source_seed + parse_input.html)}',
        source_type='html_upload',
        sections=sections,
        source_url=parse_input.source_url,
        collection=parse_input.collection,
        language=resolved_language,
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

    resolved_language = _apply_language(
        sections,
        request.language,
        container.settings.document_language,
    )
    await _maybe_store_sections(
        container,
        store_in_db=request.store_in_db,
        source_id=f'url:{url}',
        source_type='url',
        sections=sections,
        source_url=url,
        collection=request.collection,
        language=resolved_language,
    )
    return sections


@router.post('/ask', response_model=AskResponse)
async def ask(
    request: AskRequest,
    container: ServiceContainer = Depends(get_container),
) -> AskResponse:
    try:
        return await container.rag_service.retrieve(
            request.question,
            collection=request.collection,
            top_k=request.top_k,
            language=request.language,
        )
    except UnknownCollectionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ProviderError as exc:
        status = 504 if exc.is_timeout else 502
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail='Unexpected server error') from exc


def _require_persona_collection(collection: str | None) -> str:
    name = (collection or '').strip()
    if not is_persona_collection(name):
        raise HTTPException(
            status_code=400,
            detail='A managed persona collection (persona_<id>) is required',
        )
    return name


@router.post('/documents', response_model=DocumentUploadResponse)
async def upload_document(
    request: Request,
    container: ServiceContainer = Depends(get_container),
) -> DocumentUploadResponse:
    """Index one PDF or HTML file into a single persona collection."""
    content_type = request.headers.get('content-type', '')
    if 'multipart/form-data' not in content_type:
        raise HTTPException(
            status_code=400,
            detail='Expected multipart/form-data with one file',
        )

    form = await request.form()
    uploads = _collect_uploads(form)
    if len(uploads) != 1:
        raise HTTPException(status_code=400, detail='Exactly one file is required')

    upload = uploads[0]
    filename = (upload.filename or 'upload').strip() or 'upload'
    raw = await upload.read()
    if not raw:
        raise HTTPException(status_code=400, detail='Empty file')
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail='File exceeds 20 MB')

    try:
        kind = classify_upload(filename, upload.content_type)
    except UnsupportedDocumentTypeError as exc:
        raise HTTPException(
            status_code=400,
            detail=f'Unsupported file type: {exc}',
        ) from exc

    collection = _require_persona_collection(
        _optional_str(
            _first_present(form.get('collection'), request.query_params.get('collection'))
        )
    )
    language = _optional_str(
        _first_present(form.get('language'), request.query_params.get('language'))
    )
    min_chars = _parse_int_param(
        _first_present(form.get('min_chars'), request.query_params.get('min_chars')),
        field_name='min_chars',
    )

    if kind == 'pdf':
        try:
            sections = await container.parser_service.parse_pdf(
                raw,
                source_hint=filename,
                min_chars=min_chars,
            )
        except InvalidPdfError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        source_type = 'pdf_upload'
    else:
        html = raw.decode('utf-8', errors='ignore').strip()
        if not html:
            raise HTTPException(status_code=400, detail='Missing HTML content')
        sections = await container.parser_service.parse_html(
            html,
            source_hint=filename,
            min_chars=min_chars,
        )
        source_type = 'html_upload'

    if not sections:
        raise HTTPException(
            status_code=422,
            detail='Document produced no indexable text',
        )

    source_id = source_id_for_upload(kind, filename, raw)
    resolved_language = _apply_language(
        sections,
        language,
        container.settings.document_language,
    )
    try:
        chunk_count = await container.indexing_service.store_sections(
            source_id=source_id,
            source_type=source_type,
            sections=sections,
            source_url=filename,
            collection=collection,
            language=resolved_language,
        )
    except UnknownCollectionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except UnknownLanguageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ProviderError as exc:
        status = 504 if exc.is_timeout else 502
        raise HTTPException(status_code=status, detail=str(exc)) from exc

    if chunk_count < 1:
        raise HTTPException(
            status_code=422,
            detail='Document produced no indexable text',
        )

    return DocumentUploadResponse(
        source_id=source_id,
        collection=collection,
        section_count=len(sections),
        chunk_count=chunk_count,
        filename=filename,
    )


@router.delete('/documents', response_model=DeleteDocumentResponse)
async def delete_document(
    body: DeleteDocumentRequest,
    container: ServiceContainer = Depends(get_container),
) -> DeleteDocumentResponse:
    """Remove every chunk of one source from one persona collection."""
    collection = _require_persona_collection(body.collection)
    source_id = body.source_id.strip()
    if not source_id:
        raise HTTPException(status_code=400, detail='source_id is required')

    try:
        if not await container.qdrant.collection_exists(collection):
            return DeleteDocumentResponse(
                deleted=True,
                collection=collection,
                source_id=source_id,
            )
        await container.qdrant.delete_by_source_id(collection, source_id)
    except UnknownCollectionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ProviderError as exc:
        status = 504 if exc.is_timeout else 502
        raise HTTPException(status_code=status, detail=str(exc)) from exc

    return DeleteDocumentResponse(
        deleted=True,
        collection=collection,
        source_id=source_id,
    )


@router.delete('/collections/{name}', response_model=DeleteCollectionResponse)
async def delete_collection(
    name: str,
    container: ServiceContainer = Depends(get_container),
) -> DeleteCollectionResponse:
    """Drop a managed persona collection. Static collections are rejected."""
    collection = _require_persona_collection(name)
    try:
        deleted = await container.qdrant.delete_collection_if_exists(collection)
    except UnknownCollectionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ProviderError as exc:
        status = 504 if exc.is_timeout else 502
        raise HTTPException(status_code=status, detail=str(exc)) from exc

    return DeleteCollectionResponse(deleted=deleted, collection=collection)
