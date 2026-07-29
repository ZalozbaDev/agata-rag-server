from __future__ import annotations

from io import BytesIO

import fitz

from app.models.schemas import ParsedSection


class InvalidPdfError(ValueError):
    pass


def _normalize_text(text: str) -> str:
    return ' '.join(text.split()).strip()


def _base_title(source_hint: str | None, page_count: int) -> str:
    hint = (source_hint or '').strip()
    if hint:
        return hint
    return 'document' if page_count == 1 else 'document'


def parse_pdf_content(
    data: bytes,
    *,
    source_hint: str | None = None,
    min_chars: int = 40,
) -> list[ParsedSection]:
    if not data:
        raise InvalidPdfError('Empty PDF file')

    try:
        document = fitz.open(stream=data, filetype='pdf')
    except Exception as exc:
        raise InvalidPdfError(f'Invalid or corrupted PDF: {exc}') from exc

    if document.is_encrypted and not document.authenticate(''):
        raise InvalidPdfError('Encrypted PDF requires a password')

    document_title = (document.metadata.get('title') or '').strip()
    base_title = document_title or _base_title(source_hint, document.page_count)
    sections: list[ParsedSection] = []

    for page_number in range(document.page_count):
        page = document[page_number]
        text = _normalize_text(page.get_text('text') or '')
        if len(text) < min_chars:
            continue

        title = base_title if document.page_count == 1 else f'{base_title} - Page {page_number + 1}'
        sections.append(ParsedSection(title=title, text=text))

    document.close()
    return sections
