"""Helpers for persona-scoped document upload and deletion.

Managed personas each own a Qdrant collection named ``persona_<24 hex chars>``.
These helpers keep that naming rule in one place so upload, search and delete
cannot target the shared Agata or Załožba collections by accident.
"""

from __future__ import annotations

import hashlib
import re

PERSONA_COLLECTION_RE = re.compile(r'^persona_[a-f0-9]{24}$')
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

_PDF_TYPES = {'application/pdf', 'application/x-pdf'}
_HTML_TYPES = {'text/html', 'application/xhtml+xml'}


class UnsupportedDocumentTypeError(ValueError):
    """Raised when an upload is not PDF or HTML."""


def is_persona_collection(name: str | None) -> bool:
    return bool(PERSONA_COLLECTION_RE.match((name or '').strip()))


def classify_upload(filename: str, content_type: str | None) -> str:
    """Return ``pdf`` or ``html``. Anything else is rejected."""
    lowered_name = (filename or '').lower().strip()
    lowered_type = (content_type or '').split(';', 1)[0].strip().lower()

    if lowered_name.endswith('.pdf') or lowered_type in _PDF_TYPES:
        if lowered_name.endswith(('.html', '.htm')):
            raise UnsupportedDocumentTypeError(filename or 'unknown')
        return 'pdf'
    if lowered_name.endswith(('.html', '.htm')) or lowered_type in _HTML_TYPES:
        return 'html'
    raise UnsupportedDocumentTypeError(filename or 'unknown')


def source_id_for_upload(kind: str, filename: str, raw: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(kind.encode('utf-8'))
    digest.update(b'\0')
    digest.update(filename.encode('utf-8'))
    digest.update(b'\0')
    digest.update(raw)
    return f'{kind}:{digest.hexdigest()}'
