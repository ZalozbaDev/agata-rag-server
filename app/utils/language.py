from __future__ import annotations

import re
from typing import Literal

QueryLanguage = Literal['de', 'hsb']

# Diacritics that occur in Upper Sorbian but not in German.
_HSB_CHARS = frozenset('łńśźćžščěřŕ')

# German umlauts/ß are a positive German signal; Sorbian does not use them.
_DE_CHARS = frozenset('äöüß')

_HSB_MARKERS = frozenset(
    {
        'wón',
        'wona',
        'wone',
        'chcy',
        'hdy',
        'hdźe',
        'nětko',
        'wob',
        'dźě',
        'serbski',
        'serbska',
        'serbšćina',
        'hornjoserbšćina',
        'rěč',
        'wutroba',
        'wobsah',
        'přichod',
        'přeco',
        'što',
        'štó',
        'kak',
        'prošu',
        'móže',
        'njeje',
        'kotry',
        'kotra',
        'wšě',
        'jako',
        'abo',
        'tež',
        'znaje',
        'lětsa',
    }
)

_DE_MARKERS = frozenset(
    {
        'der',
        'die',
        'das',
        'des',
        'dem',
        'den',
        'und',
        'oder',
        'nicht',
        'ist',
        'sind',
        'wird',
        'werden',
        'wurde',
        'eine',
        'einen',
        'einem',
        'eines',
        'für',
        'mit',
        'auf',
        'sich',
        'von',
        'zum',
        'zur',
        'bei',
        'auch',
        'kann',
        'können',
        'muss',
        'soll',
        'durch',
        'über',
        'nach',
        'aus',
        'dass',
        'wenn',
        'sowie',
    }
)

_WORD_RE = re.compile(r'[^\W\d_]+', re.UNICODE)

ALLOWED_DOCUMENT_LANGUAGES = frozenset({'de', 'hsb', 'dsb'})


class UnknownLanguageError(ValueError):
    """Raised when a parse request asks for a language we do not store."""


def resolve_document_language(
    requested: str | None,
    texts: list[str],
    *,
    default: str = 'de',
) -> str:
    """Map a parse-request language to the payload value we actually store.

    ``de`` / ``hsb`` / ``dsb`` are used as-is. ``auto``, empty and omitted fall
    back to detecting the language from the document text, then to ``default``.
    """
    explicit = (requested or '').strip().lower()
    if explicit in {'', 'auto'}:
        return detect_language('\n'.join(texts), default=default)
    if explicit in ALLOWED_DOCUMENT_LANGUAGES:
        return explicit
    raise UnknownLanguageError(
        f'language must be de, hsb, dsb or auto, got {requested!r}'
    )


def detect_language(text: str, default: str = 'de') -> str:
    """Decide between German and Upper Sorbian by weighing signals from both.

    Scoring rather than first-match: a German document quoting a single Sorbian
    name must not flip the whole document to hsb, which matters because the
    result is stored per document and drives BM25 stemming. Returns ``default``
    when the text carries no usable signal either way.
    """
    normalized = text.lower()
    if not normalized.strip():
        return default

    words = _WORD_RE.findall(normalized)
    if not words:
        return default

    hsb_score = sum(1 for char in normalized if char in _HSB_CHARS)
    de_score = sum(1 for char in normalized if char in _DE_CHARS)

    # Function words are the stronger signal, so they weigh more than a single
    # diacritic hit.
    hsb_score += 3 * sum(1 for word in words if word in _HSB_MARKERS)
    de_score += 3 * sum(1 for word in words if word in _DE_MARKERS)

    if hsb_score == de_score:
        return default
    return 'hsb' if hsb_score > de_score else 'de'


def detect_query_language(text: str) -> QueryLanguage:
    """Short queries carry few signals, so an hsb hint is enough to switch."""
    normalized = ' '.join(text.split()).lower()
    if not normalized:
        return 'de'

    if any(char in normalized for char in _HSB_CHARS):
        return 'hsb'

    words = _WORD_RE.findall(normalized)
    if any(word in _HSB_MARKERS for word in words):
        return 'hsb'

    return 'de'
