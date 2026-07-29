from __future__ import annotations

from typing import Literal

QueryLanguage = Literal['de', 'hsb']

_HSB_CHARS = frozenset('łńśźćž')
_HSB_MARKERS = (
    'wón',
    'wonje',
    'chcy',
    'hdy',
    'nětko',
    'wob',
    'dźě',
    'serbski',
    'serbšćina',
    'hornjoserbšćina',
    'rěč',
    'wutrob',
    'wobsah',
    'přichod',
    'přeco',
)


def detect_query_language(text: str) -> QueryLanguage:
    """Heuristic HSB detection via diacritics and frequent function words.

    Defaults to German so sparse Latin-only HSB queries still get DE→HSB
    translation before sparse retrieval rather than skipping it.
    """
    normalized = ' '.join(text.split()).lower()
    if not normalized:
        return 'de'

    if any(char in normalized for char in _HSB_CHARS):
        return 'hsb'

    if any(marker in normalized for marker in _HSB_MARKERS):
        return 'hsb'

    return 'de'
