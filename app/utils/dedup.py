from __future__ import annotations

import re

_WORD_RE = re.compile(r'\w+', re.UNICODE)
_MIN_STITCH_WORDS = 4


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _shingles(text: str, size: int = 5) -> set[str]:
    tokens = _tokens(text)
    if len(tokens) <= size:
        return {' '.join(tokens)} if tokens else set()
    return {' '.join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}


def similarity(left: str, right: str) -> float:
    """Overlap coefficient over word shingles.

    Overlap rather than Jaccard, so a short chunk fully contained in a longer
    one scores 1.0 instead of being diluted by the length difference. That is
    exactly the shape produced by chunk_overlap.
    """
    left_shingles = _shingles(left)
    right_shingles = _shingles(right)
    if not left_shingles or not right_shingles:
        return 0.0

    intersection = len(left_shingles & right_shingles)
    return intersection / min(len(left_shingles), len(right_shingles))


def stitch(left: str, right: str) -> str:
    """Join two adjacent chunks, dropping the text they share at the seam."""
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    max_overlap = min(len(left_tokens), len(right_tokens))

    for size in range(max_overlap, _MIN_STITCH_WORDS - 1, -1):
        if left_tokens[-size:] == right_tokens[:size]:
            # Re-find the seam in the original text to keep punctuation intact.
            matches = list(_WORD_RE.finditer(right))
            if len(matches) >= size:
                return f'{left.rstrip()} {right[matches[size - 1].end() :].lstrip()}'.strip()

    return f'{left.rstrip()}\n\n{right.lstrip()}'
