from __future__ import annotations

from .common import parse_function


def parse_zalozba(
    html: str,
    min_text_length: int = 40,
    url: str | None = None,
) -> list[dict[str, str | int]]:
    selectors = [
        "main article",
        "article",
        ".entry-content",
        ".post-content",
        ".article-content",
        "main",
        "body",
    ]
    return parse_function(
        html,
        url=url,
        candidate_selectors=selectors,
        min_text_length=min_text_length,
    )
