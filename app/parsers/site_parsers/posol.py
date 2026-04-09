from __future__ import annotations

from .common import parse_function


def parse_posol(
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
        remove_selectors=[
            ".related",
            ".related-posts",
            ".share",
            ".social",
            ".comments",
            ".comment",
            ".widget",
            ".newsletter",
            ".sidebar",
            ".login",
            ".abo",
        ],
        stop_text_patterns=[
            r"\banmelden\b",
            r"\blogin\b",
            r"\babonnement\b",
            r"\bpasswort\b",
            r"\bverwandte beiträge\b",
        ],
        title_selectors=["h1", "article h1", ".entry-title", ".post-title"],
        date_selectors=["time", ".date", ".entry-date", ".meta-date"],
        listing_mode="auto",
        positive_selectors=["article", ".entry-content", ".post-content"],
        negative_selectors=[".sidebar", ".widget", "nav", "footer"],
        min_text_length=min_text_length,
    )
