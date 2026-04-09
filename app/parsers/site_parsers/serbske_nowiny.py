from __future__ import annotations

from .common import parse_function


def parse_serbske_nowiny(
    html: str,
    min_text_length: int = 40,
    url: str | None = None,
) -> list[dict[str, str | int]]:
    selectors = [
        "main article",
        "article.post",
        "article",
        ".td-post-content",
        ".entry-content",
        ".post-content",
        "main",
        "body",
    ]
    normalized_url = (url or "").lower()
    is_article_page = "/item/" in normalized_url
    listing_mode = "article" if is_article_page else "listing"

    return parse_function(
        html,
        url=url,
        candidate_selectors=selectors,
        remove_selectors=[
            ".td-post-sharing",
            ".td-post-source-tags",
            ".td_block_related_posts",
            ".td-post-author-name",
            ".td-post-next-prev",
            ".td-post-featured-image",
            ".td-comments-title",
            ".td-a-rec",
            ".td_block_wrap",
        ],
        stop_text_patterns=[
            r"\banmelden\b",
            r"\babonnieren\b",
            r"\bpasswort\b",
            r"\bverwandte beiträge\b",
        ],
        title_selectors=["article h1", ".entry-title", ".tdb-title-text"],
        date_selectors=["article time", "time.entry-date", ".td-post-date"],
        listing_mode=listing_mode,
        positive_selectors=["article", ".td-post-content", ".entry-content"],
        negative_selectors=[".td_block_related_posts", ".td-post-sharing"],
        min_text_length=min_text_length,
    )
