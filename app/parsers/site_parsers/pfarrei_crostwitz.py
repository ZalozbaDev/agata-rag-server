from __future__ import annotations

from .common import parse_function


def parse_pfarrei_crostwitz(
    html: str,
    min_text_length: int = 40,
    url: str | None = None,
) -> list[dict[str, str | int]]:
    selectors = [
        "main article",
        "article",
        ".entry-content",
        ".post-content",
        "main",
        "body",
    ]

    is_listing = bool(url and "/vermeldungen" in url and url.rstrip("/").endswith("vermeldungen"))

    return parse_function(
        html,
        url=url,
        candidate_selectors=selectors,
        remove_selectors=[
            ".sidebar",
            ".widget",
            ".related",
            ".share",
            ".newsletter",
            ".comments",
            ".menu",
            ".navigation",
            ".login",
            ".abo",
        ],
        stop_text_patterns=[
            r"\banmelden\b",
            r"\blogin\b",
            r"\babonnement\b",
            r"\bbenutzername\b",
            r"\bpasswort\b",
            r"\bnewsletter\b",
        ],
        title_selectors=["h1", "article h1", ".entry-title", ".page-title"],
        date_selectors=["time", ".date", ".entry-date", ".meta-date"],
        listing_mode="listing" if is_listing else "auto",
        positive_selectors=["article", ".entry-content", "main"],
        negative_selectors=[".sidebar", ".widget", "nav", "footer"],
        min_text_length=min_text_length,
    )
