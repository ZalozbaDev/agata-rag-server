from __future__ import annotations

from .common import parse_function


def parse_mdr_serbski(
    html: str,
    min_text_length: int = 40,
    url: str | None = None,
) -> list[dict[str, str | int]]:
    selectors = [
        ".sectionDetailPage",
        ".cssBoxContent",
        "#content .sectionDetailPage",
        "#content [role='main']",
        "#content",
        "main",
        "body",
    ]

    normalized_url = (url or "").lower()
    is_article_page = "/nachrichten" in normalized_url or "nachrichten" in normalized_url
    listing_mode = "article" if is_article_page else "listing"

    return parse_function(
        html,
        url=url,
        candidate_selectors=selectors,
        remove_selectors=[
            "nav",
            "header",
            "footer",
            "aside",
            ".sharing-menu-article",
            ".sharing-menu-default",
            ".jumpLabelList",
            ".voice-reader-wrapper",
            ".sectionWrapperRelated",
            ".breadcrumb",
            ".channelNavigation",
            ".channelHeaderTeaser",
            ".mediaCon",
            ".teaser",
            ".boxTeaser",
            ".modLightbox",
            ".bgWrapper",
            "#headerReact",
            "#footer",
            "script",
            "noscript",
            ".linklist",
        ],
        stop_text_patterns=[
            r"\bwobsah artikla\b",
            r"\bpowěsće\s*$",
            r"\barchiw\b",
            r"\bkontakt\b",
            r"\bimpressum\b",
            r"\bdatenschutz\b",
        ],
        title_selectors=[
            ".sectionDetailPage h1 .headline",
            ".sectionDetailPage h1",
            "h1 .headline",
            "h1",
            "meta[property='og:title']",
        ],
        date_selectors=[
            ".sectionDetailPage p.webtime",
            "meta[name='date']",
            "meta[property='article:published_time']",
            "time",
        ],
        listing_mode=listing_mode,
        positive_selectors=[
            ".sectionDetailPage",
            "h1",
            "h3.subtitle",
            ".paragraph p.text",
            "p.webtime",
        ],
        negative_selectors=[
            ".sectionWrapperRelated",
            ".jumpLabelList",
            ".sharing-menu-article",
            ".sharing-menu-default",
            ".breadcrumb",
            "#footer",
            "#headerReact",
        ],
        min_text_length=min_text_length,
    )