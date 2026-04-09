from __future__ import annotations

import re
from bs4 import BeautifulSoup
from bs4.element import Tag

HEADING_TAGS = {"h1", "h2", "h3"}
TEXT_TAGS = {"p", "li", "blockquote", "pre"}

DEFAULT_REMOVE_SELECTORS = [
    "script",
    "style",
    "noscript",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "iframe",
    ".comments",
    ".comment",
    ".related",
    ".related-posts",
    ".share",
    ".social",
    ".widget",
    ".newsletter",
    ".login",
    ".abo",
]

DEFAULT_STOP_TEXT_PATTERNS = [
    r"\banmelden\b",
    r"\blogin\b",
    r"\babonnement\b",
    r"\bbenutzername\b",
    r"\bpasswort\b",
    r"\bverwandte beiträge\b",
]

DEFAULT_TITLE_SELECTORS = [
    "h1",
    "article h1",
    ".entry-title",
    ".post-title",
]

DEFAULT_DATE_SELECTORS = [
    "time[datetime]",
    "time",
    ".date",
    ".posted-on",
    ".entry-date",
]


def _normalize_space(text: str) -> str:
    return " ".join(text.split()).strip()


def _compile_patterns(patterns: list[str] | None) -> list[re.Pattern[str]]:
    return [re.compile(pattern, re.IGNORECASE) for pattern in (patterns or [])]


def _remove_noise(container: Tag, remove_selectors: list[str] | None = None) -> None:
    selectors = list(dict.fromkeys(DEFAULT_REMOVE_SELECTORS + (remove_selectors or [])))
    for selector in selectors:
        for bad in container.select(selector):
            bad.decompose()


def _pick_first_existing(soup: BeautifulSoup, selectors: list[str]) -> Tag | None:
    for selector in selectors:
        node = soup.select_one(selector)
        if node:
            return node
    return None


def _pick_best_container(
    soup: BeautifulSoup,
    selectors: list[str],
    positive_selectors: list[str] | None = None,
    negative_selectors: list[str] | None = None,
) -> Tag | None:
    candidates: list[Tag] = []
    for selector in selectors:
        candidates.extend([node for node in soup.select(selector) if isinstance(node, Tag)])

    if not candidates:
        return None

    unique_candidates = list(dict.fromkeys(candidates))

    best: Tag | None = None
    best_score = float("-inf")

    for node in unique_candidates:
        text_len = len(_normalize_space(node.get_text(" ", strip=True)))
        pos_hits = sum(len(node.select(sel)) for sel in (positive_selectors or []))
        neg_hits = sum(len(node.select(sel)) for sel in (negative_selectors or []))
        score = float(text_len) + (120.0 * pos_hits) - (90.0 * neg_hits)

        if score > best_score:
            best_score = score
            best = node

    return best


def _extract_preferred_text(
    soup: BeautifulSoup,
    container: Tag,
    selectors: list[str] | None,
) -> str:
    for selector in selectors or []:
        node = container.select_one(selector) or soup.select_one(selector)
        if node:
            text = _normalize_space(node.get_text(" ", strip=True))
            if text:
                return text
    return ""


def _cut_at_stop_marker(text: str, stop_patterns: list[re.Pattern[str]]) -> tuple[str, bool]:
    if not stop_patterns:
        return text, False

    earliest_end: int | None = None
    found = False
    for pattern in stop_patterns:
        match = pattern.search(text)
        if not match:
            continue
        found = True
        if earliest_end is None or match.start() < earliest_end:
            earliest_end = match.start()

    if not found or earliest_end is None:
        return text, False

    return _normalize_space(text[:earliest_end]), True


def _looks_like_listing_page(container: Tag) -> bool:
    links = len(container.find_all("a"))
    paragraphs = container.find_all("p")
    long_paragraphs = [p for p in paragraphs if len(_normalize_space(p.get_text(" ", strip=True))) > 220]
    cards = len(container.select("article, .post, .entry, .teaser, .news-item"))

    if links >= 30 and len(long_paragraphs) <= 1:
        return True
    if cards >= 4 and len(long_paragraphs) <= 2:
        return True
    return False

def _pick_listing_title(node: Tag) -> str:
    preferred_selectors = [
        ".itemTitle",
        ".catItemTitle",
        ".latestItemTitle",
        ".entry-title",
        ".post-title",
        "h1",
        "h2",
        "h3",
        "h4",
        ".title",
    ]

    for selector in preferred_selectors:
        for el in node.select(selector):
            text = _normalize_space(el.get_text(" ", strip=True))
            if len(text) >= 4:
                return text

    for el in node.select("a"):
        text = _normalize_space(el.get_text(" ", strip=True))
        if len(text) >= 4:
            return text

    return ""


def _extract_listing_sections(
    container: Tag,
    min_text_length: int = 40,
    stop_text_patterns: list[str] | None = None,
) -> list[dict[str, str | int]]:
    stop_patterns = _compile_patterns(stop_text_patterns)
    listing_nodes = container.select(
        "article, .post, .entry, .teaser, .news-item, .list-item, "
        ".latestItemView, .catItemView, .itemView, .blog-item"
    )

    invalid_titles = {"kultura", "róčnica", "spomnjeće", "wo knihach a kniharni"}

    sections: list[dict[str, str | int]] = []
    seen_titles: set[str] = set()

    for node in listing_nodes[:80]:
        title = _pick_listing_title(node)
        if len(title) < 4 or title in seen_titles:
            continue
        if title.lower() in invalid_titles:
            continue

        parts: list[str] = []
        for text_node in node.select(
            "p, blockquote, .latestItemIntroText, .catItemIntroText, .introtext, .entry-summary"
        ):
            text = _normalize_space(text_node.get_text(" ", strip=True))
            if not text:
                continue
            cut_text, should_stop = _cut_at_stop_marker(text, stop_patterns)
            if cut_text:
                parts.append(cut_text)
            if should_stop:
                break

        merged = _normalize_space(" ".join(parts))
        if not merged:
            full_text = _normalize_space(node.get_text(" ", strip=True))
            merged = _normalize_space(full_text.replace(title, "", 1))

        if len(merged) < max(20, min_text_length // 2):
            continue

        seen_titles.add(title)
        sections.append({"title": title, "text": merged})

    return sections


def _extract_sections(
    container: Tag,
    min_text_length: int = 40,
    stop_text_patterns: list[str] | None = None,
    fallback_title: str | None = None,
    preferred_date: str | None = None,
) -> list[dict[str, str | int]]:
    content_nodes = container.find_all(list(HEADING_TAGS | TEXT_TAGS))
    sections: list[dict[str, str | int]] = []
    stop_patterns = _compile_patterns(stop_text_patterns)

    current_title = fallback_title or ""
    current_level = 1
    current_parts: list[str] = []
    terminated = False

    def flush_section() -> None:
        nonlocal current_title, current_level, current_parts

        if not current_title:
            current_parts = []
            return

        merged = _normalize_space(" ".join(current_parts))
        if len(merged) < min_text_length:
            current_parts = []
            return

        if preferred_date and not sections and preferred_date.lower() not in merged.lower():
            merged = _normalize_space(f"{preferred_date}. {merged}")

        section: dict[str, str | int] = {
            "title": current_title,
            "text": merged,
        }

        sections.append(section)
        current_parts = []

    for node in content_nodes:
        if terminated:
            break

        if not isinstance(node, Tag):
            continue

        name = (node.name or "").lower()
        text = _normalize_space(node.get_text(" ", strip=True))

        if not text:
            continue

        cut_text, should_stop = _cut_at_stop_marker(text, stop_patterns)
        if should_stop:
            if name in TEXT_TAGS and cut_text and current_title:
                current_parts.append(cut_text)
            terminated = True
            continue

        if name in HEADING_TAGS:
            flush_section()
            current_title = cut_text or text
            current_level = int(name[1])
            continue

        if name in TEXT_TAGS and current_title:
            current_parts.append(cut_text or text)

    flush_section()
    return sections


def parse_function(
    html: str,
    candidate_selectors: list[str] | None = None,
    remove_selectors: list[str] | None = None,
    stop_text_patterns: list[str] | None = None,
    title_selectors: list[str] | None = None,
    date_selectors: list[str] | None = None,
    listing_mode: str = "auto",
    negative_selectors: list[str] | None = None,
    positive_selectors: list[str] | None = None,
    min_text_length: int = 40,
    url: str | None = None,
) -> list[dict[str, str | int]]:
    soup = BeautifulSoup(html, "lxml")

    selectors = candidate_selectors or [
        "main article",
        "article",
        "main",
        "[role='main']",
        ".entry-content",
        ".post-content",
        ".article-content",
        "body",
    ]

    container = _pick_best_container(
        soup,
        selectors,
        positive_selectors=positive_selectors,
        negative_selectors=negative_selectors,
    )
    if container is None:
        container = _pick_first_existing(soup, selectors)
    if container is None:
        return []

    _remove_noise(container, remove_selectors=remove_selectors)

    title_text = _extract_preferred_text(
        soup,
        container,
        title_selectors or DEFAULT_TITLE_SELECTORS,
    )
    date_text = _extract_preferred_text(
        soup,
        container,
        date_selectors or DEFAULT_DATE_SELECTORS,
    )

    if listing_mode not in {"auto", "article", "listing"}:
        listing_mode = "auto"

    is_listing = (
        listing_mode == "listing"
        or (listing_mode == "auto" and _looks_like_listing_page(container))
    )

    stop_patterns = stop_text_patterns or DEFAULT_STOP_TEXT_PATTERNS

    if is_listing:
        listing_sections = _extract_listing_sections(
            container,
            min_text_length=min_text_length,
            stop_text_patterns=stop_patterns,
        )
        if listing_sections:
            return listing_sections

    sections = _extract_sections(
        container,
        min_text_length=min_text_length,
        stop_text_patterns=stop_patterns,
        fallback_title=title_text or None,
        preferred_date=date_text or None,
    )

    if sections:
        return sections

    # fallback for sparse pages: try listing extraction even when article mode failed
    return _extract_listing_sections(
        container,
        min_text_length=min_text_length,
        stop_text_patterns=stop_patterns,
    )
