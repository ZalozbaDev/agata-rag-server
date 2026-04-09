from __future__ import annotations

from bs4 import BeautifulSoup
from bs4.element import Tag


def _normalize_space(text: str) -> str:
    return " ".join(text.split()).strip()


def parse_katolski_posol_html(
    html: str,
    min_text_length: int = 40,
    url: str | None = None,
) -> list[dict[str, str | int]]:
    soup = BeautifulSoup(html, "lxml")

    sections: list[dict[str, str | int]] = []
    current_title = ""
    current_parts: list[str] = []

    skip_patterns = (
        "issn 0138-2543",
        "časopis katolskich serbow",
        "założeny 1863",
        "wudawa towarstwo",
        "cyrila a metoda",
        "katolski posoł",
    )

    text_nodes = soup.select("p")

    def flush() -> None:
        nonlocal current_title, current_parts
        if not current_title:
            current_parts = []
            return

        merged = _normalize_space(" ".join(current_parts))
        if len(merged) >= min_text_length:
            sections.append({
                "title": current_title,
                "text": merged,
            })
        current_parts = []

    for node in text_nodes:
        if not isinstance(node, Tag):
            continue

        classes = set(node.get("class", []))
        text = _normalize_space(node.get_text(" ", strip=True))
        if not text:
            continue

        lower = text.lower()
        if any(pattern in lower for pattern in skip_patterns):
            continue

        # Hauptüberschrift startet neuen Artikel
        if "NAD__PISMO" in classes:
            flush()
            current_title = text
            continue

        # Zwischenüberschriften als Teil des laufenden Artikels behalten
        if "MJEZY-nadpismo" in classes:
            if current_title:
                current_parts.append(text)
            continue

        # Vorspann/Subline
        if "POD-nad-pismo" in classes:
            if current_title:
                current_parts.append(text)
            continue

        # Bildunterschriften meistens ignorieren
        if "POD_wobraz" in classes:
            continue

        # Fließtext
        if any(cls.startswith("_-mm-EINzug") for cls in classes):
            if current_title:
                current_parts.append(text)
            continue

    flush()
    return sections