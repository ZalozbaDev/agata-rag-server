from __future__ import annotations

from app.models.schemas import ParsedSection


class Chunker:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError('chunk_overlap must be smaller than chunk_size')
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_sections(self, sections: list[ParsedSection]) -> list[dict[str, object]]:
        chunks: list[dict[str, object]] = []
        for section_idx, section in enumerate(sections):
            text = section.text.strip()
            if not text:
                continue
            start = 0
            chunk_idx = 0
            while start < len(text):
                end = min(len(text), start + self.chunk_size)
                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunks.append(
                        {
                            'title': section.title,
                            'text': chunk_text,
                            'section_idx': section_idx,
                            'chunk_idx': chunk_idx,
                        }
                    )
                if end == len(text):
                    break
                start = end - self.chunk_overlap
                chunk_idx += 1
        return chunks
