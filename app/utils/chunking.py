from __future__ import annotations

import re

from app.models.schemas import ParsedSection

_SENTENCE_SPLIT = re.compile(r'(?<=[.!?…])\s+')
_PARAGRAPH_SPLIT = re.compile(r'\n\s*\n+')


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

            for paragraph in self._split_paragraphs(text):
                section_chunks = self._chunk_paragraph(paragraph)
                for chunk_idx, chunk_text in enumerate(section_chunks):
                    chunks.append(
                        {
                            'title': section.title,
                            'text': chunk_text,
                            'section_idx': section_idx,
                            'chunk_idx': chunk_idx,
                        }
                    )
        return chunks

    def _split_paragraphs(self, text: str) -> list[str]:
        paragraphs = [part.strip() for part in _PARAGRAPH_SPLIT.split(text) if part.strip()]
        return paragraphs or [text.strip()]

    def _split_sentences(self, paragraph: str) -> list[str]:
        sentences = [part.strip() for part in _SENTENCE_SPLIT.split(paragraph) if part.strip()]
        return sentences or [paragraph.strip()]

    def _chunk_paragraph(self, paragraph: str) -> list[str]:
        sentences = self._split_sentences(paragraph)
        if not sentences:
            return []

        chunks: list[str] = []
        current_parts: list[str] = []
        current_len = 0

        for sentence in sentences:
            sentence_len = len(sentence)
            projected_len = current_len + (1 if current_parts else 0) + sentence_len

            if current_parts and projected_len > self.chunk_size:
                chunks.append(' '.join(current_parts))
                overlap_parts = self._overlap_tail(current_parts)
                current_parts = overlap_parts
                current_len = sum(len(part) for part in current_parts) + max(
                    0,
                    len(current_parts) - 1,
                )

            if sentence_len > self.chunk_size:
                if current_parts:
                    chunks.append(' '.join(current_parts))
                    current_parts = []
                    current_len = 0
                chunks.extend(self._split_long_sentence(sentence))
                continue

            if current_parts:
                current_len += 1 + sentence_len
            else:
                current_len = sentence_len
            current_parts.append(sentence)

        if current_parts:
            chunks.append(' '.join(current_parts))

        return chunks

    def _overlap_tail(self, parts: list[str]) -> list[str]:
        if self.chunk_overlap <= 0 or not parts:
            return []

        overlap_parts: list[str] = []
        overlap_len = 0

        for part in reversed(parts):
            projected = overlap_len + (1 if overlap_parts else 0) + len(part)
            if overlap_parts and projected > self.chunk_overlap:
                break
            overlap_parts.insert(0, part)
            overlap_len = projected

        return overlap_parts

    def _split_long_sentence(self, sentence: str) -> list[str]:
        chunks: list[str] = []
        start = 0
        while start < len(sentence):
            end = min(len(sentence), start + self.chunk_size)
            if end < len(sentence):
                split_at = sentence.rfind(' ', start, end)
                if split_at > start:
                    end = split_at
            chunk_text = sentence[start:end].strip()
            if chunk_text:
                chunks.append(chunk_text)
            if end >= len(sentence):
                break
            start = max(end - self.chunk_overlap, start + 1)
        return chunks
