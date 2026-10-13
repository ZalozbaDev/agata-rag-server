from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import UUID

from qdrant_client import models

from app.clients.qdrant_client import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, QdrantGateway
from app.models.schemas import ParsedSection
from app.providers.embedding_factory import EmbeddingProvider
from app.utils.chunking import Chunker
from app.utils.embedding_prompts import PromptStyle, format_document
from app.utils.hashing import stable_sha256
from app.utils.language import resolve_document_language
from app.utils.sparse_embedding import SparseEmbeddingProvider

logger = logging.getLogger(__name__)

# Enough prose to make detection stable without scanning whole books.
_DETECTION_SAMPLE_CHARS = 20_000


def _point_id(source_id: str, chunk: dict[str, object], language: str) -> str:
    raw_hash = stable_sha256(
        f"{source_id}|{chunk['section_idx']}|{chunk['chunk_idx']}|{chunk['text']}|{language}"
    )
    return str(UUID(raw_hash[:32]))


class IndexingService:
    def __init__(
        self,
        qdrant: QdrantGateway,
        embeddings: EmbeddingProvider,
        sparse_embeddings: SparseEmbeddingProvider,
        chunker: Chunker,
        document_language: str,
        prompt_style: PromptStyle,
        upsert_batch_size: int,
        bm25_language: str,
    ) -> None:
        self.qdrant = qdrant
        self.embeddings = embeddings
        self.sparse_embeddings = sparse_embeddings
        self.chunker = chunker
        self.document_language = document_language
        self.prompt_style = prompt_style
        self.upsert_batch_size = upsert_batch_size
        self.bm25_language = bm25_language

    async def store_sections(
        self,
        *,
        source_id: str,
        source_type: str,
        sections: list[ParsedSection],
        source_url: str | None = None,
        collection: str | None = None,
        language: str | None = None,
    ) -> int:
        chunks = self.chunker.split_sections(sections)
        target = self.qdrant.resolve_collection(collection)
        await self.qdrant.ensure_collection(target)

        if not chunks:
            # An empty re-parse must not silently wipe a previously good version.
            logger.warning(
                'No chunks produced for source %s; leaving %s untouched',
                source_id,
                target,
            )
            return 0

        chunk_language = self._resolve_language(language, sections)
        logger.info(
            'indexing language=%s source=%s collection=%s',
            chunk_language,
            source_id,
            target,
        )
        existing_ids = await self.qdrant.fetch_point_ids(target, source_id)
        desired_ids = [_point_id(source_id, chunk, chunk_language) for chunk in chunks]

        pending = [
            (point_id, chunk)
            for point_id, chunk in zip(desired_ids, chunks, strict=True)
            if point_id not in existing_ids
        ]
        stale_ids = sorted(existing_ids - set(desired_ids))

        logger.info(
            'indexing %s into %s | lang=%s chunks=%d new=%d unchanged=%d stale=%d',
            source_id,
            target,
            chunk_language,
            len(chunks),
            len(pending),
            len(chunks) - len(pending),
            len(stale_ids),
        )

        points: list[models.PointStruct] = []
        if pending:
            dense_vectors, sparse_vectors = await self._embed_chunks(
                [str(chunk['text']) for _, chunk in pending],
                [str(chunk['title'] or '') for _, chunk in pending],
                chunk_language,
            )
            indexed_at = datetime.now(timezone.utc).isoformat()
            points = [
                self._build_point(
                    point_id=point_id,
                    chunk=chunk,
                    dense_vector=dense_vector,
                    sparse_vector=sparse_vector,
                    source_id=source_id,
                    source_type=source_type,
                    source_url=source_url or '',
                    indexed_at=indexed_at,
                    language=chunk_language,
                )
                for (point_id, chunk), dense_vector, sparse_vector in zip(
                    pending,
                    dense_vectors,
                    sparse_vectors,
                    strict=True,
                )
            ]

        # Write the new version before removing the old one, so a provider
        # outage during embedding cannot leave the source missing entirely.
        await self.qdrant.upsert_chunks(target, points, self.upsert_batch_size)
        await self.qdrant.delete_points(target, stale_ids)
        return len(chunks)

    def _resolve_language(
        self,
        requested: str | None,
        sections: list[ParsedSection],
    ) -> str:
        """Caller can pin de/hsb; otherwise the text decides.

        The configured default is only used when the document has no language
        signal. Pinning DOCUMENT_LANGUAGE=hsb is how German PDFs used to end
        up tagged as Sorbian.
        """
        sample: list[str] = []
        length = 0
        for section in sections:
            sample.append(section.text)
            length += len(section.text)
            if length >= _DETECTION_SAMPLE_CHARS:
                break

        return resolve_document_language(
            requested,
            sample,
            default=self.document_language,
        )

    def _build_point(
        self,
        *,
        point_id: str,
        chunk: dict[str, object],
        dense_vector: list[float],
        sparse_vector: models.SparseVector,
        source_id: str,
        source_type: str,
        source_url: str,
        indexed_at: str,
        language: str,
    ) -> models.PointStruct:
        return models.PointStruct(
            id=point_id,
            vector={
                DENSE_VECTOR_NAME: dense_vector,
                SPARSE_VECTOR_NAME: sparse_vector,
            },
            payload={
                'source_id': source_id,
                'source_type': source_type,
                'source_url': source_url,
                'title': chunk['title'],
                'text': chunk['text'],
                'section_idx': chunk['section_idx'],
                'chunk_idx': chunk['chunk_idx'],
                'language': language,
                'indexed_at': indexed_at,
            },
        )

    async def _embed_chunks(
        self,
        chunk_texts: list[str],
        chunk_titles: list[str],
        language: str,
    ) -> tuple[list[list[float]], list[models.SparseVector]]:
        # The dense model sees the task-prefixed form; BM25 sees the raw text so
        # the prompt scaffolding does not pollute the lexical index.
        prompted = [
            format_document(text, title, self.prompt_style)
            for text, title in zip(chunk_texts, chunk_titles, strict=True)
        ]
        dense_vectors = await self.embeddings.embed_texts(prompted)
        sparse_vectors = await self.sparse_embeddings.embed_texts_async(
            chunk_texts,
            language,
        )
        return dense_vectors, sparse_vectors
