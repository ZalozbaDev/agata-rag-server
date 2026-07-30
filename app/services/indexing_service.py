from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from qdrant_client import models

from app.clients.qdrant_client import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, QdrantGateway
from app.models.schemas import ParsedSection
from app.providers.embedding_factory import EmbeddingProvider
from app.utils.chunking import Chunker
from app.utils.hashing import stable_sha256
from app.utils.sparse_embedding import SparseEmbeddingProvider

DOCUMENT_LANGUAGE = 'hsb'


class IndexingService:
    def __init__(
        self,
        qdrant: QdrantGateway,
        embeddings: EmbeddingProvider,
        sparse_embeddings: SparseEmbeddingProvider,
        chunker: Chunker,
    ) -> None:
        self.qdrant = qdrant
        self.embeddings = embeddings
        self.sparse_embeddings = sparse_embeddings
        self.chunker = chunker

    async def store_sections(
        self,
        *,
        source_id: str,
        source_type: str,
        sections: list[ParsedSection],
        source_url: str | None = None,
    ) -> None:
        chunks = self.chunker.split_sections(sections)
        if not chunks:
            return

        await self.qdrant.delete_by_source_id(source_id)

        chunk_texts = [str(chunk['text']) for chunk in chunks]
        dense_vectors, sparse_vectors = await self._embed_chunks(chunk_texts)
        indexed_at = datetime.now(timezone.utc).isoformat()

        points = [
            self._build_point(
                chunk=chunk,
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                source_id=source_id,
                source_type=source_type,
                source_url=source_url or '',
                indexed_at=indexed_at,
            )
            for chunk, dense_vector, sparse_vector in zip(
                chunks,
                dense_vectors,
                sparse_vectors,
                strict=True,
            )
        ]
        await self.qdrant.upsert_chunks(points)

    def _build_point(
        self,
        *,
        chunk: dict[str, object],
        dense_vector: list[float],
        sparse_vector: models.SparseVector,
        source_id: str,
        source_type: str,
        source_url: str,
        indexed_at: str,
    ) -> models.PointStruct:
        raw_hash = stable_sha256(
            f"{source_id}|{chunk['section_idx']}|{chunk['chunk_idx']}|{chunk['text']}"
        )
        return models.PointStruct(
            id=str(UUID(raw_hash[:32])),
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
                'language': DOCUMENT_LANGUAGE,
                'indexed_at': indexed_at,
            },
        )

    async def _embed_chunks(
        self,
        chunk_texts: list[str],
    ) -> tuple[list[list[float]], list[models.SparseVector]]:
        dense_vectors = await self.embeddings.embed_texts(chunk_texts)
        sparse_vectors = await self.sparse_embeddings.embed_texts_async(chunk_texts)
        return dense_vectors, sparse_vectors
