from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from qdrant_client import models

from app.clients.qdrant_client import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, QdrantGateway
from app.models.schemas import ParsedSection
from app.providers.openai_provider import OpenAIEmbeddingProvider
from app.utils.chunking import Chunker
from app.utils.hashing import stable_sha256
from app.utils.sparse_embedding import SparseEmbeddingProvider

DOCUMENT_LANGUAGE = 'hsb'


class IndexingService:
    def __init__(
        self,
        qdrant: QdrantGateway,
        embeddings: OpenAIEmbeddingProvider,
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
        now = datetime.now(timezone.utc).isoformat()

        points: list[models.PointStruct] = []
        for chunk, dense_vector, sparse_vector in zip(
            chunks,
            dense_vectors,
            sparse_vectors,
            strict=True,
        ):
            raw_hash = stable_sha256(
                f"{source_id}|{chunk['section_idx']}|{chunk['chunk_idx']}|{chunk['text']}"
            )
            chunk_id = str(UUID(raw_hash[:32]))

            payload = {
                'source_id': source_id,
                'source_type': source_type,
                'source_url': source_url or '',
                'title': chunk['title'],
                'text': chunk['text'],
                'section_idx': chunk['section_idx'],
                'chunk_idx': chunk['chunk_idx'],
                'language': DOCUMENT_LANGUAGE,
                'indexed_at': now,
            }

            points.append(
                models.PointStruct(
                    id=chunk_id,
                    vector={
                        DENSE_VECTOR_NAME: dense_vector,
                        SPARSE_VECTOR_NAME: sparse_vector,
                    },
                    payload=payload,
                )
            )

        await self.qdrant.upsert_chunks(points)

    async def _embed_chunks(
        self,
        chunk_texts: list[str],
    ) -> tuple[list[list[float]], list[models.SparseVector]]:
        dense_vectors = await self.embeddings.embed_texts(chunk_texts)
        sparse_vectors = await self.sparse_embeddings.embed_texts_async(chunk_texts)
        return dense_vectors, sparse_vectors
