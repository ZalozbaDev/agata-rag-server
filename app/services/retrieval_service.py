from __future__ import annotations

from typing import TypedDict

from app.clients.qdrant_client import QdrantGateway
from app.providers.embedding_factory import EmbeddingProvider
from app.services.indexing_service import DOCUMENT_LANGUAGE
from app.utils.sparse_embedding import SparseEmbeddingProvider


class RetrievalResult(TypedDict):
    score: float
    payload: dict[str, object]


class RetrievalService:
    def __init__(
        self,
        qdrant: QdrantGateway,
        embeddings: EmbeddingProvider,
        sparse_embeddings: SparseEmbeddingProvider,
        *,
        hybrid_prefetch_limit: int,
    ) -> None:
        self.qdrant = qdrant
        self.embeddings = embeddings
        self.sparse_embeddings = sparse_embeddings
        self.hybrid_prefetch_limit = hybrid_prefetch_limit

    async def retrieve(self, question: str, top_k: int) -> list[RetrievalResult]:
        dense_vector = await self.embeddings.embed_query(question)
        sparse_vector = await self.sparse_embeddings.embed_query_async(question)

        hits = await self.qdrant.hybrid_search(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            limit=top_k,
            prefetch_limit=self.hybrid_prefetch_limit,
            language=DOCUMENT_LANGUAGE,
        )

        return [
            {
                'score': float(hit.score),
                'payload': hit.payload or {},
            }
            for hit in hits
        ]
