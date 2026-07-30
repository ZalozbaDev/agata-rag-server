from __future__ import annotations

from typing import TypedDict

from app.clients.qdrant_client import QdrantGateway
from app.providers.embedding_factory import EmbeddingProvider
from app.providers.sotra_provider import SotraProvider
from app.services.indexing_service import DOCUMENT_LANGUAGE
from app.utils.language import QueryLanguage, detect_query_language
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
        sotra: SotraProvider,
        *,
        hybrid_prefetch_limit: int,
    ) -> None:
        self.qdrant = qdrant
        self.embeddings = embeddings
        self.sparse_embeddings = sparse_embeddings
        self.sotra = sotra
        self.hybrid_prefetch_limit = hybrid_prefetch_limit

    async def retrieve(self, question: str, top_k: int) -> list[RetrievalResult]:
        query_language = detect_query_language(question)
        sparse_query_text = await self._sparse_query_text(question, query_language)

        dense_vector = await self.embeddings.embed_query(question)
        sparse_vector = await self.sparse_embeddings.embed_query_async(sparse_query_text)

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

    async def _sparse_query_text(
        self,
        question: str,
        query_language: QueryLanguage,
    ) -> str:
        # Sparse BM25 index is Upper Sorbian; translate DE queries before sparse search.
        if query_language == 'hsb':
            return question
        return await self.sotra.translate_de_to_hsb(question)
