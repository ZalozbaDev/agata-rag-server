from __future__ import annotations

from app.clients.qdrant_client import QdrantGateway
from app.providers.openai_provider import OpenAIEmbeddingProvider


class RetrievalService:
    def __init__(
        self,
        qdrant: QdrantGateway,
        embeddings: OpenAIEmbeddingProvider,
    ) -> None:
        self.qdrant = qdrant
        self.embeddings = embeddings

    async def retrieve(self, question: str, top_k: int) -> list[dict[str, object]]:
        query_vector = await self.embeddings.embed_query(question)
        hits = await self.qdrant.search(query_vector, limit=top_k)

        return [
            {
                'score': hit.score,
                'payload': hit.payload or {},
            }
            for hit in hits
        ]
