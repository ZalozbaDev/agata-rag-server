from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from qdrant_client import models

from app.clients.qdrant_client import QdrantGateway
from app.models.schemas import ParsedSection
from app.providers.openai_provider import OpenAIEmbeddingProvider
from app.utils.chunking import Chunker
from app.utils.hashing import stable_sha256

class IndexingService:
    def __init__(
        self,
        qdrant: QdrantGateway,
        embeddings: OpenAIEmbeddingProvider,
        chunker: Chunker,
    ) -> None:
        self.qdrant = qdrant
        self.embeddings = embeddings
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

        chunk_texts = [chunk['text'] for chunk in chunks]
        vectors = await self.embeddings.embed_texts(chunk_texts)
        now = datetime.now(timezone.utc).isoformat()

        points: list[models.PointStruct] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
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
                'indexed_at': now,
            }

            points.append(
                models.PointStruct(
                    id=chunk_id,
                    vector=vector,
                    payload=payload,
                )
            )

        await self.qdrant.upsert_chunks(points)