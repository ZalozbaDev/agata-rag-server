from __future__ import annotations

from qdrant_client import AsyncQdrantClient, models

from app.core.config import Settings


_DISTANCE_MAP = {
    'cosine': models.Distance.COSINE,
    'dot': models.Distance.DOT,
    'euclid': models.Distance.EUCLID,
}


class QdrantGateway:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )

    async def ensure_collection(self) -> None:
        exists = await self.client.collection_exists(self.settings.qdrant_collection)
        if exists:
            return

        await self.client.create_collection(
            collection_name=self.settings.qdrant_collection,
            vectors_config=models.VectorParams(
                size=self.settings.embedding_dimension,
                distance=_DISTANCE_MAP[self.settings.vector_distance],
            ),
        )

    async def delete_by_source_id(self, source_id: str) -> None:
        await self.client.delete(
            collection_name=self.settings.qdrant_collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key='source_id',
                            match=models.MatchValue(value=source_id),
                        )
                    ]
                )
            ),
            wait=True,
        )

    async def upsert_chunks(self, points: list[models.PointStruct]) -> None:
        if not points:
            return
        await self.client.upsert(
            collection_name=self.settings.qdrant_collection,
            points=points,
            wait=True,
        )

    async def search(self, vector: list[float], limit: int) -> list[models.ScoredPoint]:
        result = await self.client.query_points(
            collection_name=self.settings.qdrant_collection,
            query=vector,
            limit=limit,
            with_payload=True,
        )
        return result.points

    async def close(self) -> None:
        import inspect

        result = self.client.close()
        if inspect.isawaitable(result):
            await result
