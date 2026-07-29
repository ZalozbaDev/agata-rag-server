from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

from qdrant_client import AsyncQdrantClient, models

from app.core.config import Settings
from app.core.errors import QdrantServiceError
from app.providers._provider_utils import is_timeout_error
from app.utils.retry import retry_async

logger = logging.getLogger(__name__)

T = TypeVar('T')

DENSE_VECTOR_NAME = 'dense'
SPARSE_VECTOR_NAME = 'bm25'

_DISTANCE_MAP = {
    'cosine': models.Distance.COSINE,
    'dot': models.Distance.DOT,
    'euclid': models.Distance.EUCLID,
}

_PAYLOAD_INDEX_FIELDS: dict[str, models.PayloadSchemaType] = {
    'source_id': models.PayloadSchemaType.KEYWORD,
    'source_type': models.PayloadSchemaType.KEYWORD,
    'language': models.PayloadSchemaType.KEYWORD,
    'indexed_at': models.PayloadSchemaType.DATETIME,
}


def _is_transient_qdrant_error(exc: Exception) -> bool:
    message = str(exc).lower()
    transient_markers = (
        'timeout',
        'temporarily',
        'connection',
        'unavailable',
        'reset',
        '503',
        '502',
    )
    return any(marker in message for marker in transient_markers)


class QdrantGateway:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        print(settings.qdrant_url)
        print(settings.qdrant_api_key)
        print(settings.qdrant_timeout_seconds)

        print(settings)
        self.client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=settings.qdrant_timeout_seconds,
        )

    async def ensure_collection(self) -> None:
        exists = await self._call(
            lambda: self.client.collection_exists(self.settings.qdrant_collection)
        )
        if exists and await self._collection_matches_schema():
            await self._ensure_payload_indexes()
            return

        if exists:
            logger.warning(
                'Qdrant collection %s has incompatible schema; recreating.',
                self.settings.qdrant_collection,
            )
            await self._call(
                lambda: self.client.delete_collection(self.settings.qdrant_collection)
            )

        await self._call(self._create_collection)
        await self._ensure_payload_indexes()

    async def _collection_matches_schema(self) -> bool:
        info = await self._call(
            lambda: self.client.get_collection(self.settings.qdrant_collection)
        )
        params = info.config.params

        vectors = params.vectors
        if not isinstance(vectors, dict):
            return False

        dense = vectors.get(DENSE_VECTOR_NAME)
        if dense is None:
            return False

        dense_size = getattr(dense, 'size', None)
        dense_distance = getattr(dense, 'distance', None)
        if dense_size != self.settings.embedding_dimension:
            return False
        if dense_distance != _DISTANCE_MAP[self.settings.vector_distance]:
            return False

        sparse_vectors = params.sparse_vectors or {}
        return SPARSE_VECTOR_NAME in sparse_vectors

    async def _create_collection(self) -> None:
        await self.client.create_collection(
            collection_name=self.settings.qdrant_collection,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=self.settings.embedding_dimension,
                    distance=_DISTANCE_MAP[self.settings.vector_distance],
                ),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                ),
            },
        )

    async def _ensure_payload_indexes(self) -> None:
        for field_name, schema_type in _PAYLOAD_INDEX_FIELDS.items():
            try:
                await self._call(
                    lambda field=field_name, schema=schema_type: self.client.create_payload_index(
                        collection_name=self.settings.qdrant_collection,
                        field_name=field,
                        field_schema=schema,
                    )
                )
            except Exception as exc:
                if 'already exists' not in str(exc).lower():
                    logger.warning(
                        'Could not create payload index %s on %s: %s',
                        field_name,
                        self.settings.qdrant_collection,
                        exc,
                    )

    async def delete_by_source_id(self, source_id: str) -> None:
        await self._call(
            lambda: self.client.delete(
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
        )

    async def upsert_chunks(self, points: list[models.PointStruct]) -> None:
        if not points:
            return
        await self._call(
            lambda: self.client.upsert(
                collection_name=self.settings.qdrant_collection,
                points=points,
                wait=True,
            )
        )

    async def hybrid_search(
        self,
        *,
        dense_vector: list[float],
        sparse_vector: models.SparseVector,
        limit: int,
        prefetch_limit: int,
        language: str | None = None,
    ) -> list[models.ScoredPoint]:
        query_filter = self._language_filter(language)
        result = await self._call(
            lambda: self.client.query_points(
                collection_name=self.settings.qdrant_collection,
                prefetch=[
                    models.Prefetch(
                        query=dense_vector,
                        using=DENSE_VECTOR_NAME,
                        limit=prefetch_limit,
                        filter=query_filter,
                    ),
                    models.Prefetch(
                        query=sparse_vector,
                        using=SPARSE_VECTOR_NAME,
                        limit=prefetch_limit,
                        filter=query_filter,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
        )
        return result.points

    @staticmethod
    def _language_filter(language: str | None) -> models.Filter | None:
        if not language:
            return None
        return models.Filter(
            must=[
                models.FieldCondition(
                    key='language',
                    match=models.MatchValue(value=language),
                )
            ]
        )

    async def _call(self, fn: Callable[[], Awaitable[T]]) -> T:
        try:
            return await retry_async(
                fn,
                max_attempts=self.settings.provider_max_retries,
                base_delay=self.settings.provider_retry_base_delay_seconds,
                retryable=_is_transient_qdrant_error,
            )
        except Exception as exc:
            raise QdrantServiceError(
                f'Qdrant request failed: {exc}',
                is_timeout=is_timeout_error(exc),
            ) from exc

    async def close(self) -> None:
        result = self.client.close()
        if inspect.isawaitable(result):
            await result
