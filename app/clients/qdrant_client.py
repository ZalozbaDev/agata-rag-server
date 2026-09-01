from __future__ import annotations

import inspect
import logging
import re
from collections.abc import Awaitable, Callable
from typing import TypeVar

from qdrant_client import AsyncQdrantClient, models

from app.core.config import Settings
from app.core.errors import QdrantServiceError
from app.providers._provider_utils import is_timeout_error
from app.utils.retry import retry_async

logger = logging.getLogger(__name__)

T = TypeVar('T')

COLLECTION_NAME_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_\-]{0,62}$')


class UnknownCollectionError(ValueError):
    """Raised when a request asks for a collection that is not allowed."""

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
        self.client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=settings.qdrant_timeout_seconds,
        )
        self._ensured_collections: set[str] = set()

    def resolve_collection(self, collection: str | None) -> str:
        """Map an optional request-supplied collection to a safe, known name."""
        name = (collection or '').strip()
        if not name:
            return self.settings.qdrant_collection

        if not COLLECTION_NAME_RE.match(name):
            raise UnknownCollectionError(f'Invalid collection name: {collection!r}')

        allowed = self.settings.qdrant_allowed_collections
        if allowed and name not in allowed and name != self.settings.qdrant_collection:
            raise UnknownCollectionError(f'Unknown collection: {name}')

        return name

    async def ensure_collection(self, collection: str | None = None) -> None:
        name = self.resolve_collection(collection)
        if name in self._ensured_collections:
            return

        exists = await self._call(lambda: self.client.collection_exists(name))
        if exists and await self._collection_matches_schema(name):
            await self._ensure_payload_indexes(name)
            self._ensured_collections.add(name)
            return

        if exists:
            logger.warning(
                'Qdrant collection %s has incompatible schema; recreating.',
                name,
            )
            await self._call(lambda: self.client.delete_collection(name))

        await self._call(lambda: self._create_collection(name))
        await self._ensure_payload_indexes(name)
        self._ensured_collections.add(name)

    async def _collection_matches_schema(self, collection: str) -> bool:
        info = await self._call(lambda: self.client.get_collection(collection))
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

    async def _create_collection(self, collection: str) -> None:
        await self.client.create_collection(
            collection_name=collection,
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

    async def _ensure_payload_indexes(self, collection: str) -> None:
        for field_name, schema_type in _PAYLOAD_INDEX_FIELDS.items():
            try:
                await self._call(
                    lambda field=field_name, schema=schema_type: self.client.create_payload_index(
                        collection_name=collection,
                        field_name=field,
                        field_schema=schema,
                    )
                )
            except Exception as exc:
                if 'already exists' not in str(exc).lower():
                    logger.warning(
                        'Could not create payload index %s on %s: %s',
                        field_name,
                        collection,
                        exc,
                    )

    async def delete_by_source_id(self, collection: str, source_id: str) -> None:
        await self._call(
            lambda: self.client.delete(
                collection_name=collection,
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

    async def list_collections(self) -> list[dict[str, object]]:
        response = await self._call(self.client.get_collections)
        summaries: list[dict[str, object]] = []

        for description in response.collections:
            name = description.name
            try:
                info = await self._call(
                    lambda n=name: self.client.get_collection(n)
                )
                points = info.points_count
            except Exception:
                points = None
            summaries.append({'name': name, 'points': points})

        return sorted(summaries, key=lambda item: str(item['name']))

    async def upsert_chunks(
        self,
        collection: str,
        points: list[models.PointStruct],
    ) -> None:
        if not points:
            return
        await self._call(
            lambda: self.client.upsert(
                collection_name=collection,
                points=points,
                wait=True,
            )
        )

    async def hybrid_search(
        self,
        *,
        collection: str,
        dense_vector: list[float],
        sparse_vector: models.SparseVector,
        limit: int,
        prefetch_limit: int,
    ) -> list[models.ScoredPoint]:
        result = await self._call(
            lambda: self.client.query_points(
                collection_name=collection,
                prefetch=[
                    models.Prefetch(
                        query=dense_vector,
                        using=DENSE_VECTOR_NAME,
                        limit=prefetch_limit,
                    ),
                    models.Prefetch(
                        query=sparse_vector,
                        using=SPARSE_VECTOR_NAME,
                        limit=prefetch_limit,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
        )
        return result.points

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
