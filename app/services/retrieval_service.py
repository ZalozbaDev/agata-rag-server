from __future__ import annotations

import logging
import math
from typing import TypedDict

from app.clients.qdrant_client import DENSE_VECTOR_NAME, QdrantGateway
from app.providers.embedding_factory import EmbeddingProvider
from app.services.reranker import CrossEncoderReranker
from app.utils.embedding_prompts import PromptStyle, format_query
from app.utils.query_cache import LruCache
from app.utils.sparse_embedding import SparseEmbeddingProvider

logger = logging.getLogger(__name__)


class RetrievalResult(TypedDict):
    score: float
    payload: dict[str, object]


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _dense_vector_of(hit: object) -> list[float]:
    vector = getattr(hit, 'vector', None)
    if isinstance(vector, dict):
        values = vector.get(DENSE_VECTOR_NAME)
        if isinstance(values, list):
            return [float(value) for value in values]
    elif isinstance(vector, list):
        return [float(value) for value in vector]
    return []


class RetrievalService:
    def __init__(
        self,
        qdrant: QdrantGateway,
        embeddings: EmbeddingProvider,
        sparse_embeddings: SparseEmbeddingProvider,
        *,
        hybrid_prefetch_limit: int,
        prompt_style: PromptStyle,
        dense_score_threshold: float,
        candidate_multiplier: int,
        bm25_language: str,
        query_cache_size: int,
        reranker: CrossEncoderReranker | None = None,
        rerank_score_threshold: float = 0.0,
    ) -> None:
        self.qdrant = qdrant
        self.embeddings = embeddings
        self.sparse_embeddings = sparse_embeddings
        self.hybrid_prefetch_limit = hybrid_prefetch_limit
        self.prompt_style = prompt_style
        self.dense_score_threshold = dense_score_threshold
        self.candidate_multiplier = max(1, candidate_multiplier)
        self.bm25_language = bm25_language
        self.reranker = reranker
        self.rerank_score_threshold = rerank_score_threshold
        self._query_vectors: LruCache[list[float]] = LruCache(query_cache_size)

    async def _dense_query_vector(self, question: str) -> list[float]:
        prompted = format_query(question, self.prompt_style)
        cached = self._query_vectors.get(prompted)
        if cached is not None:
            return cached

        vector = await self.embeddings.embed_query(prompted)
        self._query_vectors.put(prompted, vector)
        return vector

    async def retrieve(
        self,
        question: str,
        top_k: int,
        collection: str | None = None,
        language: str | None = None,
    ) -> list[RetrievalResult]:
        target = self.qdrant.resolve_collection(collection)
        # A persona may point at a collection that has not been filled yet.
        await self.qdrant.ensure_collection(target)

        bm25_language = language or self.bm25_language
        dense_vector = await self._dense_query_vector(question)
        sparse_vector = await self.sparse_embeddings.embed_query_async(
            question,
            bm25_language,
        )

        # Over-fetch so the rerank and dedup stages have something to choose
        # from; without it they can only reorder what fusion already picked.
        candidate_limit = top_k * self.candidate_multiplier
        hits = await self.qdrant.hybrid_search(
            collection=target,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            limit=candidate_limit,
            prefetch_limit=max(self.hybrid_prefetch_limit, candidate_limit),
            dense_score_threshold=self.dense_score_threshold,
        )

        # RRF only encodes rank, so a hit that BM25 liked and the dense branch
        # never saw would otherwise reach the prompt unchecked. Re-scoring every
        # candidate against the query vector gives one comparable relevance
        # number no matter which branch surfaced it.
        scored: list[RetrievalResult] = []
        rejected: list[float] = []
        for hit in hits:
            similarity = _cosine(dense_vector, _dense_vector_of(hit))
            if similarity < self.dense_score_threshold:
                rejected.append(similarity)
                continue
            scored.append({'score': similarity, 'payload': hit.payload or {}})

        scored.sort(key=lambda item: item['score'], reverse=True)
        results = scored

        logger.info(
            'retrieval | collection=%s bm25=%s candidates=%d kept=%d '
            'dropped=%d best_dropped=%.3f threshold=%.2f cache=%d/%d',
            target,
            bm25_language,
            len(hits),
            len(results),
            len(rejected),
            max(rejected) if rejected else 0.0,
            self.dense_score_threshold,
            self._query_vectors.hits,
            self._query_vectors.hits + self._query_vectors.misses,
        )

        if self.reranker is not None and results:
            results = await self._rerank(question, results)

        return results[:top_k]

    async def _rerank(
        self,
        question: str,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        documents = [str(result['payload'].get('text', '')) for result in results]

        try:
            scores = await self.reranker.score(question, documents)  # type: ignore[union-attr]
        except Exception:
            logger.exception('Reranking failed; falling back to fusion order')
            return results

        scored = [
            ({**result, 'score': score}, score)
            for result, score in zip(results, scores, strict=True)
        ]
        scored.sort(key=lambda item: item[1], reverse=True)

        kept = [
            result
            for result, score in scored
            if score >= self.rerank_score_threshold
        ]
        logger.info(
            'rerank | in=%d kept=%d best=%.3f threshold=%.3f',
            len(results),
            len(kept),
            scored[0][1] if scored else 0.0,
            self.rerank_score_threshold,
        )
        return kept
